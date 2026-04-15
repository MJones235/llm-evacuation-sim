"""
Azure OpenAI LLM adapter for Concordia.

This provides a synchronous interface compatible with Concordia's language model
expectations, calling Azure OpenAI API directly for general text completion.

Separate from scenarios.common.llm.azure_provider which is designed for
structured evacuation decision responses.
"""

import contextvars
import json
import os
from datetime import datetime
from pathlib import Path

import requests

from evacusim.utils.logger import get_logger

# Context variables set by the decision processor before each agent.act() call.
# asyncio.to_thread copies the active context, so these propagate correctly into
# the thread-pool workers that execute LLM requests.
llm_current_agent_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llm_current_agent_id", default=None
)
llm_current_sim_time: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "llm_current_sim_time", default=None
)

logger = get_logger(__name__)


class AzureLLMConcordia:
    """
    Azure OpenAI LLM adapter for Concordia.

    Provides synchronous text generation compatible with Concordia's
    sample_text() interface. Uses direct REST API calls to avoid
    async/sync event loop conflicts.
    """

    # Pricing per million tokens (in £) - same as AzureLLMProvider
    PRICE_INPUT_PER_M = 0.04
    PRICE_OUTPUT_PER_M = 0.30

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str | None = None,
        api_version: str = "2024-02-15-preview",
        temperature: float = 0.7,
        max_retries: int = 3,
        max_completion_tokens: int = 8000,
        timeout: float = 90.0,
    ):
        """
        Initialize Azure OpenAI client for Concordia.

        Args:
            endpoint: Azure OpenAI endpoint URL
            api_key: Azure OpenAI API key
            model: Deployment name (optional, extracted from endpoint if not provided)
            api_version: Azure API version
            temperature: Sampling temperature (0.0 to 2.0)
            max_retries: Maximum number of retry attempts on failure
            max_completion_tokens: Maximum tokens in completion
            timeout: Request timeout in seconds (default: 90s)
        """
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.api_version = api_version
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_completion_tokens = max_completion_tokens
        self.timeout = timeout

        # Token usage tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_requests = 0

        # Extract model/deployment name from endpoint if not provided
        if model:
            self.model = model
        else:
            # Extract from endpoint like: .../openai/deployments/gpt-4
            parts = self.endpoint.split("/deployments/")
            if len(parts) == 2:
                self.model = parts[1].split("/")[0]
            else:
                self.model = "gpt-4"  # Default fallback

        logger.info(
            f"Initialized AzureLLMConcordia with model: {self.model}, timeout: {self.timeout}s"
        )

    def sample_text(
        self, prompt: str, max_tokens: int | None = None, temperature: float | None = None, **kwargs
    ) -> str:
        """
        Generate text from a prompt.

        This is the primary interface method expected by Concordia.
        Uses synchronous REST API calls to avoid event loop conflicts.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (overrides default if provided)
            **kwargs: Additional parameters (for compatibility)

        Returns:
            Generated text string

        Raises:
            Exception: If all retry attempts fail
        """
        temp = temperature if temperature is not None else self.temperature
        if temp != 1:
            logger.warning(
                "Azure model only supports default temperature=1. Overriding temp=%s -> 1",
                temp,
            )
            temp = 1
        if max_tokens is None:
            max_tokens = self.max_completion_tokens
        else:
            max_tokens = max(max_tokens, self.max_completion_tokens)

        # Build the API URL
        url = f"{self.endpoint}/chat/completions?api-version={self.api_version}"

        # Build the request
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

        system_message = (
            "You are a simulation engine for everyday station scenarios. "
            "Generate realistic behavioral responses for simulated agents based on their personality profiles, "
            "situational context, and normal station routines. "
            "When a fire alarm is sounding with no clear visible fire or additional instructions from authorities, use this empirical prior for initial behavior: "
            "about 10% evacuate immediately, about 15% decide to leave but delay, and about 75% initially hesitate, "
            "wait for others, or ignore at first. "
            "Use this as a population-level prior while still adapting each individual response to local observations, "
            "social cues, and personal goals."
        )

        # Retry logic
        last_error = None
        base_prompt = prompt
        for attempt in range(1, self.max_retries + 1):
            try:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ]

                payload = {
                    "messages": messages,
                    "max_completion_tokens": max_tokens,  # Use max_completion_tokens for newer models
                    "temperature": temp,
                }

                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)

                if response.status_code == 200:
                    result = response.json()
                    choice = result["choices"][0]
                    text = choice.get("message", {}).get("content", "").strip()
                    finish_reason = choice.get("finish_reason")

                    self._log_prompt_response(
                        prompt, text, result.get("usage", {}), finish_reason, max_tokens
                    )

                    if not text:
                        logger.warning(
                            f"Empty response received (attempt {attempt}/{self.max_retries}, "
                            f"finish_reason={finish_reason}). Retrying with stricter instruction."
                        )
                        prompt = (
                            f"{base_prompt}\n\nIMPORTANT: Respond with 1-3 complete sentences. "
                            "Do not leave the answer blank."
                        )
                        last_error = Exception("Empty response")
                        continue

                    # Log and accumulate token usage
                    usage = result.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

                    self.total_prompt_tokens += prompt_tokens
                    self.total_completion_tokens += completion_tokens
                    self.total_tokens += total_tokens
                    self.total_requests += 1

                    logger.debug(
                        f"LLM call successful. Tokens: "
                        f"{prompt_tokens} prompt, "
                        f"{completion_tokens} completion"
                    )

                    return text

                else:
                    error_msg = f"Azure API error {response.status_code}: {response.text}"

                    # Check if it's a content filter / jailbreak error
                    if response.status_code == 400:
                        try:
                            if (
                                "content_filter" in response.text
                                or "jailbreak" in response.text.lower()
                            ):
                                logger.error("=" * 80)
                                logger.error("CONTENT FILTER / JAILBREAK DETECTED")
                                logger.error("=" * 80)
                                logger.error(f"Error: {error_msg}")
                                logger.error("\n" + "=" * 80)
                                logger.error("FULL PROMPT THAT TRIGGERED THE FILTER:")
                                logger.error("=" * 80)
                                logger.error(f"System message:\n{messages[0]['content']}")
                                logger.error("\n" + "-" * 80)
                                logger.error(f"User prompt:\n{messages[1]['content']}")
                                logger.error("=" * 80)
                        except Exception:
                            pass

                    logger.warning(f"Attempt {attempt}/{self.max_retries} failed: {error_msg}")
                    last_error = Exception(error_msg)

            except requests.exceptions.Timeout as e:
                logger.warning(f"Attempt {attempt}/{self.max_retries} timed out")
                last_error = e

            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt}/{self.max_retries} failed: {e}")
                last_error = e

            except (KeyError, IndexError, json.JSONDecodeError) as e:
                logger.error(f"Failed to parse Azure response: {e}")
                last_error = e

        # All retries failed
        error_msg = f"Failed after {self.max_retries} attempts. Last error: {last_error}"
        logger.error(error_msg)

        # Return a fallback response rather than crashing the simulation
        logger.warning("Returning fallback response due to API failures")
        return "No clear information available."

    def get_usage_stats(self) -> dict:
        """
        Get cumulative token usage statistics with cost estimates.

        Returns:
            Dict with prompt_tokens, completion_tokens, total_tokens, total_requests,
            and estimated_cost_gbp
        """
        # Calculate cost in £
        input_cost = (self.total_prompt_tokens / 1_000_000) * self.PRICE_INPUT_PER_M
        output_cost = (self.total_completion_tokens / 1_000_000) * self.PRICE_OUTPUT_PER_M
        total_cost = input_cost + output_cost

        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_requests": self.total_requests,
            "estimated_cost_gbp": total_cost,
            "input_cost_gbp": input_cost,
            "output_cost_gbp": output_cost,
        }

    def _log_prompt_response(
        self,
        prompt: str,
        response: str,
        usage: dict,
        finish_reason: str | None,
        max_completion_tokens: int,
    ) -> None:
        """Log the full prompt and response for debugging."""
        try:
            env_path = os.getenv("CONCORDIA_LLM_LOG_PATH")
            log_path = (
                Path(env_path)
                if env_path
                else Path("scenarios/station_concordia/output/llm_prompt_log.jsonl")
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "agent_id": llm_current_agent_id.get(),
                "sim_time": llm_current_sim_time.get(),
                "model": self.model,
                "prompt": prompt,
                "response": response,
                "finish_reason": finish_reason,
                "max_completion_tokens": max_completion_tokens,
                "usage": usage,
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as e:
            logger.debug(f"Failed to log LLM prompt/response: {e}")

    @classmethod
    def from_env(cls, **kwargs) -> "AzureLLMConcordia":
        """
        Create instance from environment variables.

        Expects:
            - AZURE_LLM_ENDPOINT
            - AZURE_LLM_API_KEY
            - AZURE_LLM_MODEL (optional)

        Args:
            **kwargs: Additional arguments passed to constructor

        Returns:
            AzureLLMConcordia instance

        Raises:
            ValueError: If required environment variables are missing
        """
        endpoint = os.getenv("AZURE_LLM_ENDPOINT")
        api_key = os.getenv("AZURE_LLM_API_KEY")
        model = os.getenv("AZURE_LLM_MODEL")

        if not endpoint or not api_key:
            raise ValueError(
                "Missing required environment variables: "
                "AZURE_LLM_ENDPOINT and AZURE_LLM_API_KEY"
            )

        return cls(endpoint=endpoint, api_key=api_key, model=model, **kwargs)


def create_concordia_llm_from_config(config: dict) -> AzureLLMConcordia:
    """
    Create Azure LLM instance from configuration.

    Args:
        config: Configuration dictionary with llm settings

    Returns:
        AzureLLMConcordia instance
    """
    from dotenv import load_dotenv

    load_dotenv()

    llm_config = config.get("llm", {})

    return AzureLLMConcordia.from_env(
        temperature=llm_config.get("temperature", 0.7),
        max_retries=llm_config.get("max_retries", 3),
    )
