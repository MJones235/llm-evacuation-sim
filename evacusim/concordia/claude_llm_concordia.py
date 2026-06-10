"""
Anthropic Claude LLM adapter for Concordia.

Mirrors AzureLLMConcordia: a synchronous, duck-typed language model exposing the
sample_text() interface Concordia expects, plus get_usage_stats() for the
financial reporter. Uses direct REST calls to the Anthropic Messages API (no SDK
dependency) to match the Azure adapter's request/retry shape.

Unlike Azure, Claude has no response_format=json_object switch; JSON reliability
comes from the decision prompt itself ("Respond with ONLY this JSON") plus the
ActionTranslator's prefix-strip and wait-fallback parsing.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import requests

from evacusim.utils.logger import get_logger

# Shared with the Azure adapter's contextvars so the prompt logger can attribute
# each call to an agent/sim-time regardless of which provider is active.
from evacusim.concordia.azure_llm_concordia import (
    llm_current_agent_id,
    llm_current_sim_time,
)

logger = get_logger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# Same population-level prior and framing as the Azure adapter, so agent behaviour
# is comparable across providers.
_SYSTEM_MESSAGE = (
    "You are a simulation engine for everyday station scenarios. "
    "Generate realistic behavioral responses for simulated agents based on their personality profiles, "
    "situational context, and normal station routines. "
    "When a fire alarm is sounding with no clear visible fire or additional instructions from authorities, use this empirical prior for initial behavior: "
    "about 10% evacuate immediately, about 15% decide to leave but delay, and about 75% initially hesitate, "
    "wait for others, or ignore at first. "
    "Use this as a population-level prior while still adapting each individual response to local observations, "
    "social cues, and personal goals."
)


class ClaudeLLMConcordia:
    """
    Anthropic Claude LLM adapter for Concordia.

    Provides synchronous text generation compatible with Concordia's sample_text()
    interface via direct REST API calls.
    """

    # Approximate pricing per million tokens (in £) for the default model.
    # Defaults are for Claude Haiku 4.5 (~$1 in / $5 out per MTok ≈ £0.80 / £4.0).
    # Adjust if you switch models; this only affects the cost figures in
    # financial_report.txt, not the simulation.
    PRICE_INPUT_PER_M = 0.80
    PRICE_OUTPUT_PER_M = 4.00

    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5",
        temperature: float = 0.7,
        max_retries: int = 3,
        max_completion_tokens: int = 8000,
        timeout: float = 90.0,
    ):
        """
        Initialize the Claude client for Concordia.

        Args:
            api_key: Anthropic API key.
            model: Claude model id (e.g. "claude-haiku-4-5", "claude-sonnet-4-6").
            temperature: Sampling temperature (clamped to [0, 1] for Anthropic).
            max_retries: Maximum retry attempts on failure.
            max_completion_tokens: Maximum tokens in completion.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key
        self.model = model
        self.temperature = max(0.0, min(1.0, temperature))
        self.max_retries = max_retries
        self.max_completion_tokens = max_completion_tokens
        self.timeout = timeout

        # Token usage tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_requests = 0

        logger.info(
            f"Initialized ClaudeLLMConcordia with model: {self.model}, timeout: {self.timeout}s"
        )

    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        terminators=(),
        temperature: float | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> str:
        """
        Generate text from a prompt (primary Concordia interface method).

        Extra Concordia kwargs (top_p, top_k, seed) are accepted and ignored for
        cross-provider compatibility.
        """
        if max_tokens is None:
            max_tokens = self.max_completion_tokens
        else:
            # Concordia components hardcode small max_tokens; never let an upstream
            # default starve the budget for a full JSON decision response.
            max_tokens = max(max_tokens, self.max_completion_tokens)

        temp = self.temperature if temperature is None else max(0.0, min(1.0, temperature))
        req_timeout = self.timeout if timeout is None else timeout

        headers = {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }

        last_error = None
        base_prompt = prompt
        for attempt in range(1, self.max_retries + 1):
            try:
                payload = {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": temp,
                    "system": _SYSTEM_MESSAGE,
                    "messages": [{"role": "user", "content": prompt}],
                }
                stop_sequences = [t for t in terminators if t]
                if stop_sequences:
                    payload["stop_sequences"] = stop_sequences

                response = requests.post(
                    ANTHROPIC_API_URL, headers=headers, json=payload, timeout=req_timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    text = "".join(
                        block.get("text", "")
                        for block in result.get("content", [])
                        if block.get("type") == "text"
                    ).strip()
                    finish_reason = result.get("stop_reason")
                    usage = result.get("usage", {})

                    self._log_prompt_response(prompt, text, usage, finish_reason, max_tokens)

                    if not text:
                        logger.warning(
                            f"Empty response received (attempt {attempt}/{self.max_retries}, "
                            f"stop_reason={finish_reason}). Retrying with stricter instruction."
                        )
                        prompt = (
                            f"{base_prompt}\n\nIMPORTANT: Respond with 1-3 complete sentences. "
                            "Do not leave the answer blank."
                        )
                        last_error = Exception("Empty response")
                        continue

                    prompt_tokens = usage.get("input_tokens", 0)
                    completion_tokens = usage.get("output_tokens", 0)
                    self.total_prompt_tokens += prompt_tokens
                    self.total_completion_tokens += completion_tokens
                    self.total_tokens += prompt_tokens + completion_tokens
                    self.total_requests += 1

                    logger.debug(
                        f"LLM call successful. Tokens: "
                        f"{prompt_tokens} prompt, {completion_tokens} completion"
                    )
                    return text

                # Non-200: 429 / 529 (overloaded) are retryable; 4xx auth/bad-request are not.
                error_msg = f"Anthropic API error {response.status_code}: {response.text}"
                if response.status_code in (400, 401, 403):
                    logger.error(error_msg)
                    last_error = Exception(error_msg)
                    break
                logger.warning(f"Attempt {attempt}/{self.max_retries} failed: {error_msg}")
                last_error = Exception(error_msg)

            except requests.exceptions.Timeout as e:
                logger.warning(f"Attempt {attempt}/{self.max_retries} timed out")
                last_error = e
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt}/{self.max_retries} failed: {e}")
                last_error = e
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                logger.error(f"Failed to parse Anthropic response: {e}")
                last_error = e

        error_msg = f"Failed after {self.max_retries} attempts. Last error: {last_error}"
        logger.error(error_msg)
        logger.warning("Returning fallback response due to API failures")
        return "No clear information available."

    def sample_choice(self, prompt: str, responses, *, seed: int | None = None):
        """
        Pick one of ``responses`` (Concordia's multiple-choice interface).

        Rarely exercised on the FREE-output decision path, but implemented for
        completeness: asks the model for the best index and falls back to 0.
        """
        options = list(responses)
        numbered = "\n".join(f"{i}: {r}" for i, r in enumerate(options))
        choice_prompt = (
            f"{prompt}\n\nChoose the single best option by number only:\n{numbered}\n\n"
            "Answer with just the number."
        )
        answer = self.sample_text(choice_prompt, max_tokens=16)
        idx = 0
        for token in answer.replace(".", " ").split():
            if token.isdigit() and int(token) < len(options):
                idx = int(token)
                break
        return idx, options[idx], {}

    def get_usage_stats(self) -> dict:
        """Cumulative token usage with cost estimates (same shape as Azure adapter)."""
        input_cost = (self.total_prompt_tokens / 1_000_000) * self.PRICE_INPUT_PER_M
        output_cost = (self.total_completion_tokens / 1_000_000) * self.PRICE_OUTPUT_PER_M
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_requests": self.total_requests,
            "estimated_cost_gbp": input_cost + output_cost,
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
        """Append the full prompt/response to the per-run JSONL log."""
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
    def from_env(cls, **kwargs) -> "ClaudeLLMConcordia":
        """
        Create an instance from environment variables.

        Expects:
            - ANTHROPIC_API_KEY
            - ANTHROPIC_MODEL (optional; defaults to claude-haiku-4-5)
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
        if not api_key:
            raise ValueError("Missing required environment variable: ANTHROPIC_API_KEY")
        return cls(api_key=api_key, model=model, **kwargs)
