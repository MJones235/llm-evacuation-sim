"""
Blablador (Helmholtz / FZ-Jülich) LLM adapter for Concordia.

Blablador is a free, OpenAI-compatible inference service
(https://api.helmholtz-blablador.fz-juelich.de/v1/). This adapter mirrors the
Azure/Claude adapters: a synchronous, duck-typed language model exposing the
sample_text() interface Concordia expects, plus get_usage_stats() for the
financial reporter. Uses direct REST calls (no SDK) so it carries no new
dependency; because the API is OpenAI-compatible, pointing base_url elsewhere
(OpenRouter, a local vLLM/Ollama) reuses this class unchanged.
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

DEFAULT_BASE_URL = "https://api.helmholtz-blablador.fz-juelich.de/v1/"
DEFAULT_MODEL = "alias-large"

# Same population-level prior and framing as the other adapters, so agent
# behaviour is comparable across providers.
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


class BlabladorLLMConcordia:
    """
    OpenAI-compatible LLM adapter for Concordia, defaulting to Blablador.

    Provides synchronous text generation compatible with Concordia's sample_text()
    interface via direct REST API calls to a `/chat/completions` endpoint.
    """

    # Blablador is free → zero cost. Override if you point base_url at a paid
    # OpenAI-compatible endpoint; only affects financial_report.txt figures.
    PRICE_INPUT_PER_M = 0.0
    PRICE_OUTPUT_PER_M = 0.0

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.7,
        max_retries: int = 3,
        max_completion_tokens: int = 8000,
        timeout: float = 90.0,
        disable_thinking: bool = True,
    ):
        """
        Initialize the Blablador / OpenAI-compatible client for Concordia.

        Args:
            api_key: Bearer API key.
            model: Model id (e.g. "alias-large", "alias-fast", "alias-qwen-huge").
            base_url: OpenAI-compatible base URL ending in /v1/.
            temperature: Sampling temperature.
            max_retries: Maximum retry attempts on failure.
            max_completion_tokens: Maximum tokens in completion.
            timeout: Request timeout in seconds.
            disable_thinking: When True, append the Qwen3 ``/no_think`` switch to
                the system message so reasoning models answer directly instead of
                emitting (and being billed/timed-out on) a long hidden chain of
                thought. Harmless plain text for non-Qwen models.
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_completion_tokens = max_completion_tokens
        self.timeout = timeout
        self.disable_thinking = disable_thinking
        self._system_message = (
            f"{_SYSTEM_MESSAGE} /no_think" if disable_thinking else _SYSTEM_MESSAGE
        )

        # Token usage tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_requests = 0

        logger.info(
            f"Initialized BlabladorLLMConcordia with model: {self.model}, "
            f"base_url: {self.base_url}, timeout: {self.timeout}s"
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
        # max_tokens is an upper *cap*, not a target — the model stops at
        # finish_reason="stop" when it is done. Concordia components hardcode
        # small caps; enforce a floor so a reasoning model (e.g. Qwen3) is never
        # truncated mid-thought (which yields an empty, finish_reason="length"
        # response). For Qwen3 we also disable thinking via /no_think below.
        if max_tokens is None:
            max_tokens = self.max_completion_tokens
        else:
            max_tokens = max(max_tokens, self.max_completion_tokens)

        temp = self.temperature if temperature is None else temperature
        req_timeout = self.timeout if timeout is None else timeout

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_error = None
        base_prompt = prompt
        for attempt in range(1, self.max_retries + 1):
            try:
                payload = {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": temp,
                    "messages": [
                        {"role": "system", "content": self._system_message},
                        {"role": "user", "content": prompt},
                    ],
                }
                stop_sequences = [t for t in terminators if t]
                if stop_sequences:
                    payload["stop"] = stop_sequences

                response = requests.post(
                    url, headers=headers, json=payload, timeout=req_timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    choice = result["choices"][0]
                    text = (choice.get("message", {}).get("content") or "").strip()
                    finish_reason = choice.get("finish_reason")
                    usage = result.get("usage", {})

                    self._log_prompt_response(prompt, text, usage, finish_reason, max_tokens)

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

                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
                    self.total_prompt_tokens += prompt_tokens
                    self.total_completion_tokens += completion_tokens
                    self.total_tokens += total_tokens
                    self.total_requests += 1

                    logger.debug(
                        f"LLM call successful. Tokens: "
                        f"{prompt_tokens} prompt, {completion_tokens} completion"
                    )
                    return text

                # Non-200: 429 / 5xx are retryable; 4xx auth/bad-request are not.
                error_msg = f"Blablador API error {response.status_code}: {response.text}"
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
                logger.error(f"Failed to parse Blablador response: {e}")
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
    def from_env(cls, **kwargs) -> "BlabladorLLMConcordia":
        """
        Create an instance from environment variables.

        Expects:
            - BLABLADOR_API_KEY
            - BLABLADOR_MODEL (optional; defaults to alias-large)
            - BLABLADOR_BASE_URL (optional; defaults to the Helmholtz endpoint)
            - BLABLADOR_DISABLE_THINKING (optional; defaults to true)
        """
        api_key = os.getenv("BLABLADOR_API_KEY")
        model = os.getenv("BLABLADOR_MODEL", DEFAULT_MODEL)
        base_url = os.getenv("BLABLADOR_BASE_URL", DEFAULT_BASE_URL)
        disable_thinking = os.getenv("BLABLADOR_DISABLE_THINKING", "true").strip().lower() in (
            "1", "true", "yes", "on"
        )
        if not api_key:
            raise ValueError("Missing required environment variable: BLABLADOR_API_KEY")
        return cls(
            api_key=api_key,
            model=model,
            base_url=base_url,
            disable_thinking=disable_thinking,
            **kwargs,
        )
