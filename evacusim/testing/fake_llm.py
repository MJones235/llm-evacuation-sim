"""A deterministic stand-in for the language model and embedder.

Runs the whole LLM decision pipeline (prompt rendering, Concordia agents,
prompt cache, schema repair, fallback) without network calls or cost, so the
pipeline can be regression-tested and dry-run. Not a model of behaviour:
choices are a hash of the prompt, not reasoning.

The fake reads the offered actions from the prompt's "Choose exactly one
action from this offered set:" block and answers with a valid decision.
One in five first-attempt prompts gets a malformed answer, to exercise the
repair path (a repair prompt, which carries a "SYSTEM NOTE", always gets a
valid one). The model is stateless, so answers do not depend on call order. Prompts and responses are logged like the Azure client's
(``CONCORDIA_LLM_LOG_PATH``), minus timestamps, so logs are reproducible.
"""

from __future__ import annotations

import json
import os
import re
import threading
import zlib
from pathlib import Path

import numpy as np

from evacusim.concordia.azure_llm_concordia import llm_current_agent_id, llm_current_sim_time

_OFFERED_LINE = re.compile(r"^- (\w+)(?: \(required: (\w+) (?:in|from) \{([^}]*)\}\))?", re.M)
_PACES = ("normal_pace", "hurrying", "running")
_ASSESSMENT = (
    "source_credibility",
    "situation_appraisal",
    "personal_relevance",
    "options_considered",
    "option_chosen_because",
    "information_gap",
)


def _choose(options: list[str], key: int) -> str:
    return options[key % len(options)]


class FakeLanguageModel:
    """Concordia-compatible model that answers decision prompts deterministically."""

    def __init__(self) -> None:
        self.total_requests = 0
        self._lock = threading.Lock()

    def sample_text(self, prompt: str, *args, **kwargs) -> str:
        key = zlib.crc32(prompt.encode())
        with self._lock:
            self.total_requests += 1
        malformed = key % 5 == 0 and "SYSTEM NOTE" not in prompt
        response = "I am not sure what to do." if malformed else self._decide(prompt, key)
        self._log(prompt, response)
        return response

    def _decide(self, prompt: str, key: int) -> str:
        block = prompt.rsplit("Choose exactly one action from this offered set:", 1)[-1]
        block = block.split("\n\n", 1)[0]  # the offered set ends at the first blank line
        offered = {m[0]: [o.strip() for o in m[2].split(",")] for m in _OFFERED_LINE.findall(block)}
        if not offered:
            return json.dumps({"action": "wait", "wait_reason": "awaiting_information"})
        action = _choose(sorted(offered), key)
        payload = {
            "assessment": {field: f"fake model, choice {key % 1000}" for field in _ASSESSMENT},
            "action": action,
            "wait_reason": _choose(offered[action], key >> 4) if action == "wait" else None,
            "exit_id": _choose(offered[action], key >> 8) if action == "evacuate" else None,
            "pace": None if action == "wait" else _choose(list(_PACES), key >> 12),
            "reassess_when": "next_interval",
        }
        return json.dumps(payload)

    def sample_choice(self, prompt: str, responses, *args, **kwargs):
        index = zlib.crc32(prompt.encode()) % len(responses)
        return index, responses[index], {}

    def get_usage_stats(self) -> dict:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_requests": self.total_requests,
            "estimated_cost_gbp": 0.0,
            "input_cost_gbp": 0.0,
            "output_cost_gbp": 0.0,
        }

    def _log(self, prompt: str, response: str) -> None:
        path = os.getenv("CONCORDIA_LLM_LOG_PATH")
        if not path:
            return
        record = {
            "agent_id": llm_current_agent_id.get(),
            "sim_time": llm_current_sim_time.get(),
            "model": "fake",
            "prompt": prompt,
            "response": response,
        }
        with self._lock, Path(path).open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def fake_embedder(text: str) -> np.ndarray:
    """A deterministic 16-dimensional embedding of ``text``."""
    rng = np.random.default_rng(zlib.crc32(text.encode()))
    return rng.standard_normal(16)
