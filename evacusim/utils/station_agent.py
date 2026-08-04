"""
Agent demographics utilities.

Provides OCEAN_ANCHORS dict and build_personality_anchor() / generate_random_demographics()
helpers used by AgentFactory when building Concordia agents.

Anchor sentences are grounded in Costa and McCrae (1992) NEO-PI-R facet definitions.
Only the three dimensions most relevant to emergency decision-making under the PADM
framework are used in the prompt: N (Anxiety/Vulnerability), O (Ideas/openness to
reinterpretation), and C (Deliberation/Dutifulness).

Note: The StationAgent class from the original NewcastleSim has been intentionally
omitted — it depended on the old scenarios.base framework which is not part of
evacusim.  Concordia-based agents are constructed via evacusim.concordia.agent_builder.
"""

import random

from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


# OCEAN anchor sentences per trait and level.
# Grounded in Costa & McCrae (1992) NEO-PI-R facet definitions.
# N: Anxiety (N1) and Vulnerability (N6) facets.
# O: Ideas (O5) facet — openness to reinterpreting the situation.
# C: Deliberation (C6) and Dutifulness (C3) facets.
OCEAN_ANCHORS: dict[str, dict[str, str]] = {
    "N": {
        "high":   "You tend to feel anxious and unsettled in uncertain or threatening "
                  "situations, and find it difficult to remain composed when facing potential danger.",
        "medium": "You experience some unease in uncertain situations but can generally "
                  "keep it in check under moderate pressure.",
        "low":    "You tend to remain calm and emotionally stable even in uncertain "
                  "or potentially threatening situations.",
    },
    "O": {
        "high":   "You tend to question your initial assumptions about what is happening "
                  "and are willing to consider that the situation may be different from what it first appears.",
        "medium": "You generally accept your initial reading of a situation, but will "
                  "reconsider if presented with clear evidence that it is wrong.",
        "low":    "You tend to take situations at face value and are unlikely to question "
                  "your initial interpretation of what is happening.",
    },
    "C": {
        "high":   "You prefer to think carefully before acting and feel a strong sense "
                  "of responsibility for doing the right thing.",
        "medium": "You generally consider your options before acting, though you can also "
                  "act on instinct when the situation seems to call for it.",
        "low":    "You tend to act on instinct rather than deliberating, and are less "
                  "concerned with following rules or procedures.",
    },
}

_OCEAN_LEVELS = ("high", "medium", "low")


def build_personality_anchor(n_level: str, o_level: str, c_level: str) -> str:
    """Concatenate OCEAN anchor sentences for N, O, and C in that order."""
    n = OCEAN_ANCHORS["N"].get(n_level, OCEAN_ANCHORS["N"]["medium"])
    o = OCEAN_ANCHORS["O"].get(o_level, OCEAN_ANCHORS["O"]["medium"])
    c = OCEAN_ANCHORS["C"].get(c_level, OCEAN_ANCHORS["C"]["medium"])
    return f"{n} {o} {c}"


def generate_random_demographics() -> dict:
    """Generate random age, gender, and OCEAN personality levels for an agent."""
    age = random.randint(18, 75)
    gender = random.choice(["man", "woman"])
    n_level = random.choice(_OCEAN_LEVELS)
    o_level = random.choice(_OCEAN_LEVELS)
    c_level = random.choice(_OCEAN_LEVELS)
    anchor = build_personality_anchor(n_level, o_level, c_level)
    return {
        "age": age,
        "gender": gender,
        "ocean_n": n_level,
        "ocean_o": o_level,
        "ocean_c": c_level,
        "personality_anchor": anchor,
        "personality_type": f"N:{n_level}/O:{o_level}/C:{c_level}",
    }
