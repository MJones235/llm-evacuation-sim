"""
Agent demographics utilities.

Provides PERSONALITY_TYPES dict and generate_random_demographics() helper used
by AgentFactory when building Concordia agents.

Note: The StationAgent class from the original NewcastleSim has been intentionally
omitted — it depended on the old scenarios.base framework which is not part of
evacusim.  Concordia-based agents are constructed via evacusim.concordia.agent_builder.
"""

import random

from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


# Myers-Briggs personality types with descriptions
PERSONALITY_TYPES = {
    "ISTJ": "Practical, fact-minded, reliable. Prefers order and clear procedures.",
    "ISFJ": "Caring, loyal, patient. Values harmony and helping others.",
    "INFJ": "Idealistic, insightful, principled. Seeks meaning and authenticity.",
    "INTJ": "Strategic, independent, analytical. Values competence and logic.",
    "ISTP": "Practical, observant, adaptable. Prefers hands-on problem-solving.",
    "ISFP": "Gentle, sensitive, spontaneous. Values personal freedom and aesthetics.",
    "INFP": "Idealistic, empathetic, reflective. Guided by personal values.",
    "INTP": "Analytical, curious, theoretical. Seeks logical understanding.",
    "ESTP": "Energetic, pragmatic, direct. Thrives on action and excitement.",
    "ESFP": "Enthusiastic, sociable, spontaneous. Lives in the moment.",
    "ENFP": "Enthusiastic, creative, expressive. Sees possibilities everywhere.",
    "ENTP": "Inventive, analytical, outspoken. Enjoys intellectual challenges.",
    "ESTJ": "Organized, decisive, practical. Values tradition and order.",
    "ESFJ": "Warm, cooperative, responsible. Values harmony and helping.",
    "ENFJ": "Charismatic, empathetic, inspiring. Focused on helping others grow.",
    "ENTJ": "Bold, strategic, decisive. Natural leader who values efficiency.",
}


def generate_random_demographics() -> dict:
    """Generate random age, gender, and personality type for an agent."""
    age = random.randint(18, 75)
    gender = random.choice(["man", "woman"])
    personality_type = random.choice(list(PERSONALITY_TYPES.keys()))
    return {
        "age": age,
        "gender": gender,
        "personality_type": personality_type,
        "personality_description": PERSONALITY_TYPES[personality_type],
    }
