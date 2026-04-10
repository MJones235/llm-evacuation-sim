"""
Unified station agent that works with any movement simulator.
Separates decision-making from movement implementation.
"""

import random
import sys
from pathlib import Path
from typing import Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from base.agent_base import AgentBase
from base.decision_maker_base import Decision, DecisionMakerBase
from base.movement_provider import MovementProvider

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
    gender = random.choice(["male", "female"])
    personality_type = random.choice(list(PERSONALITY_TYPES.keys()))

    return {
        "age": age,
        "gender": gender,
        "personality_type": personality_type,
        "personality_description": PERSONALITY_TYPES[personality_type],
    }


class StationAgent(AgentBase):
    """
    Unified agent for station simulations.
    Works with any simulator (SUMO, JuPedSim, etc.) via MovementProvider.
    """

    def __init__(
        self,
        agent_id: str,
        walking_speed: float,
        decision_maker: DecisionMakerBase,
        movement_provider: MovementProvider,
        initial_zone: str,
        destination: str,
        spawn_params: Optional[dict[str, Any]] = None,
        demographics: Optional[dict] = None,
    ):
        """
        Initialize a station agent.

        Args:
            agent_id: Unique agent identifier
            walking_speed: Preferred walking speed in m/s
            decision_maker: Decision making module
            movement_provider: Handles simulator-specific movement
            initial_zone: Starting zone/location name
            destination: Target destination name
            spawn_params: Simulator-specific spawn parameters
            demographics: Optional demographic information (age, gender, personality)
        """
        # Generate demographics if not provided
        if demographics is None:
            demographics = generate_random_demographics()

        super().__init__(agent_id, demographics)

        # Store demographics for easy access
        self.age = demographics.get("age", 30)
        self.gender = demographics.get("gender", "unknown")
        self.personality_type = demographics.get("personality_type", "ISTJ")
        self.personality_description = demographics.get("personality_description", "")

        # Movement properties
        self.walking_speed = walking_speed
        self.movement_provider = movement_provider
        self.spawn_params = spawn_params or {}

        # Location
        self.initial_zone = initial_zone
        self.destination = destination
        self.position = (0.0, 0.0)

        # Decision making
        self.decision_maker = decision_maker
        self.current_decision: Optional[Decision] = None

        # State
        self.is_spawned = False
        self.is_evacuating = False
        self.is_active = True
        self._spawn_time = None  # Track when agent spawned
        self._last_message_index = -1

        # For tracking evacuation
        self.original_destination = destination
        self.evacuation_target = None

        # For agent-to-agent communication
        self._get_all_agents_callback = None  # Set by simulation manager

    def spawn(self) -> bool:
        """
        Spawn this agent in the simulator.

        Returns:
            True if spawn successful
        """
        if self.is_spawned:
            return True

        self.is_spawned = self.movement_provider.spawn_agent(self, self.spawn_params)
        if self.is_spawned:
            self.is_active = True
            import time

            self._spawn_time = time.time()  # Track spawn time
            # Increment trip_starts counter if diagnostics available
            if self.diagnostics:
                self.diagnostics.trip_starts += 1
        return self.is_spawned

    def update(self, sim_time: int):
        """
        Update agent state each simulation step.

        Args:
            sim_time: Current simulation time in seconds
        """
        if not self.is_spawned or not self.is_active:
            return

        # Update position from simulator
        self.position = self.movement_provider.update_agent_position(self)

        # Check if still active
        # Skip active check on spawn timestep - person may not be in SUMO list yet
        import time

        time_since_spawn = time.time() - self._spawn_time if self._spawn_time else 999
        if time_since_spawn > 0.05:  # Allow 50ms grace period after spawn
            was_active = self.is_active
            self.is_active = self.movement_provider.is_agent_active(self)

            # Track completion
            if was_active and not self.is_active:
                if self.diagnostics:
                    self.diagnostics.trip_completions += 1

        # Process any new messages
        self._process_messages()

    def _process_messages(self):
        """Process new messages and make decisions."""
        # Only process new messages
        new_messages = self.messages[self._last_message_index + 1 :]

        if not new_messages:
            return

        # Update message index
        self._last_message_index = len(self.messages) - 1

        # Get current state for decision making
        agent_state = {
            "position": self.position,
            "destination": self.destination,
            "is_evacuating": self.is_evacuating,
            "walking_speed": self.walking_speed,
            "zone": self.get_current_zone(),
        }

        # Get simulation context
        context = {
            "time": 0,  # Will be set by manager
            "location_info": self.movement_provider.get_agent_location_info(self),
        }

        # Make decision for most recent message
        latest_message = new_messages[-1]
        decision = self.decision_maker.make_decision(latest_message, agent_state, context)

        if decision == Decision.EVACUATE and not self.is_evacuating:
            self._start_evacuation()

    def start_evacuation(self):
        """
        Public method to initiate evacuation (called by LLM decision system).
        """
        if not self.is_evacuating:
            self._start_evacuation()

    def _start_evacuation(self):
        """Initiate evacuation behavior."""
        import logging

        logger = logging.getLogger(__name__)

        logger.info(f"Agent {self.id} deciding to evacuate")
        self.is_evacuating = True
        self.evacuation_target = self._select_evacuation_exit()

        # Request reroute to evacuation exit
        if self.evacuation_target:
            success = self.movement_provider.reroute_to_evacuation_exit(
                self, self.evacuation_target
            )
            if success:
                logger.info(f"  → Agent {self.id} rerouted to {self.evacuation_target}")
            else:
                logger.warning(f"  ✗ Agent {self.id} failed to reroute to {self.evacuation_target}")

    def _select_evacuation_exit(self) -> str:
        """
        Select closest evacuation exit based on current position.

        Returns:
            Name of evacuation exit to target
        """
        # Get available evacuation exits from movement provider
        if hasattr(self.movement_provider, "evacuation_exits"):
            exit_names = list(self.movement_provider.evacuation_exits.keys())
            if exit_names:
                # For now, select randomly
                # Could be improved to select nearest exit based on position
                import random

                return random.choice(exit_names)

        # Fallback to initial zone (entrance)
        return self.initial_zone

    def get_current_location(self) -> tuple[float, float]:
        """Get agent's current (x, y) position."""
        return self.position

    def get_current_zone(self) -> str:
        """Get agent's current zone/area name."""
        location_info = self.movement_provider.get_agent_location_info(self)
        return location_info.get("zone", "unknown")

    def send_message_to_nearby(self, message: str, radius: float) -> int:
        """
        Send a message to all agents within a given radius.

        Args:
            message: Message to broadcast
            radius: Distance in meters within which agents will receive the message

        Returns:
            Number of agents who received the message
        """
        if self._get_all_agents_callback is None:
            logger.warning(f"Agent {self.id} cannot send messages - callback not set")
            return 0

        if not self.is_spawned:
            logger.debug(f"Agent {self.id} cannot send messages - not spawned")
            return 0

        all_agents = self._get_all_agents_callback()
        recipients = 0

        # Calculate distance to each agent and send message if within radius
        for other_agent in all_agents:
            if other_agent.id == self.id:
                continue  # Don't send to self

            if not other_agent.is_spawned:
                continue  # Skip agents not yet spawned

            # Calculate Euclidean distance
            dx = other_agent.position[0] - self.position[0]
            dy = other_agent.position[1] - self.position[1]
            distance = (dx**2 + dy**2) ** 0.5

            if distance <= radius:
                # Format message with sender information
                formatted_message = f"[Agent {self.id}]: {message}"
                other_agent.receive_message(formatted_message)
                recipients += 1

        logger.info(
            f"Agent {self.id} sent message to {recipients} nearby agents (radius={radius}m)"
        )
        return recipients

    def send_message_to_agent(self, target_agent: "StationAgent", message: str) -> bool:
        """
        Send a message to a specific agent.

        Args:
            target_agent: The agent to send the message to
            message: Message to send

        Returns:
            True if message was delivered, False otherwise
        """
        if not self.is_spawned:
            logger.debug(f"Agent {self.id} cannot send messages - not spawned")
            return False

        if target_agent.id == self.id:
            logger.debug(f"Agent {self.id} cannot send message to self")
            return False

        if not target_agent.is_spawned:
            logger.debug(f"Agent {self.id} cannot send to {target_agent.id} - target not spawned")
            return False

        # Format message with sender information
        formatted_message = f"[Agent {self.id}]: {message}"
        target_agent.receive_message(formatted_message)

        logger.info(f"Agent {self.id} sent message to agent {target_agent.id}")
        return True

    def set_destination(self, new_destination: str, target: Any = None) -> bool:
        """
        Set a new destination for the agent.

        Args:
            new_destination: New destination name
            target: Simulator-specific target identifier

        Returns:
            True if destination updated successfully
        """
        if target is not None:
            success = self.movement_provider.set_agent_target(self, target)
            if success:
                self.destination = new_destination
                return True
        else:
            self.destination = new_destination
            return True
        return False

    def is_schedule_complete(self) -> bool:
        """Check if agent has completed their journey."""
        return not self.is_active

    def __repr__(self):
        return f"StationAgent(id={self.id}, zone={self.initial_zone}, dest={self.destination})"
