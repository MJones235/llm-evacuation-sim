"""
Observation Coordinator

Coordinates the generation of observations for all agents by gathering information from:
- Agent positions and nearby agents (via state queries)
- Recent events (via event manager)
- Received messages and conversation history (via message system)
- Agent destinations (for enriching nearby agent info)

This module acts as the glue between multiple systems to create comprehensive
observations for agent decision-making.
"""

from typing import Any

from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


class ObservationCoordinator:
    """Coordinates observation generation for all agents."""

    def __init__(
        self,
        concordia_agents: dict[str, Any],
        exited_agents: set[str],
        observation_generator,
        state_queries,
        event_manager,
        message_system,
        agent_destinations: dict[str, str],
        agent_injured: set[str],
        agent_action: dict[str, str],
        agent_last_decision: dict[str, dict],
        test_scenarios: dict[str, Any],
        jps_sim=None,
    ):
        """
        Initialize observation coordinator.

        Args:
            concordia_agents: Dict of agent_id -> Concordia entity
            exited_agents: Set of agent IDs who have exited
            observation_generator: ObservationGenerator for formatting observations
            state_queries: Simulation state query interface
            event_manager: EventManager for accessing event history and blocked exits
            message_system: MessageSystem for agent messages and conversations
            agent_destinations: Dict of agent_id -> current exit name
            agent_injured: Set of injured agent IDs (physical capability dimension)
            agent_action: Dict of agent_id -> action ("moving"|"waiting")
            agent_last_decision: Dict of agent_id -> last translated_action (for memory)
            test_scenarios: Test scenario configuration for observation radius
            jps_sim: JuPedSim simulation (for multi-level support)
        """
        self.concordia_agents = concordia_agents
        self.exited_agents = exited_agents
        self.observation_generator = observation_generator
        self.state_queries = state_queries
        self.event_manager = event_manager
        self.message_system = message_system
        self.agent_destinations = agent_destinations
        self.agent_injured = agent_injured
        self.agent_action = agent_action
        self.agent_last_decision = agent_last_decision
        self.test_scenarios = test_scenarios
        self.jps_sim = jps_sim

    def generate_all_observations(self, current_sim_time: float) -> dict[str, str]:
        """
        Generate observations for all agents based on simulation state.

        Args:
            current_sim_time: Current simulation time in seconds

        Returns:
            Dict of agent_id -> observation string
        """
        observations = {}

        for agent_id in self.concordia_agents.keys():
            # Skip exited agents
            if agent_id in self.exited_agents:
                continue

            try:
                # Get agent state from JuPedSim
                position = self.state_queries.get_agent_position(agent_id)
                if position is None:
                    # Agent has exited, skip observation
                    continue

                # Get observation radius from config
                help_config = self.test_scenarios.get("help_behavior", {})
                observation_radius = help_config.get("observation_radius", 20.0)
                nearby_agents = self.state_queries.get_nearby_agents(
                    agent_id, radius=observation_radius
                )

                # Enrich nearby_agents with target exit info and follower detection
                for agent_info in nearby_agents:
                    other_id = agent_info.get("id")
                    if other_id:
                        agent_info["target_exit"] = self.agent_destinations.get(other_id)

                        # Detect if this nearby agent is following the current agent
                        is_following_me = False
                        if other_id in self.agent_last_decision:
                            last_decision = self.agent_last_decision[other_id]
                            if (
                                last_decision.get("target_type") == "agent"
                                and last_decision.get("target_agent") == agent_id
                            ):
                                is_following_me = True
                        agent_info["is_following_me"] = is_following_me

                # Get recent events
                recent_events = self.state_queries.get_recent_events(
                    self.event_manager.event_history, current_sim_time
                )

                # Get messages received by this agent
                received_messages = self.message_system.get_received_messages(agent_id)

                # Get conversation history for this agent
                conversation_history = self.message_system.get_conversation_history(agent_id)

                # Get agent's current level for multi-level simulations
                agent_level = None
                if self.jps_sim and hasattr(self.jps_sim, "agent_levels"):
                    agent_level = self.jps_sim.agent_levels.get(agent_id)

                # Generate observation
                obs = self.observation_generator.generate_observation(
                    agent_id=agent_id,
                    position=position,
                    nearby_agents=nearby_agents,
                    events=recent_events,
                    sim_time=current_sim_time,
                    blocked_exits=self.event_manager.blocked_exits,
                    agent_injured=self.agent_injured,
                    agent_action=self.agent_action,
                    agent_level=agent_level,
                    agent_last_decision=self.agent_last_decision,
                    state_queries=self.state_queries,
                    received_messages=received_messages,
                    conversation_history=conversation_history,
                )

                observations[agent_id] = obs

            except Exception as e:
                logger.error(f"Error generating observation for {agent_id}: {e}")
                observations[agent_id] = f"[Time: {current_sim_time:.1f}s] You are in the station."

        return observations
