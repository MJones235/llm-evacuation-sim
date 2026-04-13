"""
Agent manager for Station Concordia simulations.

This module is responsible for:
- Complete agent lifecycle management
- Creating agent configurations
- Generating spawn positions
- Adding agents to the pedestrian simulation with appropriate speeds
- Coordinating all agent-related operations
"""

import random
from typing import Any

from evacusim.utils.logger import get_logger
from evacusim.utils.walking_speed import sample_walking_speed
from evacusim.jps.simulation_interface import PedestrianSimulation
from evacusim.setup.agent_factory import AgentFactory
from evacusim.setup.spawn_manager import SpawnManager

logger = get_logger(__name__)


class AgentManager:
    """Handles complete agent lifecycle management."""

    @staticmethod
    def create_and_populate_agents(
        jps_sim: PedestrianSimulation, config: dict
    ) -> list[dict[str, Any]]:
        """
        Create agents and add them to the pedestrian simulation.

        This is the main entry point for all agent-related operations.
        It handles:
        - Determining how many agents to create
        - Generating spawn positions
        - Creating agent configurations
        - Adding agents to the simulation with appropriate walking speeds

        Args:
            jps_sim: Pedestrian simulation instance (implements PedestrianSimulation)
            config: Configuration dictionary

        Returns:
            List of agent configuration dictionaries
        """
        # Determine number of agents
        agent_config = config.get("agents", {})
        num_agents = agent_config.get("count", 1)

        # Generate spawn positions
        spawn_positions = SpawnManager.generate_spawn_positions(jps_sim, num_agents)

        # Warn and cap if fewer positions were generated than requested
        # (e.g. some rejected because they fell inside escalator corridors).
        actual_count = len(spawn_positions)
        if actual_count < num_agents:
            logger.warning(
                f"Only {actual_count} spawn positions available for {num_agents} requested agents. "
                f"Reducing agent count to {actual_count}."
            )
            num_agents = actual_count

        # Create agent configurations
        agents_config, injured_agents = AgentFactory.create_agents(num_agents, config)

        # Add agents to JuPedSim
        AgentManager._add_agents_to_jupedsim(
            jps_sim, agents_config, spawn_positions, injured_agents, config
        )

        logger.info(f"Agent population complete: {num_agents} agents ready")
        return agents_config

    @staticmethod
    def _add_agents_to_jupedsim(jps_sim, agents_config, spawn_positions, injured_agents, config):
        """
        Add agents to JuPedSim simulation with appropriate walking speeds.

        Args:
            jps_sim: JuPedSim simulation instance
            agents_config: List of agent configuration dictionaries
            spawn_positions: List of spawn position tuples (x, y) or (x, y, level_id)
            injured_agents: Set of injured agent indices
            config: Configuration dictionary
        """
        for i, agent_cfg in enumerate(agents_config):
            agent_id = agent_cfg["id"]
            spawn_data = spawn_positions[i]

            # Handle both (x, y) and (x, y, level_id) formats
            if len(spawn_data) == 3:
                # Multi-level: (x, y, level_id)
                x, y, level_id = spawn_data
                start_pos = (x, y)
            else:
                # Single-level: (x, y)
                start_pos = spawn_data
                level_id = "0"  # Default level

            # Store level_id in agent config for use during agent initialization
            agent_cfg["level_id"] = level_id
            agent_cfg["start_position"] = start_pos
            # Assign initial_zone from coordinate-based rules defined in
            # config under station.zone_boundaries.
            zone_boundaries = config.get("station", {}).get("zone_boundaries", {})
            level_zones = zone_boundaries.get(str(level_id), {})
            px, py = float(start_pos[0]), float(start_pos[1])
            assigned_zone = None
            default_zone = None
            for zone_name, conditions in level_zones.items():
                if conditions.get("default", False):
                    default_zone = zone_name
                    continue
                x_lt = conditions.get("x_lt")
                x_gt = conditions.get("x_gt")
                y_lt = conditions.get("y_lt")
                y_gt = conditions.get("y_gt")
                if (
                    (x_lt is None or px < x_lt)
                    and (x_gt is None or px > x_gt)
                    and (y_lt is None or py < y_lt)
                    and (y_gt is None or py > y_gt)
                ):
                    assigned_zone = zone_name
                    break
            agent_cfg["initial_zone"] = assigned_zone or default_zone or "station"

            # Assign role from config: find roles whose spawn_zones include this zone.
            initial_zone = agent_cfg["initial_zone"]
            roles_config = config["agents"].get("roles", {})
            agent_cfg["agent_role"] = AgentManager._assign_role(initial_zone, roles_config)

            # Format goal and memory templates now that role, target, and purpose are known.
            role = agent_cfg["agent_role"]
            role_cfg = roles_config.get(role, {})
            subs = {
                "target": agent_cfg.get("target", ""),
                "purpose": agent_cfg.get("purpose", "their destination"),
            }
            agent_cfg["goal_state"] = role_cfg.get("goal", "Continue your planned journey.").format(
                **subs
            )
            agent_cfg["purpose_memories"] = [t.format(**subs) for t in role_cfg.get("memories", [])]
            agent_cfg["initial_goal"] = agent_cfg["goal_state"]

            is_injured = i in injured_agents
            walking_speed = sample_walking_speed() if not is_injured else 0.5

            # Add agent with level_id for multi-level simulations
            if hasattr(jps_sim, "simulations"):
                # Multi-level simulation
                jps_sim.add_agent(
                    agent_id, start_pos, walking_speed=walking_speed, level_id=level_id
                )
            else:
                # Single-level simulation
                jps_sim.add_agent(agent_id, start_pos, walking_speed=walking_speed)

    @staticmethod
    def _assign_role(initial_zone: str, roles_config: dict) -> str:
        """
        Pick a role whose spawn_zones include ``initial_zone``, using configured weights.

        When multiple roles share the same spawn zone (e.g. ``waiting_for_train`` and
        ``just_arrived`` both spawn on platforms) the ``weight`` values are used for
        random selection.  Falls back to a uniform draw across all roles when no
        spawn_zones match.
        """
        matching = [
            (role, cfg)
            for role, cfg in roles_config.items()
            if initial_zone in cfg.get("spawn_zones", [])
        ]
        if not matching:
            matching = list(roles_config.items())
        if not matching:
            return "default"
        roles_list = [r for r, _ in matching]
        weights = [float(cfg.get("weight", 1.0)) for _, cfg in matching]
        return random.choices(roles_list, weights=weights, k=1)[0]
