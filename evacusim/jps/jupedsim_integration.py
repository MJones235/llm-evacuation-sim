"""
Real JuPedSim integration for Concordia station evacuation simulation.

This module provides a concrete implementation of JuPedSim pedestrian dynamics
for the Concordia evacuation scenario, replacing the mock simulation.

Features:
    - Real pedestrian movement physics
    - Station geometry loading from SUMO network files
    - Exit and evacuation routing
    - Spatial queries for agent observations
    - Real-time position tracking for visualization
"""

from pathlib import Path
from typing import Any

import jupedsim as jps

from evacusim.utils.logger import get_logger
from evacusim.jps.agent_tracker import AgentTracker
from evacusim.jps.exit_manager import ExitManager
from evacusim.jps.geometry_manager import GeometryManager
from evacusim.jps.stage_manager import StageManager

logger = get_logger(__name__)


class ConcordiaJuPedSimulation:
    """
    Real JuPedSim simulation wrapper for Concordia integration.

    This class implements the PedestrianSimulation interface using JuPedSim
    as the underlying pedestrian dynamics engine.

    Satisfies the PedestrianSimulation protocol, making it swappable with
    other simulation backends.
    """

    def __init__(
        self,
        network_path: Path | None = None,
        dt: float = 0.05,
        exit_radius: float = 10.0,
        level_id: int | str = 0,
    ):
        """
        Initialize JuPedSim simulation with station geometry.

        Args:
            network_path: Path to network directory containing level_*.xml or walking_areas.add.xml
            dt: Timestep in seconds (matches JuPedSim convention)
            exit_radius: Radius of circular exits in meters
            level_id: Level ID to load (default: 0). Loads level_{level_id}.xml or falls back to walking_areas.add.xml
        """
        self.dt = dt
        self.exit_radius = exit_radius
        self.current_step = 0
        self.is_complete = False
        self.level_id = str(level_id)  # Store level_id for reference

        if network_path is None:
            raise ValueError("network_path required")

        self.network_path = network_path
        self.geometry_manager = GeometryManager(network_path, dt, level_id)
        self.simulation = self.geometry_manager.simulation
        self.stage_manager = StageManager(self.simulation)

        self.exit_manager = ExitManager(
            self.stage_manager,
            self.geometry_manager.entrance_areas,
            self.geometry_manager.walkable_areas_with_obstacles,
            self.geometry_manager.walkable_areas,
            level_id=self.level_id,
        )

        self.agent_tracker = AgentTracker(self.simulation)
        self.last_known_positions: dict[str, tuple[float, float]] = {}
        self.agent_assigned_exits: dict[str, str] = {}

        logger.info(
            f"JuPedSim simulation initialized: "
            f"{len(self.geometry_manager.walkable_areas)} walkable areas, "
            f"{len(self.geometry_manager.entrance_areas)} entrances, "
            f"{len(self.exit_manager.evacuation_exits)} exits"
        )

    def add_agent(
        self, agent_id: str, position: tuple[float, float], walking_speed: float = 1.34
    ) -> None:
        """
        Add an agent to the simulation.

        Agent is placed at position with a valid default evacuation journey.
        The LLM can immediately switch this journey when making its first decision.

        Args:
            agent_id: Concordia agent ID
            position: Initial (x, y) position
            walking_speed: Desired walking speed in m/s (default: 1.34 m/s)

        Raises:
            Exception: If JuPedSim fails to add the agent
        """
        # JuPedSim requires integer journey_id and stage_id at agent creation.
        # Assign a valid default journey immediately; LLM can override on first decision.
        exit_name, stage_id, journey_id = self.exit_manager.get_default_exit()

        jps_id = self.simulation.add_agent(
            jps.CollisionFreeSpeedModelAgentParameters(
                position=position,
                journey_id=journey_id,
                stage_id=stage_id,
                v0=walking_speed,
            )
        )

        # Register in agent tracker
        self.agent_tracker.add_agent(agent_id, jps_id)
        self.agent_assigned_exits[agent_id] = exit_name

        logger.info(
            f"Added agent {agent_id} at {position} "
            f"with speed {walking_speed:.2f} m/s (JPS ID: {jps_id})"
        )

    def step(self) -> bool:
        """
        Advance simulation by one timestep.

        Returns:
            True if simulation should continue, False if complete
        """
        if self.is_complete:
            return False

        # Run JuPedSim step
        self.simulation.iterate()
        self.current_step += 1

        # Check if any agents remain
        if self.simulation.agent_count() == 0:
            logger.info("All agents have exited the simulation")
            self.is_complete = True
            return False

        return True

    def get_agent_position(self, agent_id: str) -> tuple[float, float] | None:
        """
        Get agent's current position.

        Args:
            agent_id: Concordia agent ID

        Returns:
            Agent's (x, y) position, or None if agent has exited
        """
        return self.agent_tracker.get_position(agent_id)

    def set_agent_target(self, agent_id: str, target: tuple[float, float]) -> None:
        """
        Set an agent's movement target by creating a waypoint.

        Args:
            agent_id: Concordia agent ID
            target: Target (x, y) position

        Raises:
            KeyError: If agent is not in the simulation (may have exited)
            Exception: If JuPedSim fails to create waypoint or switch journey
        """
        if not self.agent_tracker.is_agent_active(agent_id):
            logger.debug(f"Cannot set target for agent {agent_id} - already exited")
            return

        jps_id = self.agent_tracker.get_jps_id(agent_id)
        self.agent_tracker.set_target(agent_id, target)

        # Create a waypoint stage at the target location
        stage_id = self.simulation.add_waypoint_stage(target, distance=2.0)

        # Create a journey to this waypoint
        journey = jps.JourneyDescription([stage_id])
        journey_id = self.simulation.add_journey(journey)

        # Update agent's journey and stage
        self.simulation.switch_agent_journey(
            agent_id=jps_id,
            journey_id=journey_id,
            stage_id=stage_id,
        )

        logger.info(f"Set target for agent {agent_id} to {target}")

    def set_agent_evacuation_exit(self, agent_id: str, exit_name: str) -> None:
        """
        Direct an agent to a specific evacuation exit.

        Args:
            agent_id: Concordia agent ID
            exit_name: Name of the evacuation exit

        Raises:
            KeyError: If agent or exit is not found
            Exception: If JuPedSim fails to switch journey
        """
        if not self.agent_tracker.is_agent_active(agent_id):
            logger.debug(f"Cannot set exit for agent {agent_id} - already exited")
            return

        jps_id = self.agent_tracker.get_jps_id(agent_id)
        stage_id, journey_id = self.exit_manager.get_exit_ids(exit_name)

        self.simulation.switch_agent_journey(
            agent_id=jps_id,
            journey_id=journey_id,
            stage_id=stage_id,
        )

        self.agent_assigned_exits[agent_id] = exit_name

        logger.info(f"Directed agent {agent_id} to exit '{exit_name}'")

    def set_agent_speed(self, agent_id: str, speed: float) -> None:
        """
        Set an agent's walking speed mid-simulation.

        Args:
            agent_id: Concordia agent ID
            speed: Walking speed in m/s

        Raises:
            Exception: If JuPedSim fails to update agent's speed
        """
        if not self.agent_tracker.is_agent_active(agent_id):
            logger.debug(f"Cannot set speed for agent {agent_id} - already exited")
            return

        jps_id = self.agent_tracker.get_jps_id(agent_id)

        # Get the agent object and update its desired speed
        agent = self.simulation.agent(jps_id)
        agent.model.v0 = speed

        logger.info(f"Set agent {agent_id} speed to {speed:.2f} m/s")

    def get_agent_speed(self, agent_id: str) -> float | None:
        """
        Get an agent's current desired walking speed.

        Args:
            agent_id: Concordia agent ID

        Returns:
            Agent's desired speed in m/s, or None if agent has exited
        """
        if not self.agent_tracker.is_agent_active(agent_id):
            return None

        jps_id = self.agent_tracker.get_jps_id(agent_id)
        agent = self.simulation.agent(jps_id)
        return float(agent.model.v0)

    def get_nearby_agents(self, agent_id: str, radius: float) -> list[dict[str, Any]]:
        """
        Get information about agents within radius of given agent.

        Args:
            agent_id: ID of the querying agent
            radius: Search radius in meters

        Returns:
            List of nearby agent info dictionaries
        """
        return self.agent_tracker.get_nearby_agents(agent_id, radius)

    def get_all_nearby_agents_bulk(self, radius: float) -> dict[str, list[dict[str, Any]]]:
        """
        Return nearby-agent lists for ALL agents in a single O(n) pass.

        Prefer this over calling get_nearby_agents() in a loop to avoid the
        O(n²) cost of reading simulation.agents() N times.

        Args:
            radius: Search radius in metres

        Returns:
            Mapping agent_id -> list of nearby-agent info dicts
        """
        return self.agent_tracker.get_all_nearby_agents_bulk(radius)

    def get_simulation_time(self) -> float:
        """Get current simulation time in seconds."""
        return self.current_step * self.dt

    def get_all_agent_positions(self) -> dict[str, tuple[float, float]]:
        """
        Get positions of all agents for visualization.

        Returns:
            Dictionary mapping Concordia agent IDs to (x, y) positions
        """
        return self.agent_tracker.get_all_positions()

    def check_exits(self) -> dict[str, str]:
        """
        Check for agents that have exited this level and determine exit type.

        Returns:
            Dictionary mapping agent_id -> exit_name for agents that just exited.
            Agents are matched to exits by finding the nearest exit to their last position.
        """
        exited = {}

        # Update last known positions for all still-active agents
        current_positions = self.agent_tracker.get_all_positions()
        self.last_known_positions.update(current_positions)

        # Get list of all agents that were registered but are no longer in simulation
        current_registered = set(self.agent_tracker.agent_ids.keys())
        still_active = set(current_positions.keys())
        exited_agent_ids = current_registered - still_active

        if len(exited_agent_ids) > 0:
            logger.info(f"Detected {len(exited_agent_ids)} exited agents on level {self.level_id}")

        # For each exited agent, find which exit they likely used
        for agent_id in exited_agent_ids:
            assigned_exit = self.agent_assigned_exits.get(agent_id)
            if assigned_exit in self.exit_manager.exit_coordinates:
                exited[agent_id] = assigned_exit
                logger.info(f"Agent {agent_id} exited through assigned exit {assigned_exit}")
                self.agent_tracker.remove_agent(agent_id)
                self.last_known_positions.pop(agent_id, None)
                self.agent_assigned_exits.pop(agent_id, None)
                continue

            # Prefer explicit movement target; fallback to last known position.
            match_point = self.agent_tracker.agent_targets.get(agent_id)
            if match_point is None:
                match_point = self.last_known_positions.get(agent_id)

            if match_point is None:
                logger.warning(
                    f"Agent {agent_id} exited with no target and no last known position - no exit match possible"
                )
                self.agent_tracker.remove_agent(agent_id)
                self.agent_assigned_exits.pop(agent_id, None)
                continue

            # Find nearest exit to the target
            if not self.exit_manager.exit_coordinates:
                logger.error(
                    f"No exit coordinates available (dict is empty). Available exits in exit_manager: {list(self.exit_manager.evacuation_exits.keys())}"
                )
                self.agent_tracker.remove_agent(agent_id)
                self.agent_assigned_exits.pop(agent_id, None)
                continue

            nearest_exit = None
            min_distance = float("inf")

            logger.debug(
                f"Agent {agent_id} match point: {match_point}, checking against exits: "
                f"{list(self.exit_manager.exit_coordinates.keys())}"
            )

            for exit_name, exit_pos in self.exit_manager.exit_coordinates.items():
                distance = (
                    (match_point[0] - exit_pos[0]) ** 2 + (match_point[1] - exit_pos[1]) ** 2
                ) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    nearest_exit = exit_name

            if nearest_exit:
                exited[agent_id] = nearest_exit
                logger.info(
                    f"Agent {agent_id} exited through {nearest_exit} "
                    f"(match point was {min_distance:.2f}m away)"
                )
            else:
                logger.warning(f"Agent {agent_id} exited but no exit found nearby")

            self.agent_tracker.remove_agent(agent_id)
            self.last_known_positions.pop(agent_id, None)
            self.agent_assigned_exits.pop(agent_id, None)

        return exited

    def get_geometry(self) -> dict[str, Any]:
        """
        Get geometry information for visualization.

        Returns:
            Dictionary with walkable areas, entrances, platforms, etc.
        """
        geometry_data = self.geometry_manager.get_geometry_data()

        # Add evacuation exits info
        geometry_data["evacuation_exits"] = {
            name: geometry_data["entrance_areas"].get(name)
            for name in self.exit_manager.evacuation_exits.keys()
            if name in geometry_data["entrance_areas"]
        }

        return geometry_data

    def generate_spawn_positions(
        self, num_agents: int, seed: int = 42
    ) -> list[tuple[float, float]]:
        """
        Generate spawn positions for agents within the walkable geometry.

        Uses JuPedSim's distribute_by_number on the actual simulation geometry
        to ensure positions respect boundary constraints and obstacles.

        Args:
            num_agents: Number of spawn positions to generate
            seed: Random seed for reproducibility

        Returns:
            List of (x, y) coordinate tuples for spawn positions

        Raises:
            RuntimeError: If unable to generate spawn positions
        """
        import random

        random.seed(seed)

        walkable_areas = self.geometry_manager.walkable_areas_with_obstacles

        if not walkable_areas:
            raise RuntimeError("Cannot spawn agents without geometry")

        spawn_positions = []
        area_list = list(walkable_areas.items())
        total_area = sum(poly.area for _, poly in area_list)

        logger.info(f"Distributing {num_agents} agents across {len(area_list)} walkable areas")

        for idx, (area_name, poly) in enumerate(area_list):
            # Proportional allocation based on polygon area
            poly_agents = int(num_agents * (poly.area / total_area))

            # Last polygon gets remainder to ensure exact count
            if idx == len(area_list) - 1:
                poly_agents = num_agents - len(spawn_positions)

            if poly_agents > 0:
                try:
                    # Use JuPedSim's distribution with safe boundary distances
                    positions = jps.distribute_by_number(
                        polygon=poly,
                        number_of_agents=poly_agents,
                        distance_to_agents=0.5,  # Min 0.5m between agents
                        distance_to_polygon=0.4,  # Min 0.4m from boundaries
                        seed=seed + idx,
                    )
                    # Reject any position that falls inside an escalator corridor so
                    # agents are never spawned mid-escalator.
                    escalator_corridors = getattr(self.geometry_manager, "escalator_corridors", {})
                    if escalator_corridors:
                        from shapely.geometry import Point as _Point

                        filtered = [
                            p
                            for p in positions
                            if not any(
                                cpoly.contains(_Point(p)) for cpoly in escalator_corridors.values()
                            )
                        ]
                        rejected = len(positions) - len(filtered)
                        if rejected:
                            logger.info(
                                f"  {area_name}: filtered out {rejected} spawn position(s) "
                                "inside escalator corridors"
                            )
                        positions = filtered
                    spawn_positions.extend(positions)
                    logger.info(f"  {area_name}: {len(positions)} agents")
                except Exception as e:
                    logger.warning(f"  {area_name}: Failed to place {poly_agents} agents - {e}")
                    # Try with fewer agents if density is too high
                    if poly_agents > 1:
                        try:
                            reduced = max(1, poly_agents // 2)
                            positions = jps.distribute_by_number(
                                polygon=poly,
                                number_of_agents=reduced,
                                distance_to_agents=0.5,
                                distance_to_polygon=0.4,
                                seed=seed + idx,
                            )
                            spawn_positions.extend(positions)
                            logger.info(f"  {area_name}: {len(positions)} agents (reduced)")
                        except Exception as e2:
                            logger.error(f"  {area_name}: Could not place any agents - {e2}")

        if len(spawn_positions) < num_agents:
            logger.warning(
                f"Only generated {len(spawn_positions)} of {num_agents} requested positions"
            )

        return spawn_positions
