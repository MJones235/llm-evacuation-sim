"""The decision cycle: which agents decide, and what happens to their decisions.

Every decision cycle the simulation runner calls
:meth:`DecisionProcessor.process_all_agents`. For each agent due to decide:

1. **Skip** agents with nothing to decide: those who left, are still clearing
   an escalator landing (:mod:`~evacusim.decision.transfer_routing`) or are
   queueing for an escalator.
2. **Perceive** (:class:`~evacusim.decision.situation.SituationAssembler`):
   goal, observation, usable exits, new cues. An agent that asked to be
   woken only by a new cue, and has none, stops here.
3. **Frame** the choice: offered actions and exits, with routing signals,
   as a :class:`~evacusim.core.decision_engine.DecisionContext`.
4. **Decide**: the configured engine (LLM or rule-based) returns a payload.
5. **Record** the decision in the agent's history.
6. **Execute**: translate the payload into a pedestrian-simulation command
   and apply it.

A cycle has two phases. Steps 1-4 run for all deciding agents concurrently
(the LLM engine waits on the network), so everyone decides from the same
state of the world. Steps 5-6 then run agent by agent in a fixed order, so a
run is reproducible whatever order the engine's calls complete in.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Any

from evacusim.core.decision_engine import DecisionContext, DecisionEngine, DecisionResult
from evacusim.decision import payload as payload_lib
from evacusim.decision.action_utils import extract_exit_name
from evacusim.decision.situation import SituationAssembler, zone_containing
from evacusim.decision.transfer_routing import PostTransferRouting
from evacusim.utils.logger import get_logger

logger = get_logger(__name__)

MAX_DECISIONS_KEPT_PER_AGENT = 200


@dataclass
class _Planned:
    """A decision made in phase 1 of a cycle, applied in phase 2."""

    agent_id: str
    position: tuple[float, float]
    ctx: DecisionContext
    result: DecisionResult


class DecisionProcessor:
    """Runs decision cycles: perceive, decide, record, execute.

    Args:
        concordia_agents: Agent entities by id (Concordia entities for the LLM
            engine, placeholders for the rule-based engine); live.
        exited_agents: Ids of agents who have left the simulation; live.
        action_translator: Turns decision payloads into simulation commands.
        action_executor: Applies those commands to the pedestrian simulation.
        message_system: Messages agents have received (LLM engine input).
        state_queries: Agent positions.
        station_layout: Station knowledge and geometry-derived layout.
        agent_decisions: Every agent's decision history (written here).
        agent_destinations: Each agent's current target exit; live.
        perf_timer: Performance timer.
        jps_sim: The pedestrian simulation.
        agent_configs: Per-agent records of the initial population.
        llm_semaphore_limit: Concurrent model calls (LLM engine).
        per_agent_timeout_secs: Wall-clock limit per model call (LLM engine).
        min_redecision_interval_secs: Re-decision throttle (LLM engine).
        wait_nudge_enabled: Remind long-waiting agents to reassess.
        decision_prompt_template_path: Prompt template override (LLM engine).
        decision_engine: The engine; ``None`` builds the LLM engine.
    """

    def __init__(
        self,
        concordia_agents: dict[str, Any],
        exited_agents: set[str],
        action_translator,
        action_executor,
        message_system,
        state_queries,
        station_layout: dict[str, Any],
        agent_decisions: dict[str, dict[str, Any]],
        agent_destinations: dict[str, str],
        perf_timer,
        jps_sim=None,
        agent_configs: list[dict] | None = None,
        llm_semaphore_limit: int = 10,
        per_agent_timeout_secs: float | None = 30.0,
        min_redecision_interval_secs: float = 0.0,
        wait_nudge_enabled: bool = False,
        decision_prompt_template_path: str | None = None,
        decision_engine: DecisionEngine | None = None,
    ):
        self.concordia_agents = concordia_agents
        self.exited_agents = exited_agents
        self.action_translator = action_translator
        self.action_executor = action_executor
        self.state_queries = state_queries
        self.station_layout = station_layout
        self.agent_decisions = agent_decisions
        self.agent_destinations = agent_destinations
        self.perf_timer = perf_timer
        self.jps_sim = jps_sim
        self._agent_cfg: dict[str, dict] = {cfg["id"]: cfg for cfg in (agent_configs or [])}

        self.situation = SituationAssembler(
            agent_cfg=self._agent_cfg,
            station_layout=station_layout,
            action_translator=action_translator,
            action_executor=action_executor,
            agent_destinations=agent_destinations,
            jps_sim=jps_sim,
            wait_nudge_enabled=wait_nudge_enabled,
        )
        self.transfers = PostTransferRouting(
            jps_sim,
            getattr(action_translator, "zones_polygons", {}),
            self._agent_cfg,
            self.situation.goals,
            street_level=station_layout.get("street_level", "0"),
        )

        # How each agent asked to be re-assessed: "next_interval" or "new_cue_only".
        self._reassess_modes: dict[str, str] = {}
        # Agents deferred this cycle (clearing an escalator landing or queueing);
        # the runner reads and clears this.
        self._deferred_escalator_agents: set[str] = set()
        # Agents whose per-agent state has been released after they left.
        self._released: set[str] = set()

        if decision_engine is None:
            from evacusim.decision.llm_decision_engine import LLMDecisionEngine
            from evacusim.decision.llm_prompt import DecisionPromptBuilder

            decision_engine = LLMDecisionEngine(
                agents=concordia_agents,
                prompt_builder=DecisionPromptBuilder(
                    exit_registry=action_translator.exit_registry,
                    exit_semantic_tags=station_layout.get("exit_semantic_tags", {}),
                    agent_decisions=agent_decisions,
                    template_path=decision_prompt_template_path,
                ),
                message_system=message_system,
                agent_decisions=agent_decisions,
                perf_timer=perf_timer,
                max_parallel=llm_semaphore_limit,
                timeout_s=per_agent_timeout_secs,
                min_redecision_interval_s=min_redecision_interval_secs,
            )
        self.engine = decision_engine

    # -- runner interface --------------------------------------------------------

    @property
    def agent_goals(self) -> dict[str, str]:
        """Each agent's current goal text."""
        return self.situation.goals.goals

    def register_agent(self, cfg: dict) -> None:
        """Add a runtime-spawned agent's record (the runner adds its entity)."""
        self._agent_cfg[cfg["id"]] = cfg

    def clear_goal_for_redecision(self, agent_id: str) -> None:
        """Let a non-evacuation goal be re-derived, e.g. after changing level."""
        self.situation.goals.clear_for_redecision(agent_id)

    def reset_agent_decision(self, agent_id: str) -> None:
        """Make the agent's next decision a fresh one (no reuse of the last)."""
        self.engine.reset_agent(agent_id)

    def on_agent_exit(self, agent_id: str) -> None:
        """Release per-agent state when an agent leaves."""
        self.engine.on_agent_exit(agent_id)
        self.situation.wait_since.pop(agent_id, None)
        logger.debug(f"{agent_id}: Cache cleared on exit")

    def consume_deferred_escalator_agents(self) -> set[str]:
        """Agents deferred in the last cycle (the set is then cleared)."""
        deferred = set(self._deferred_escalator_agents)
        self._deferred_escalator_agents.clear()
        return deferred

    def get_cache_statistics(self) -> dict[str, Any]:
        """The engine's counters (LLM calls made and saved, for the LLM engine)."""
        return self.engine.statistics()

    def log_cache_summary(self) -> None:
        stats = self.get_cache_statistics()
        if not stats:
            return
        logger.info("=" * 70)
        logger.info("LLM CALL OPTIMIZATION SUMMARY (Prompt Caching)")
        logger.info("=" * 70)
        logger.info(f"  Total decision cycles: {stats['total_decision_cycles']}")
        logger.info(f"  LLM calls made:        {stats['llm_calls_made']}")
        logger.info(f"  LLM calls skipped:     {stats['llm_calls_skipped']} ✓")
        logger.info(f"  Skip rate:             {stats['skip_rate_percent']:.1f}%")
        logger.info(f"  Cache size:            {stats['cache_stats']['cached_agents']} agents")
        logger.info("=" * 70)

    # -- the decision cycle ------------------------------------------------------

    def process_all_agents(
        self,
        observations: dict[str, str],
        current_sim_time: float,
        agent_ids: list[str] | None = None,
    ) -> float:
        """Run one decision cycle.

        Args:
            observations: Each agent's natural-language observation.
            current_sim_time: Simulation time (s).
            agent_ids: Agents due to decide; ``None`` for all.

        Returns:
            ``current_sim_time``.
        """
        if agent_ids is None:
            logger.info(f"Agent decisions at t={current_sim_time:.1f}s")
        else:
            logger.info(
                f"Targeted agent decisions at t={current_sim_time:.1f}s for {len(agent_ids)} agents"
            )
        asyncio.run(self._decide_all(observations, current_sim_time, agent_ids))
        return current_sim_time

    async def _decide_all(
        self, observations: dict[str, str], current_sim_time: float, agent_ids: list[str] | None
    ) -> None:
        self._deferred_escalator_agents.clear()
        self._released &= self.exited_agents  # agents recovered after a level change
        for agent_id in sorted(self.exited_agents - self._released):
            self.on_agent_exit(agent_id)
            self._released.add(agent_id)
        candidates = agent_ids if agent_ids is not None else list(self.concordia_agents)
        deciding = [
            a for a in candidates if a in self.concordia_agents and a not in self.exited_agents
        ]

        # Zones once per cycle, not once per agent per lookup.
        zones_polygons = getattr(self.action_translator, "zones_polygons", {})
        zones: dict[str, str | None] = {}
        if zones_polygons:
            for agent_id in deciding:
                position = self.state_queries.get_agent_position(agent_id)
                zones[agent_id] = (
                    None if position is None else zone_containing(position, zones_polygons)
                )

        # Phase 1: every agent decides from the same state of the world.
        with self.perf_timer.measure("parallel_agent_processing"):
            planned = await asyncio.gather(
                *(
                    self._decide_one(agent_id, observations, current_sim_time, zones)
                    for agent_id in deciding
                ),
                return_exceptions=True,
            )
        # Phase 2: decisions take effect in a fixed order (that of ``deciding``),
        # whatever order the engine's calls completed in, so a run is
        # reproducible given the same engine responses.
        for agent_id, plan in zip(deciding, planned, strict=True):
            if isinstance(plan, BaseException):
                logger.error(f"Error deciding for {agent_id}: {plan}", exc_info=plan)
            elif plan is not None:
                self._apply(plan, current_sim_time)

    async def _decide_one(
        self,
        agent_id: str,
        observations: dict[str, str],
        current_sim_time: float,
        zones: dict[str, str | None],
    ) -> _Planned | None:
        """Steps 1-4 for one agent: the decision, not yet applied (``None``: nothing to do)."""
        # 1. Skip agents with nothing to decide.
        position = self.state_queries.get_agent_position(agent_id)
        if position is None:
            logger.debug(f"{agent_id}: No position found, likely exited")
            return None
        if self.transfers.should_defer(agent_id, position):
            self._deferred_escalator_agents.add(agent_id)
            return None
        escalators = getattr(self.jps_sim, "escalator_system", None)
        if escalators is not None and escalators.is_committed(agent_id):
            # Re-deciding every cycle would only make queueing agents hop queues.
            self._deferred_escalator_agents.add(agent_id)
            logger.debug(f"{agent_id}: queueing for an escalator — deferring decision")
            return None

        # 2. Perceive.
        zone_id = zones.get(agent_id) if zones else None
        perception = self.situation.perceive(
            agent_id, position, zone_id, observations.get(agent_id, ""), current_sim_time
        )
        if self._reassess_modes.get(agent_id) == "new_cue_only" and not perception.cues:
            logger.debug(f"{agent_id}: gated by reassess_when='new_cue_only' (no cue)")
            return None

        # 3. Frame the choice; 4. decide.
        ctx = self.situation.frame(perception, current_sim_time)
        result = await self.engine.decide(ctx)
        if result.skip_downstream:
            return None
        if result.payload is None:
            logger.warning(f"{agent_id}: no decision payload available, skipping")
            return None
        return _Planned(agent_id, position, ctx, result)

    def _apply(self, plan: _Planned, current_sim_time: float) -> None:
        """Steps 5-6 for one agent: record the decision and carry it out."""
        agent_id, ctx, result = plan.agent_id, plan.ctx, plan.result
        decision = result.payload
        try:
            action_json = result.action_json or payload_lib.to_json(decision)
            self.situation.goals.apply_decision(agent_id, decision, ctx.observation)
            self._reassess_modes[agent_id] = str(decision.get("reassess_when", "next_interval"))

            with self.perf_timer.measure("translate_action", is_parallel=True):
                translated = self.action_translator.translate(agent_id, decision, plan.position)
            if translated.get("action_type") == "wait":
                self.situation.wait_since.setdefault(agent_id, current_sim_time)
            else:
                self.situation.wait_since.pop(agent_id, None)
            summary = payload_lib.summarize_assessment(decision.get("assessment", {}))
            if summary:
                translated["reasoning"] = summary

            # 5. Record.
            with self.perf_timer.measure("decision_storage", is_parallel=True):
                new_exit = extract_exit_name(translated, self.station_layout)
                self._record(agent_id, ctx, result, action_json, translated, new_exit)

            # 6. Execute.
            with self.perf_timer.measure("apply_to_jupedsim", is_parallel=True):
                self.action_executor.execute_action(agent_id, translated, current_sim_time)
            logger.info(f"{agent_id} action: {action_json[:100]}...")
        except Exception as e:
            logger.error(f"Error processing {agent_id}: {e}", exc_info=True)

    def _record(self, agent_id, ctx, result, action_json: str, translated: dict, new_exit) -> None:
        """Append the decision to the agent's history, noting any change of route."""
        decision = result.payload
        reasoning = {
            "assessment": decision.get("assessment", {}),
            "action": decision.get("action", ""),
            "wait_reason": decision.get("wait_reason"),
            "exit_id": decision.get("exit_id"),
            "pace": decision.get("pace"),
            "reassess_when": decision.get("reassess_when"),
        }
        run_meta = self.station_layout.get("run_metadata", {})
        record = {
            "time": ctx.current_sim_time,
            "observation": ctx.observation,
            "prompt": result.prompt if result.llm_was_called else "cached",
            "action": action_json,
            "reasoning": reasoning,
            "decision_payload": decision,
            "translated": translated,
            "offered_set": sorted(ctx.offered_actions_set),
            "repair_status": result.repair_status,
            "run_id": run_meta.get("run_id"),
            "condition": run_meta.get("condition"),
            "agent_id": agent_id,
            "t": ctx.current_sim_time,
            "zone": ctx.zone_id,
            "action_verb": decision.get("action"),
            "wait_reason": decision.get("wait_reason"),
            "exit_id": decision.get("exit_id"),
            "resolved_info_source": translated.get("resolved_info_source"),
            "pace": decision.get("pace"),
            "reassess_when": decision.get("reassess_when"),
            "assessment": decision.get("assessment"),
            "prompt_hash": (
                hashlib.sha256(result.prompt.encode("utf-8")).hexdigest() if result.prompt else None
            ),
            "model_id": run_meta.get("model_id"),
            "seed": run_meta.get("seed"),
            "cue_types": ctx.cues,
        }

        old_exit = self.agent_destinations.get(agent_id)
        if new_exit and old_exit and old_exit != new_exit:
            logger.info(f"🔄 {agent_id} changed route: {old_exit} → {new_exit}")
            record["route_change"] = {
                "from_exit": old_exit,
                "to_exit": new_exit,
                "reason": payload_lib.summarize_assessment(reasoning["assessment"]),
            }

        decisions = self.agent_decisions.setdefault(agent_id, {"decisions": []})["decisions"]
        decisions.append(record)
        if len(decisions) > MAX_DECISIONS_KEPT_PER_AGENT:
            decisions.pop(0)
