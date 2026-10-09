"""Typed run parameters: the single definition of every configuration key.

A run is configured by one YAML file (optionally ``extends:``-ing others). This
module declares what that YAML may contain, the type of each value, its
default, and what it means. :func:`evacusim.config.config_loader.load_run_config`
validates a YAML file against :class:`RunConfig`; anything not declared here is
rejected, so a typo or a stale key fails loudly instead of being ignored.

Sections
--------
================  ===========================================================
``seed``          Master random seed for the run.
``simulation``    Time step, duration, geometry and escalators.
``agents``        Initial population: size, personalities, roles.
``events``        Timed scenario events: alarm, PA, trains, blocked exits.
``systems``       Staff agents (e.g. RCIs, fire brigade) that direct others.
``station``       Station knowledge given to agents: zones, exits, memories.
``decision``      Which decision engine chooses agents' actions.
``llm``           Language-model settings (LLM engine only).
``calibration``   Normal-operations runs: arrivals from usage data.
``monitoring``    Zone population time series.
``performance``   Decision scheduling, concurrency and logging.
``output``        Where results are written.
``video``         MP4 rendering of the run.
``prompts``       Prompt template overrides (LLM engine only).
================  ===========================================================

Each field's ``description`` is the reference documentation for that key.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, model_validator

LevelId = str
"""A JuPedSim level id as written in the geometry files, e.g. ``"0"`` or ``"-1"``."""

Point = Annotated[list[float], Field(min_length=2, max_length=2)]
"""An ``[x, y]`` position in metres, in the level's coordinate frame."""


class Section(BaseModel):
    """Base for every config section.

    Unknown keys are errors, and values are not coerced between types
    (``"5"`` is not a number, ``true`` is not ``1``); an integer is accepted
    where a float is expected.
    """

    model_config = ConfigDict(extra="forbid", strict=True)


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------


class EscalatorParams(Section):
    """Conveyor-escalator parameters (see ``evacusim.escalators``).

    ``walk_share`` and ``walk_speed_factor`` take either one number or a
    per-direction mapping ``{up: .., down: ..}``.
    """

    belt_speed: float = Field(0.5, gt=0, description="Belt speed (m/s).")
    step_depth: float = Field(
        0.4, gt=0, description="Step depth (m); at most one person per step per lane."
    )
    walk_share: float | dict[Literal["up", "down"], float] = Field(
        {"up": 0.25, "down": 0.40},
        description="Fraction of riders who walk in the left lane rather than stand.",
    )
    stander_step_gap_prob: float = Field(
        0.5, ge=0, le=1, description="Probability a stander leaves a free step in front."
    )
    walk_speed_factor: float | dict[Literal["up", "down"], float] = Field(
        {"up": 0.5, "down": 0.7},
        description="Walking pace on the incline as a fraction of floor walking speed.",
    )
    queue_line_slots: int = Field(4, ge=1, description="Queue slots per lane in front of a comb.")
    queue_line_spacing: float = Field(0.6, gt=0, description="Spacing between queue slots (m).")
    queue_slot_grid: float = Field(0.7, gt=0, description="Overflow queue grid spacing (m).")
    queue_max_route_m: float = Field(
        25.0, gt=0, description="Agents farther than this route distance do not join the queue."
    )
    queue_time_gap: float = Field(
        0.5, gt=0, description="JuPedSim time gap for queueing agents (s)."
    )
    landing_search_depth: float = Field(
        1.5, ge=0, description="Depth of landing searched for space before the belt pauses (m)."
    )


class EscalatorPartialParams(Section):
    """Per-escalator overrides: any subset of :class:`EscalatorParams`."""

    belt_speed: float | None = None
    step_depth: float | None = None
    walk_share: float | dict[Literal["up", "down"], float] | None = None
    stander_step_gap_prob: float | None = None
    walk_speed_factor: float | dict[Literal["up", "down"], float] | None = None
    queue_line_slots: int | None = None
    queue_line_spacing: float | None = None
    queue_slot_grid: float | None = None
    queue_max_route_m: float | None = None
    queue_time_gap: float | None = None
    landing_search_depth: float | None = None
    length_m: float | None = Field(None, description="Override the incline length (m).")


class EscalatorsConfig(Section):
    defaults: EscalatorParams = Field(
        default_factory=EscalatorParams, description="Parameters for every escalator."
    )
    overrides: dict[str, EscalatorPartialParams] = Field(
        default_factory=dict,
        description="Per-escalator overrides keyed by exit name, e.g. ``escalator_f_up``.",
    )


class SimulationConfig(Section):
    dt: float = Field(0.05, gt=0, description="Physics time step (s).")
    max_iterations: int = Field(
        200, ge=1, description="Run length in steps; duration = max_iterations × dt."
    )
    decision_interval: float = Field(
        5.0, gt=0, description="Simulated seconds between decision cycles."
    )
    start_time_s: float = Field(
        0.0,
        ge=0,
        lt=86400,
        description="Time of day the run starts (s after midnight); events before it are skipped.",
    )
    network_path: str = Field(description="Directory of the station geometry (level XML files).")
    multi_level: bool = Field(False, description="Simulate several levels joined by escalators.")
    levels: list[LevelId] = Field(
        ["0", "-1"], description="Level ids to load when ``multi_level`` is true."
    )
    level_id: LevelId | int = Field(0, description="Level to load when ``multi_level`` is false.")
    initially_blocked_exits: list[str] = Field(
        default_factory=list, description="Exits closed from the start of the run."
    )
    escalators: EscalatorsConfig | None = Field(
        None, description="Conveyor escalators; omit to run without escalators."
    )


# ---------------------------------------------------------------------------
# agents
# ---------------------------------------------------------------------------

LevelWeights = dict[Literal["high", "medium", "low"], float]


class AgentRole(Section):
    """A behavioural role assigned to agents by spawn zone.

    ``goal`` and ``memories`` are templates; ``{target}`` and ``{purpose}`` are
    filled from ``target`` (sampled) and ``agents.purposes`` (sampled).
    """

    spawn_zones: list[str] = Field(
        default_factory=list, description="Zones whose agents may get this role."
    )
    weight: float = Field(
        1.0, ge=0, description="Relative probability when several roles match a zone."
    )
    goal: str = Field("Continue your planned journey.", description="Goal template.")
    memories: list[str] = Field(default_factory=list, description="Initial memory templates.")
    target: list[str | int] = Field(
        default_factory=list, description="Values sampled for the ``{target}`` placeholder."
    )
    decision_prompt_extra: str = Field(
        "", description="Extra decision-prompt text for this role (LLM engine)."
    )


class AgeRange(Section):
    min: int = Field(18, ge=18)
    max: int = Field(75, ge=18)


class AgentsConfig(Section):
    count: int = Field(ge=0, description="Number of agents in the initial population.")
    snapshot_load_path: str | None = Field(
        None, description="Load the population from this snapshot instead of sampling it."
    )
    snapshot_save_path: str | None = Field(
        None, description="Save the sampled population to this snapshot."
    )
    spawn_min_separation: float = Field(
        0.5,
        gt=0,
        description="Minimum distance between spawned agents (m); JuPedSim rejects closer pairs.",
    )
    knowledge_profiles: dict[str, float] = Field(
        min_length=1,
        description="Relative weights of station-knowledge profiles; keys must exist in "
        "``station.knowledge.profiles``.",
    )
    personalities: dict[Literal["N", "O", "C"], LevelWeights] = Field(
        default_factory=dict,
        description="OCEAN level weights per dimension (Neuroticism, Openness, "
        "Conscientiousness); a missing dimension is sampled uniformly.",
    )
    age: AgeRange = Field(default_factory=AgeRange, description="Uniform age range (years).")
    purposes: list[str] = Field(
        ["their destination"], description="Values sampled for the ``{purpose}`` placeholder."
    )
    roles: dict[str, AgentRole] = Field(
        default_factory=dict, description="Behavioural roles, keyed by role name."
    )

    @model_validator(mode="after")
    def _positive_profile_weights(self) -> AgentsConfig:
        for name, weight in self.knowledge_profiles.items():
            if weight <= 0:
                raise ValueError(f"knowledge_profiles.{name} must be positive")
        return self


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------


class Cue(Section):
    """What a message conveys, for the rule-based decision engine.

    The LLM engine reads the message text; the rule-based engine reads this
    annotation instead. ``strength`` sets how quickly people respond (see
    ``decision.response``); ``instruction`` is what they are told to do.
    """

    strength: Literal["weak", "medium", "strong"]
    instruction: Literal["none", "leave_station", "board_train"] = "none"
    route: list[str] = Field(
        default_factory=list,
        description="Exits the message names (e.g. ``[escalator_b_up]``). Agents who hear "
        "it learn them, and are offered them wherever they can reach them.",
    )


class _TimedEvent(Section):
    time: float = Field(ge=0, description="Simulation time the event fires (s).")


class MessageEvent(_TimedEvent):
    """A cue every agent perceives, e.g. the fire alarm sounding.

    ``{elapsed_time}`` in the message is replaced by the time since it began.
    """

    type: Literal["message"] = "message"
    message: str
    repeat_interval: float | None = Field(
        None, gt=0, description="Re-deliver every this many seconds."
    )
    cue: Cue | None = Field(None, description="Warning conveyed to everyone (rule engine).")


class PAAnnouncementEvent(_TimedEvent):
    """A public-address announcement, optionally different per zone."""

    type: Literal["pa_announcement"]
    message: str = Field("", description="Text heard in zones without a specific message.")
    zone_messages: dict[str, str] | None = Field(
        None, description="Zone-specific text, keyed by zone id (``default`` is a fallback)."
    )
    sender_label: str = Field("PA system", description="Who agents hear the announcement from.")
    repeat_interval: float | None = Field(None, gt=0, description="Repeat every this many seconds.")
    cue: Cue | None = Field(None, description="Warning conveyed by ``message`` (rule engine).")
    zone_cues: dict[str, Cue] = Field(
        default_factory=dict, description="Warning conveyed by each zone's message."
    )


class TrainArrivalEvent(_TimedEvent):
    """A train arrives and opens a boarding exit on each listed platform."""

    type: Literal["train_arrival"]
    platforms: list[int | str] = Field(min_length=1, description="Platform numbers served.")
    dwell_seconds: float = Field(30.0, gt=0, description="Time the doors stay open (s).")
    message: str = Field("", description="Announcement made on arrival.")
    sender_label: str = Field("PA system")
    departure_announce: bool = Field(True, description="Announce the departure.")
    departure_message: str = Field(
        "The train doors have closed and the train has departed. "
        "If you are still on the platform, please use the escalators to leave.",
    )
    departure_sender_label: str | None = Field(
        None, description="Sender of the departure announcement (default: ``sender_label``)."
    )


class BlockExitEvent(_TimedEvent):
    """Close exits, e.g. escalators near the fire."""

    type: Literal["block_exit"]
    exits: list[str] = Field(min_length=1)
    message: str | None = Field(None, description="Optional cue delivered with the closure.")


def _event_type(value: Any) -> str:
    if isinstance(value, dict):
        return value.get("type", "message")
    return getattr(value, "type", "message")


Event = Annotated[
    Annotated[MessageEvent, Tag("message")]
    | Annotated[PAAnnouncementEvent, Tag("pa_announcement")]
    | Annotated[TrainArrivalEvent, Tag("train_arrival")]
    | Annotated[BlockExitEvent, Tag("block_exit")],
    Discriminator(_event_type),
]


# ---------------------------------------------------------------------------
# systems (staff)
# ---------------------------------------------------------------------------


class ZoneRef(Section):
    """A place, given as a zone (its centroid is used) or an explicit position."""

    zone: str | None = None
    level_id: LevelId | None = None
    position: Point | None = None


class StaffPhase(Section):
    """One behaviour phase of a staff agent; phases run in order."""

    trigger: Literal["immediate", "on_event", "on_reach_zone", "after_seconds"] = Field(
        "immediate",
        description="What starts this phase: ``on_event`` is the first warning event "
        "(an event with a cue, e.g. the alarm).",
    )
    trigger_zone: str | None = Field(None, description="Zone for ``on_reach_zone``.")
    trigger_level_id: LevelId | None = None
    after_seconds: float | None = Field(None, ge=0, description="Delay for ``after_seconds``.")
    movement: Literal["hold", "zone_patrol"] = "hold"
    hold_zone: str | None = Field(None, description="Hold at this zone's centroid.")
    hold_level_id: LevelId | None = None
    patrol_zones: list[ZoneRef] = Field(default_factory=list)
    patrol_dwell_time: float = Field(20.0, ge=0, description="Pause at each patrol point (s).")
    directive_radius: float = Field(
        10.0, ge=0, description="Agents within this distance hear it (m); 0 = silent."
    )
    directive_interval: float = Field(10.0, gt=0, description="Seconds between directives.")
    message: str = Field("", description="Directive spoken to nearby agents.")
    messages_by_zone: dict[str, str] = Field(
        default_factory=dict, description="Zone-specific directives."
    )
    cue: Cue | None = Field(None, description="Warning conveyed by ``message`` (rule engine).")
    cues_by_zone: dict[str, Cue] = Field(
        default_factory=dict, description="Warning conveyed by each zone's directive."
    )


class StaffSystemConfig(Section):
    """A group of staff agents, e.g. Revenue Control Inspectors."""

    enabled: bool = Field(False, description="Disabled systems are not created.")
    role_label: str | None = Field(None, description="How agents refer to them.")
    walking_speed: float = Field(1.2, gt=0, description="Walking speed (m/s).")
    spawn_positions: list[ZoneRef] = Field(
        default_factory=list, description="One staff agent per entry."
    )
    phases: list[StaffPhase] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# station
# ---------------------------------------------------------------------------


class ZoneBoundary(Section):
    """Axis-aligned rule assigning spawn positions to a zone."""

    default: bool = False
    x_lt: float | None = None
    x_gt: float | None = None
    y_lt: float | None = None
    y_gt: float | None = None


class LocationMemoryCondition(Section):
    profiles: list[str] | None = None
    level_ids: list[LevelId | int] | None = None
    zones: list[str] | None = None


class LocationMemory(Section):
    """Memories given to agents that match every condition in ``when``."""

    when: LocationMemoryCondition = Field(default_factory=LocationMemoryCondition)
    memories: list[str] = Field(min_length=1)


class StationKnowledge(Section):
    base_memories: list[str] = Field(min_length=1, description="Known by every agent.")
    profiles: dict[str, list[str]] = Field(
        min_length=1, description="Memories per knowledge profile."
    )
    location_memories: list[LocationMemory] = Field(default_factory=list)


class GoalSemanticPolicy(Section):
    """Maps a goal (by keyword) to preferred/avoided kinds of exit.

    Policies are evaluated in order; the first match applies.
    """

    when_goal_contains_any: list[str] = Field(min_length=1)
    applies_in_zones: list[str] | None = None
    prefer_exit_tags: list[str] | None = None
    avoid_exit_tags: list[str] | None = None
    instruction: str | None = Field(None, description="Guidance added to the LLM prompt.")


class StationConfig(Section):
    """What agents know about the station, and how zones and exits are named."""

    knowledge: StationKnowledge
    street_level: LevelId = Field("0", description="Level id of the street exits (concourse).")
    platform_level: LevelId = Field("-1", description="Level id of the train platforms.")
    zone_labels: dict[str, str] = Field(
        default_factory=dict, description="Human-readable names of zones and levels."
    )
    zone_boundaries: dict[LevelId, dict[str, ZoneBoundary]] = Field(
        default_factory=dict, description="Per level, rules that assign spawn zones."
    )
    arrival_exits_by_zone: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Exits that lead *into* a zone and so are not ways out of it.",
    )
    zone_known_exits_by_profile: dict[str, dict[str, list[str]]] = Field(
        default_factory=dict, description="Exits each knowledge profile knows, per zone."
    )
    zones_hidden_for_zone: dict[str, list[str]] = Field(
        default_factory=dict, description="Zones not offered as destinations from a zone."
    )
    zone_goal_keywords: dict[str, list[str]] = Field(
        default_factory=dict, description="Goal substrings meaning 'this zone is my destination'."
    )
    platform_down_exits: dict[str, list[str]] = Field(
        default_factory=dict, description="Down escalators leading to each platform zone."
    )
    street_exits: list[str] = Field(default_factory=list)
    custom_exit_display_names: dict[str, str] = Field(default_factory=dict)
    exit_semantic_tags: dict[str, list[str]] = Field(
        default_factory=dict, description="Tags describing where each exit leads."
    )
    goal_semantic_policies: list[GoalSemanticPolicy] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------


class LLMDecisionConfig(Section):
    """Agents decide by prompting a language model (see the ``llm`` section)."""

    engine: Literal["llm"] = "llm"


class RuleWeights(Section):
    """Weights of the exit-scoring terms; they need not sum to one."""

    proximity: float = Field(0.5, ge=0)
    busyness: float = Field(0.3, ge=0)
    familiarity: float = Field(0.2, ge=0)
    visibility: float = Field(0.0, ge=0)


class ResponseDelays(Section):
    """Median delay (s) from perceiving a cue to starting to evacuate, by cue strength.

    Starting values follow Proulx (1991) "time to start to move" (bell only
    ~8 min, minimal PA ~1:15, directive PA ~0:40); fitted in the comparison study.
    """

    weak: float = Field(450.0, gt=0)
    medium: float = Field(75.0, gt=0)
    strong: float = Field(40.0, gt=0)


class SocialCue(Section):
    """Seeing others leave: a cue when enough neighbours are evacuating."""

    enabled: bool = True
    radius_m: float = Field(5.0, gt=0, description="Who counts as a neighbour (m).")
    threshold: float = Field(
        0.5, gt=0, le=1, description="Fraction of neighbours evacuating that is a cue."
    )
    min_neighbours: int = Field(2, ge=1, description="Fewer neighbours never make a cue.")
    strength: Literal["weak", "medium", "strong"] = "medium"


class RuleBasedDecisionConfig(Section):
    """Agents decide with deterministic rules; no language model is used.

    Agents go from unaware (normal journey) to aware (investigating) on a
    warning cue, then evacuate after a delay set by the strongest cue
    (see ``evacusim.decision.rule_based_decision_engine``).
    """

    engine: Literal["rule_based"]
    rule_weights: RuleWeights = Field(default_factory=RuleWeights)
    crowd_radius_m: float = Field(
        5.0, gt=0, description="Radius for counting the crowd at an exit (m)."
    )
    response_median_s: ResponseDelays = Field(default_factory=ResponseDelays)
    response_sigma: float = Field(
        0.6, ge=0, description="Lognormal shape of response delays (0: always the median)."
    )
    social: SocialCue = Field(default_factory=SocialCue)
    evacuation_pace: Literal["normal_pace", "hurrying", "running"] = Field(
        "normal_pace", description="Pace once evacuating."
    )


DecisionConfig = Annotated[
    LLMDecisionConfig | RuleBasedDecisionConfig, Field(discriminator="engine")
]


class LLMConfig(Section):
    temperature: float = Field(0.7, ge=0)
    max_retries: int = Field(3, ge=0)
    max_completion_tokens: int = Field(8000, ge=1)
    timeout: float = Field(90.0, gt=0, description="Request timeout (s).")
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    response_format: str = "json_object"
    embedder: str = "sentence-transformers/all-mpnet-base-v2"


# ---------------------------------------------------------------------------
# calibration (normal operations)
# ---------------------------------------------------------------------------


class SpawnPoint(Section):
    level: LevelId
    xy: Point
    door_points: list[Point] = Field(
        default_factory=list, description="Train door positions, for platform spawn points."
    )


class CalibrationConfig(Section):
    """Normal operations: passengers arrive over time from observed usage data."""

    enabled: bool = False
    entrance_usage_csv: str = Field(description="Per-entrance arrivals per interval.")
    timetable_csv: str | None = Field(None, description="Train arrivals and alightings.")
    entrance_level: LevelId = "0"
    platform_level: LevelId = "-1"
    entrance_dest_exits: list[str] = Field(default_factory=list)
    platform_exit: str = ""
    spawn_points: dict[str, SpawnPoint] = Field(min_length=1)
    knowledge_profile: str = "novice"
    spawn_jitter_m: float = Field(0.5, ge=0)
    train_door_jitter_m: float = Field(0.3, ge=0)
    train_alighting_duration_s: float = Field(12.0, gt=0)
    walking_speed_mean: float = Field(1.34, gt=0)
    walking_speed_std: float = Field(0.0, ge=0)
    walking_speed_min: float = Field(0.3, gt=0)
    walking_speed_max: float = Field(2.2, gt=0)


# ---------------------------------------------------------------------------
# monitoring, performance, output, video, prompts
# ---------------------------------------------------------------------------


class MonitoredZone(Section):
    name: str
    description: str | None = None
    type: Literal["level", "exited", "escalator_queue", "on_escalator"] = "level"
    level: LevelId | None = Field(None, description="Only count agents on this level.")
    area_patterns: list[str] = Field(
        default_factory=list, description="Only count agents inside areas with these prefixes."
    )
    exclude_area_patterns: list[str] = Field(default_factory=list)


class MonitoringConfig(Section):
    interval_seconds: float = Field(60.0, gt=0)
    zones: list[MonitoredZone] | None = Field(None, description="Omit for the default zones.")


class PerformanceConfig(Section):
    pace_multipliers: dict[str, float] = Field(
        default_factory=dict, description="Speed multiplier per pace, e.g. ``running: 2.2``."
    )
    max_parallel_agents: int = Field(10, ge=1, description="Concurrent LLM decisions.")
    decision_timeout_seconds: float = Field(30.0, gt=0)
    min_redecision_interval_seconds: float = Field(0.0, ge=0)
    wait_nudge_enabled: bool = False
    immediate_redecision_on_transfer: bool = False
    decision_groups: int = Field(
        3, ge=1, description="Agents decide in this many staggered groups."
    )
    bootstrap_initial_decisions: bool = True
    file_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class OutputConfig(Section):
    directory: str = Field("results", description="Runs are written below this directory.")


class VideoConfig(Section):
    enabled: bool = False
    fps: int = Field(20, ge=1)
    speedup: float = Field(1.0, gt=0)


class PromptsConfig(Section):
    decision_prompt_template_path: str | None = None


# ---------------------------------------------------------------------------
# the whole run
# ---------------------------------------------------------------------------


def as_dict(section: BaseModel) -> dict[str, Any]:
    """A section as a plain dict, for components that still take dicts.

    Unset optional values (``None``) are omitted, so ``d.get(key, default)``
    in those components behaves as it did when they read the YAML directly.
    """
    return section.model_dump(exclude_none=True)


class RunConfig(Section):
    """Every parameter of one simulation run."""

    seed: int = Field(
        0,
        description="Master random seed; every random component's seed is derived from it "
        "(see evacusim.utils.seeding). Same seed and config, same run.",
    )
    simulation: SimulationConfig = Field(description="Time step, duration, geometry, escalators.")
    agents: AgentsConfig = Field(description="Initial population: size, personalities, roles.")
    events: list[Event] = Field(
        default_factory=list,
        description="Timed scenario events: alarm, PA announcements, trains, blocked exits.",
    )
    systems: dict[str, StaffSystemConfig] = Field(
        default_factory=dict,
        description="Staff agents (e.g. RCIs, fire brigade) that direct others, by name.",
    )
    station: StationConfig = Field(description="Station knowledge given to agents.")
    decision: DecisionConfig = Field(
        default_factory=LLMDecisionConfig,
        description="Which decision engine chooses agents' actions (``engine: llm`` or "
        "``engine: rule_based``).",
    )
    llm: LLMConfig = Field(
        default_factory=LLMConfig, description="Language-model settings (LLM engine only)."
    )
    calibration: CalibrationConfig | None = Field(
        None, description="Normal-operations runs: arrivals from usage data."
    )
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig, description="Zone population time series."
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Decision scheduling, concurrency and logging.",
    )
    output: OutputConfig = Field(default_factory=OutputConfig, description="Where results go.")
    video: VideoConfig = Field(default_factory=VideoConfig, description="MP4 rendering.")
    prompts: PromptsConfig = Field(
        default_factory=PromptsConfig, description="Prompt template overrides (LLM engine only)."
    )

    @model_validator(mode="after")
    def _cross_section_rules(self) -> RunConfig:
        unknown = set(self.agents.knowledge_profiles) - set(self.station.knowledge.profiles)
        if unknown:
            raise ValueError(
                f"agents.knowledge_profiles names unknown profiles {sorted(unknown)}; "
                "define them in station.knowledge.profiles"
            )
        if self.calibration and self.calibration.enabled and self.decision.engine == "llm":
            raise ValueError(
                "calibration.enabled requires decision.engine: rule_based "
                "(runtime spawning does not support the LLM engine)"
            )
        return self
