"""The rule engine's cue-driven stages: unaware -> aware -> evacuating."""

import asyncio
import unittest

from evacusim.core.decision_engine import DecisionContext, ExitOption
from evacusim.decision import payload
from evacusim.decision.rule_based_decision_engine import (
    AWARE,
    EVACUATING,
    UNAWARE,
    RuleBasedDecisionEngine,
)

STREET = ExitOption("grey_street", "Grey Street", distance_m=20.0, semantic_tags=("to_street",))
DOWN = ExitOption("escalator_a_down", "Escalator A", distance_m=5.0, semantic_tags=("to_platform",))


def engine(**kw):
    defaults = dict(response_median_s={"weak": 400.0, "medium": 80.0, "strong": 40.0})
    return RuleBasedDecisionEngine(response_sigma=0.0, **{**defaults, **kw})


def cue(t, strength, source="alarm", instruction="none"):
    return {"time": t, "source": source, "strength": strength, "instruction": instruction}


def ctx(
    t,
    warnings=(),
    actions=("continue_activity", "wait", "evacuate"),
    agent="a",
    goal="Catch your train from Platform 3.",
    nearby=(),
    agent_cfg=None,
):
    exits = [STREET, DOWN]
    return DecisionContext(
        agent_id=agent,
        position=(0.0, 0.0),
        zone_id="concourse",
        goal=goal,
        observation="",
        agent_cfg=agent_cfg or {},
        offered_actions=list(actions),
        offered_wait_reasons=["awaiting_information"],
        offered_exit_ids=[e.exit_id for e in exits],
        exit_options={e.exit_id: e for e in exits},
        route_blocked=False,
        cues=[],
        current_sim_time=t,
        offered_actions_set=set(actions),
        offered_wait_reasons_set={"awaiting_information"},
        offered_exit_ids_set={e.exit_id for e in exits},
        warnings=tuple(warnings),
        nearby_agent_ids=tuple(nearby),
    )


def decide(eng, c):
    result = asyncio.run(eng.decide(c))
    offered = payload.OfferedSet(
        tuple(c.offered_actions), tuple(c.offered_wait_reasons), tuple(c.offered_exit_ids)
    )
    assert payload.validate(result.payload, offered) == [], result.payload
    return result


class StageTests(unittest.TestCase):
    def test_without_cues_the_journey_continues(self):
        result = decide(engine(), ctx(10.0))
        self.assertEqual(result.stage, UNAWARE)
        # A boarder heads down toward the platform, as before.
        self.assertEqual(result.payload["exit_id"], "escalator_a_down")

    def test_a_cue_makes_the_agent_stop_and_investigate(self):
        result = decide(engine(), ctx(20.0, [cue(15.0, "weak")]))
        self.assertEqual(result.stage, AWARE)
        self.assertEqual(result.payload["action"], "wait")

    def test_aware_agents_seek_information_when_they_can(self):
        actions = ("continue_activity", "wait", "evacuate", "seek_information")
        result = decide(engine(), ctx(20.0, [cue(15.0, "weak")], actions=actions))
        self.assertEqual(result.payload["action"], "seek_information")

    def test_evacuation_follows_the_delay_and_leaves_the_station(self):
        eng = engine()
        warnings = [cue(15.0, "medium")]
        self.assertEqual(decide(eng, ctx(90.0, warnings)).stage, AWARE)  # 15 + 80 = 95
        result = decide(eng, ctx(100.0, warnings))
        self.assertEqual(result.stage, EVACUATING)
        self.assertEqual(result.payload["exit_id"], "grey_street")  # not down to the platform

    def test_a_stronger_later_cue_brings_evacuation_forward(self):
        eng = engine()
        decide(eng, ctx(20.0, [cue(15.0, "weak")]))  # would act at 415
        warnings = [cue(15.0, "weak"), cue(30.0, "strong", source="pa")]  # acts at 70
        self.assertEqual(decide(eng, ctx(75.0, warnings)).stage, EVACUATING)

    def test_a_weaker_later_cue_never_delays_evacuation(self):
        eng = engine()
        decide(eng, ctx(20.0, [cue(15.0, "strong")]))  # acts at 55
        warnings = [cue(15.0, "strong"), cue(30.0, "weak")]
        self.assertEqual(decide(eng, ctx(60.0, warnings)).stage, EVACUATING)

    def test_instruction_to_board_the_train_is_followed(self):
        actions = ("continue_activity", "wait", "evacuate", "leave_by_train")
        warnings = [cue(15.0, "strong", source="pa", instruction="board_train")]
        result = decide(engine(), ctx(60.0, warnings, actions=actions))
        self.assertEqual(result.payload["action"], "leave_by_train")

    def test_cues_heard_before_arrival_start_on_arrival(self):
        eng = engine()
        warnings = [cue(15.0, "strong")]
        spawned = {"spawn_time_s": 500.0}
        self.assertEqual(decide(eng, ctx(510.0, warnings, agent_cfg=spawned)).stage, AWARE)
        self.assertEqual(decide(eng, ctx(545.0, warnings, agent_cfg=spawned)).stage, EVACUATING)

    def test_a_first_decision_after_the_cue_does_not_delay_the_response(self):
        # Present from the start, first deciding at t=60: the alarm at 15 counts from 15.
        self.assertEqual(decide(engine(), ctx(60.0, [cue(15.0, "strong")])).stage, EVACUATING)


class SocialCueTests(unittest.TestCase):
    def test_neighbours_evacuating_are_a_cue(self):
        eng = engine(social_threshold=0.5, social_min_neighbours=2, social_strength="strong")
        for n in ("n1", "n2"):
            decide(eng, ctx(0.0, [cue(0.0, "strong")], agent=n))
            self.assertEqual(
                decide(eng, ctx(50.0, [cue(0.0, "strong")], agent=n)).stage, EVACUATING
            )
        # 'a' has no cue of its own, but both neighbours have been evacuating.
        self.assertEqual(decide(eng, ctx(60.0, agent="a", nearby=("n1", "n2"))).stage, AWARE)
        self.assertEqual(decide(eng, ctx(101.0, agent="a", nearby=("n1", "n2"))).stage, EVACUATING)

    def test_neighbours_who_only_just_started_do_not_count_yet(self):
        eng = engine(social_min_neighbours=1)
        decide(eng, ctx(0.0, [cue(0.0, "strong")], agent="n1"))
        decide(eng, ctx(50.0, [cue(0.0, "strong")], agent="n1"))  # evacuating since t=50
        self.assertEqual(decide(eng, ctx(50.0, agent="a", nearby=("n1",))).stage, UNAWARE)


class DelayTests(unittest.TestCase):
    def test_delays_are_reproducible_and_differ_between_agents(self):
        def evacuate_at(seed, agent):
            eng = RuleBasedDecisionEngine(response_sigma=0.6, seed=seed)
            decide(eng, ctx(15.0, [cue(15.0, "medium")], agent=agent))
            return eng._agents[agent].evacuate_at

        self.assertEqual(evacuate_at(1, "a"), evacuate_at(1, "a"))
        self.assertNotEqual(evacuate_at(1, "a"), evacuate_at(1, "b"))
        self.assertNotEqual(evacuate_at(1, "a"), evacuate_at(2, "a"))


if __name__ == "__main__":
    unittest.main()
