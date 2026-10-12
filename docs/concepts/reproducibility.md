# Reproducibility

The same configuration, seed and code give the same run, byte for byte, with
the rule-based engine or the fake LLM. A real LLM samples, so its runs repeat
only in distribution.

**One seed.** A run has a single `seed`. Each source of randomness draws from
its own stream derived from it by name (`evacusim.utils.seeding.derive_seed`):

| Stream | Randomises |
|---|---|
| `population` | who is sampled (profile, personality, role) and where they start |
| `calibration` | arrival times and spawn jitter in calibration runs |
| `escalators` | lane choice, stander step gaps |
| `decisions` | rule-engine response delays, one stream per person |
| `global` | anything still using the `random` module |

Adding randomness to one component never shifts another's. The streams do not
depend on `PYTHONHASHSEED`: everything that is iterated is ordered first.

**Decide all, then apply.** Decisions in a cycle are applied in a fixed order,
whatever order concurrent LLM calls complete in
([ADR 4](../adr/0004-decide-then-apply.md)).

**Self-describing runs.** `manifest.json` records:

- the resolved parameters, defaults included;
- the seed and the command line;
- the versions of Python, evacusim, JuPedSim and pydantic;
- the git commit of evacusim and of the study repository, each with a `dirty`
  flag.

A run made from uncommitted changes says so.

**Golden runs.** A study repository should keep short seeded runs whose
outputs are hashed and compared with committed snapshots. Then any change in
behaviour is noticed, and re-recorded only on purpose. monument-evacuation
does this for a rule-based drill, a fake-LLM drill and a calibration window.
