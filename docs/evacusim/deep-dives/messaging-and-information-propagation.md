# Deep Dive: Messaging and Information Propagation

## Messaging Role in Behavior Dynamics

Messaging is the social-information layer that alters what agents know independently of direct line-of-sight.

Key components:

- [MessageSystem.extract_and_deliver_message](../function-reference/systems__messaging__message_system.md)
- [MessageParser](../function-reference/systems__messaging__message_parser.md)
- [MessageMemory](../function-reference/systems__messaging__message_memory.md)
- [ConversationTracker](../function-reference/systems__messaging__conversation_tracker.md)

## Message Flow

Typical path:

1. Agent emits JSON action with optional communication fields.
2. Message parser extracts message text/type/target data.
3. Message system enforces cooldown and duplicate controls.
4. Recipients are selected by range and eligibility.
5. Message is written into recipient context and history.
6. Observation generation includes message content in later prompts.

## Propagation Controls

Built-in controls reduce spam and unrealistic broadcast saturation:

- per-sender cooldown,
- duplicate suppression via short-term memory,
- per-step global shout cap,
- recipient filtering for exited/non-active agents.

These controls are critical for prompt-size control and decision quality.

## Coupling With Observation and Exit Knowledge

Message content can update perceived options:

- [ObservationGenerator.generate_observation](../function-reference/translation__observation_generator.md)
- [ObservationCoordinator.generate_all_observations](../function-reference/coordination__observation_coordinator.md)

In particular, message-derived exit mentions can affect an agent's known exits and future route decisions.

## Design Tradeoffs

- Strong controls improve stability and cost predictability.
- Heavy suppression may hide emergent communication behavior.
- Message schema flexibility is useful but can weaken strict validation.

## Review Guidance

1. Verify cooldown and dedup windows against desired realism.
2. Confirm target resolution behavior for direct vs local broadcasts.
3. Validate parser robustness on malformed or partial action JSON.
4. Check prompt growth from message histories in long runs.
