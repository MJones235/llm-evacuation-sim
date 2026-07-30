# Deep Dive: Line of Sight and Exit Visibility

## Core Spatial Components

- [SpatialAnalyzer.identify_zone](../function-reference/translation__spatial_analyzer.md)
- [SpatialAnalyzer.get_visible_exits](../function-reference/translation__spatial_analyzer.md)
- [SpatialAnalyzer._has_line_of_sight](../function-reference/translation__spatial_analyzer.md)
- [ObservationGenerator.generate_observation](../function-reference/translation__observation_generator.md)

## Zone Resolution

Zone identification uses ordered geometric tests to avoid broad polygons masking higher-priority semantic zones (for example platform regions and connectors).

## Visible Exit Resolution

`get_visible_exits` generally performs:

1. Candidate exit collection from level geometry and access links.
2. Candidate filtering for inactive or blocked exits.
3. Platform-side filtering where applicable.
4. Geometric line-of-sight checks.
5. Distance categorization and deduplication.

## Line-of-Sight Mechanics

Primary checks in `_has_line_of_sight`:

- Sight segment must be covered by walkable geometry union.
- Sight segment must not intersect blocking obstacles (subject to level-specific logic).

The walkable-union cache improves repeated LOS performance within frequent observation updates.

## Interaction With Memory and Messaging

Observation path integrates spatial visibility with:

- agent remembered exits from prior messages,
- blocked-exit memory persistence,
- ongoing event context.

Key coordinator:
- [ObservationCoordinator.generate_all_observations](../function-reference/coordination__observation_coordinator.md)

## Design Implications

- Visibility is physically constrained but also policy-filtered (blocked/inactive/platform-side).
- Inference quality depends heavily on geometry naming conventions and consistency.
