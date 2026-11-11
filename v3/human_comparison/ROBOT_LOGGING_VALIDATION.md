# Optional measured robot landing inputs

The released comparison code works from saved decisions. It does not run robot
simulations, instrument the evaluator, or require model checkpoints. Existing
Nash-Q-versus-Random and Potential-IBR-versus-Random logs contain selected
skills and decision-state coordinates, but no measured bounce coordinates.

Consequently the main comparison uses **human realized landing proxies** and
**robot selected high-level skills**. Planner predictions, nominal skill
targets, successful-return print lines, and inferred contact/miss flags are
not measured landings. Missing robot landing outcomes stay unknown, not zero
or failed landings.

`analyze_spatial_v2.py` can optionally read a `realized_shot_outcome_json`
column produced by separate simulator instrumentation. The supported record
contains:

- `observation_complete`: whether the physical outcome was resolved.
- `valid_opponent_landing`: whether the observed landing is on the opponent half.
- `landing_depth_normalized`: 0 at the net, 1 at the opponent endline.
- `landing_lateral_normalized`: +1 hitter-left, -1 hitter-right.
- `racket_contacts`: measured contact records, not predicted contacts.
- `exclusion_reason`: why the outcome is unavailable or excluded.

Such instrumentation must associate each selected action with its actual
racket contact and subsequent physical table bounce, handle duplicate and
ambiguous contacts, retain raw coordinates, and leave policy decisions and
physics unchanged. Validate paired seeded runs with logging disabled/enabled
before collecting large evaluations. That simulator work is separate from
this saved-data analysis package; it is not needed to reproduce the current
selected-skill comparisons.

Use identical landing-region definitions for measured human and robot
landings when robot measurements become available. Keep selected robot skills
separate so command-to-realized-placement variability remains inspectable.
