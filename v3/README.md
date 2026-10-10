# V3 Rotating Attacker/Defender Shot and Pre-Position Pipeline

This directory models a rally as a sequence of rotating attacker/defender
decisions. At every net crossing, the player receiving the ball becomes the
current attacker and the other physical player becomes the current defender.
The attacker selects one of five semantic shot skills while the defender
selects one of three semantic gantry pre-positions:

```text
Q(canonical_state, attack_skill, defense_preposition)
```

The selected pair is executed in the next rally segment: the attacker uses
the existing PPO racket controller for the selected shot, and the defender's
gantry is driven toward the selected ready position.

```text
left
center
right
```

These are not return skills. They only drive the defender's gantry toward a fixed ready target while the attacker executes the existing shot skill.

## Reused Existing Logic

v3.0 reuses:

```text
mujoco_env_comp.KukaTennisEnv
nash_skills.env_wrapper.SkillEnv
nash_skills.skills.SKILL_NAMES and skill profiles
nash_skills.eval_matchup PPO observation builders
nash_skills.eval_matchup stdout/contact parsing
nash_skills.eval_matchup terminal winner inference
```

The current five shot skills are not changed:

```text
left
left_short
center_safe
right_short
right
```

## V3.1 Behavior

`v3.env_wrapper.V3PrePositionEnv` subclasses the current `SkillEnv`.

The attacker shot skill still uses the current target-pose calculation and frozen PPO action. The defender pre-position is implemented by overriding only the defender's gantry action components before calling the existing `SkillEnv.step()`:

```text
P1 gantry: action[0], action[1] drives qpos[0], qpos[1]
P2 gantry: action[9], action[10] drives qpos[9], qpos[10]
```

The wrapper does not teleport the robot and does not directly write `qpos`.

`v3/eval_fixed_preposition.py` runs fixed attacker-shot x defender-pre-position sweeps. `--all-combinations` covers all 15 combinations:

```text
5 attacker skills x 3 defender prepositions
```

`v3/collect_random_preposition.py` samples both variables uniformly at every decision event and logs one row per decision for future model-design work. It does not train a model.

## Initial Pre-Position Coordinates

Coordinates live in `v3/constants.py`.

The existing environment clips both players' gantry coordinates to:

```text
x in [-1.0, -0.5]
y in [-1.0,  1.0]
```

Ready reset uses:

```text
x = -0.5
y =  0.0
```

V3 semantic defensive targets are physical/world coordinates:

```text
P1 left   = (-0.5, +0.35)    P2 left   = (3.5, -0.35)
P1 center = (-0.5,  0.00)    P2 center = (3.5,  0.00)
P1 right  = (-0.5, -0.35)    P2 right  = (3.5, +0.35)
```

These targets are defined from the defender's own physical perspective. The
P2 target is converted to raw gantry qpos before control: for example, P2
left world y `-0.35` becomes raw qpos y `+0.35`. This is a coordinate-frame
conversion, not a semantic left/right relabeling.

## Timing

At each net crossing:

1. capture the current decision state
2. identify the receiver as the next attacker
3. set that attacker to the fixed existing shot skill
4. commit the other player to the fixed defensive pre-position
5. execute normal PPO-controlled racket motion while the defender's gantry action moves toward the committed pre-position

The defender position is committed at the crossing, before the following return outcome is observed.

## Single Combination

```bash
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/matplotlib \
venv/bin/python v3/eval_fixed_preposition.py \
  --attack-skill left \
  --defense-position right \
  --episodes 100 \
  --steps 600 \
  --warmup 300 \
  --reset-mode ready \
  --skill-profile aggressive \
  --gantry-speed-scale 1.0 \
  --output-csv skill_eval/v3/left_vs_defense_right.csv \
  --output-json skill_eval/v3/left_vs_defense_right.json
```

## Live Visualization

The existing live viewer path from `comp.py` is:

```python
env.render()
time.sleep(0.01)
```

`env.render()` delegates to the current MuJoCo environment, which launches `mujoco.viewer.launch_passive(...)` and synchronizes with `viewer.sync()`.

Use `--render` to enable that same live viewer during v3 fixed pre-position evaluation. Do not set `MUJOCO_GL=egl` for this mode.

```bash
MPLCONFIGDIR=/tmp/matplotlib \
venv/bin/python v3/eval_fixed_preposition.py \
  --attack-skill left \
  --defense-position right \
  --episodes 5 \
  --steps 600 \
  --warmup 0 \
  --render
```

During `--render`, decision debug lines are printed automatically. To print the same role trace without opening the viewer, add `--debug-decisions`:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
venv/bin/python v3/eval_fixed_preposition.py \
  --attack-skill left \
  --defense-position right \
  --episodes 2 \
  --steps 300 \
  --warmup 0 \
  --debug-decisions
```

Each decision line has this form:

```text
decision 1: attacker=P1 skill=left defender=P2 pre_position=right target_world=(3.500, 0.350) target_qpos=(-0.500, -0.350)
decision 2: attacker=P2 skill=left defender=P1 pre_position=right target_world=(-0.500, -0.350) target_qpos=(-0.500, -0.350)
```

The attacker/defender roles are derived from the existing net-crossing event and the side the ball just entered. They are not a permanent P1/P2 assignment.

## Short Smoke Test For All 15 Combinations

```bash
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/matplotlib \
venv/bin/python v3/eval_fixed_preposition.py \
  --all-combinations \
  --episodes 2 \
  --steps 300 \
  --warmup 50 \
  --max-total-steps 10000 \
  --reset-mode ready \
  --skill-profile aggressive \
  --gantry-speed-scale 1.0 \
  --output-csv skill_eval/v3/smoke_fixed_preposition_summary.csv \
  --output-json skill_eval/v3/smoke_fixed_preposition_results.json
```

## Larger Fixed 15-Combination Sweep

```bash
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/matplotlib \
venv/bin/python v3/eval_fixed_preposition.py \
  --all-combinations \
  --episodes 100 \
  --steps 600 \
  --warmup 300 \
  --reset-mode ready \
  --skill-profile aggressive \
  --gantry-speed-scale 1.0 \
  --output-csv skill_eval/v3/full_fixed_preposition_summary.csv \
  --output-json skill_eval/v3/full_fixed_preposition_results.json
```

The evaluator writes:

```text
summary CSV
per-episode CSV with `_episodes` in the filename
JSON containing both summaries and per-episode rows
```

Each episode row includes the attacker skill, defender position, target gantry coordinate, actual defender gantry coordinate at commit/receive/contact/final moments when accessible, winner, return success counts, termination reason, rally length, and truncation status. The row-level receive fields include the associated defender and decision index. The JSON `decision_events` records the unambiguous per-decision target, coordinate frame, commit position, receive position, and contact position; use these event fields when comparing target versus actual position.

The summary CSV is grouped by `attack_skill x defense_position` and includes all-combination rows plus P1/P2 split rows for asymmetry checks.

Pass `--planner optimized` to run the same fixed sweep with the V3 optimized
racket-target planner. The planner is printed at startup and stored in the
episode CSV, summary CSV, and JSON metadata. When the default output paths are
used, optimized runs are written to `fixed_preposition_summary_optimized.csv`
and `fixed_preposition_results_optimized.json` so they do not overwrite legacy
artifacts. Explicit `--output-csv` and `--output-json` paths are respected.

For example, a short optimized 15-combination sweep is:

```bash
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/matplotlib \
venv/bin/python v3/eval_fixed_preposition.py \
  --planner optimized \
  --all-combinations \
  --episodes 2 \
  --steps 300 \
  --warmup 50 \
  --max-total-steps 10000 \
  --output-csv skill_eval/v3/smoke_fixed_preposition_summary_optimized.csv \
  --output-json skill_eval/v3/smoke_fixed_preposition_results_optimized.json
```

The random V3 collector and planner ablation evaluator expose the same planner
choices. Use `--planner optimized` with
`v3/collect_random_preposition.py` to recollect data under the optimized
planner; its default output paths receive the same `_optimized` suffix.

## Short Random Collection Run

```bash
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/matplotlib \
venv/bin/python v3/collect_random_preposition.py \
  --decisions 100 \
  --max-rallies 50 \
  --steps 600 \
  --reset-mode ready \
  --skill-profile aggressive \
  --gantry-speed-scale 1.0 \
  --progress-every 5 \
  --seed 0 \
  --output-csv skill_eval/v3/random_preposition_decisions_smoke.csv \
  --coverage-csv skill_eval/v3/random_preposition_coverage_smoke.csv \
  --output-json skill_eval/v3/random_preposition_decisions_smoke.json \
  --output-pkl data/v3/random_preposition_decisions_smoke.pkl
```

The collector logs:

```text
state_before_action
attacker_id
attack_skill
defender_id
defense_preposition
defender target and actual gantry positions
rally_id
decision_index
final rally winner
discounted return
```

It also writes coverage counts for all 15 shot x pre-position combinations.

The pickle output preserves structured NumPy arrays in the old collector style:

```text
states              list of 76-dim float32 encoded states
raw_obs             list of 116-dim float32 raw observations
skill_pairs         list of attack skill x defense pre-position labels
attacker_returns    per-decision attacker-perspective discounted returns
decisions           structured per-decision metadata
winner              physical rally winner label: 1=P1/ego, 2=P2/opp, 0=truncated
```

The existing `train_q_model_5skill_v3.py` preprocessing can consume the 76-dim `states` feature arrays, but its current label construction still uses the top-level physical `winner`. For V3 training, use `attacker_returns` or the per-decision `discounted_return` values rather than treating the old physical-player labels as the V3 attacker objective.

## V3.2 Labeling

Convert a V3 collector pickle into per-decision attacker-perspective examples:

```bash
python3 v3/label_preposition.py \
  --input-pkl data/v3/random_preposition_decisions_smoke.pkl \
  --output-pkl data/v3/labeled_preposition_examples.pkl \
  --output-csv skill_eval/v3/labeled_preposition_examples.csv \
  --gamma 0.7
```

Each labeled example contains:

```text
state                  76-dim float32 encoded state
attack_skill           one of the existing five return skills
defense_preposition    left, center, or right
attacker_return        discounted return from the current attacker's perspective
```

## Legacy scalar Q Training

Train `Q(state, attack_skill, defense_preposition) -> attacker_return`:

```bash
python3 v3/train_q_preposition.py \
  --labels data/v3/labeled_preposition_examples.pkl \
  --epochs 1500 \
  --lr 0.001 \
  --model-out models/v3/q_preposition.pth \
  --log-csv logs/v3/train_q_preposition.csv \
  --metrics-json skill_eval/v3/q_preposition_metrics.json \
  --bins-csv skill_eval/v3/q_preposition_bins.csv \
  --q-matrix-csv skill_eval/v3/q_preposition_sample_matrix.csv
```

This is the older scalar-action trainer. It is retained for comparison only;
use the one-hot trainer below for the current V3 model. Its input is:

```text
76 state features + normalized attack skill index + normalized defense pre-position index = 78 dims
```

Diagnostics include train/val/test loss, prediction-target correlation, predicted-Q bins vs empirical average return / attacker win rate, and one sample `5 x 3` Q matrix.

## Racket Planner Ablation

`v3/planner_env.py` keeps the existing five-skill `SkillEnv` path and adds a
planner switch. `legacy` delegates target updates to the original MuJoCo
methods. `optimized` searches multiple contact points and face normals, uses
the same reflection model, rejects unsafe net/table trajectories, and scores
normalized landing, reachability, paddle-center, orientation, and small
legacy-reference costs.

Run a bounded comparison with identical policies, matchup, seed, and reset
settings:

```bash
MPLCONFIGDIR=/tmp/mpl-pingpong venv/bin/python -u v3/eval_planner_ablation.py \
  --planner both \
  --episodes 10 \
  --steps 600 \
  --warmup 0 \
  --seed 17 \
  --v3-5skill \
  --output-dir skill_eval/v3/planner_ablation
```

Use `--matchup STRATEGY1 STRATEGY2` for one exact policy pair. Planner search
parameters are exposed as CLI flags, including contact/normal candidate
counts, safety margins, normalized cost weights/scales, and normal bounds.
The output directory contains per-planner config JSON, raw per-return attempt
CSV, the base evaluator CSV/JSON, `planner_skill_comparison.csv`, and
`planner_symmetry.csv`.

The optimized planner now treats analytic and grid reflection as candidate
generators. Their valid candidates share one landing-first cost and the
selected result reports `solution_method=explicit` or `solution_method=grid`.
Run a controlled fixed-skill benchmark with identical deterministic resets:

```bash
MPLCONFIGDIR=/tmp/mpl-pingpong venv/bin/python -u \
  v3/benchmark_planner_fixed_matchups.py \
  --planner both \
  --matchup left right \
  --episodes 20 \
  --seeds 0 1 2 \
  --steps 600 \
  --reset-mode clean \
  --skill-profile current \
  --output-dir skill_eval/v3/planner_fixed_matchup_benchmark
```

Use `--all-matchups` for all 25 fixed skill pairs. The benchmark reports
racket contact rate, legal-return rate, actual landing error to the requested
skill target, average rally length, miss/net/out rates, fixed-matchup win
rates, and explicit/grid selection counts.

The original `legacy` normal selection remains unchanged. Compare it with the
legality-filtered legacy variant using identical deterministic resets:

```bash
MPLCONFIGDIR=/tmp/mpl-pingpong venv/bin/python -u \
  v3/benchmark_planner_fixed_matchups.py \
  --compare-legacy-legality \
  --matchup left right \
  --episodes 20 \
  --seeds 0 1 2 \
  --steps 600 \
  --reset-mode clean \
  --skill-profile current \
  --output-dir skill_eval/v3/legacy_legality_benchmark
```

`legacy_legal` keeps the legacy contact point and pose construction, then
selects the closest landing among normals that clear the net and land inside
the opponent table. If none are legal, it records and uses the original
legacy normal selection. Attempt CSVs include candidate counts, selected
predicted landing, net-crossing height, landing error, and fallback status.

## Current canonical V3 pipeline

The current rotating attacker/defender pipeline uses a role-based canonical
state encoder:

```text
state dimension: 76
first player block: current attacker
second player block: current defender
spatial features: current attacker-relative canonical frame
attack action: 5-d one-hot
defense action: 3-d one-hot
model input: 76 + 5 + 3 = 84
```

The semantic actions are:

```text
attack:  left, left_short, center_safe, right_short, right
defense: left, center, right
```

Action names are defined from the acting player's physical perspective.
Execution converts semantic actions to world targets and then to
player-specific raw gantry coordinates. The Q model never outputs raw
coordinates.

### Canonical-state symmetry test

Run this before collecting a new canonical dataset:

```bash
python3 v3/diagnose_canonical_state_symmetry.py
```

The test reports maximum/mean absolute differences and the largest-difference
state indices for mirrored P1-attacker/P2-attacker situations.

### Defense semantic-coordinate sanity test

This checks that the semantic label, world y, raw qpos y, and player-relative
side agree for both physical players:

```bash
python3 v3/diagnose_defense_action_logging.py
```

Default output:

```text
skill_eval/v3/defense_action_logging_sanity.csv
```

Expected lateral mapping:

```text
P1 left   -> world y +0.35 -> raw qpos y +0.35
P1 right  -> world y -0.35 -> raw qpos y -0.35
P2 left   -> world y -0.35 -> raw qpos y +0.35
P2 right  -> world y +0.35 -> raw qpos y -0.35
```

### Recollect canonical V3 data

Use the existing PPO controller and collect random semantic attack/pre-position
decisions. This does not train a model:

```bash
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/matplotlib \
venv/bin/python v3/collect_random_preposition.py \
  --decisions 10000 \
  --max-rallies 5000 \
  --steps 600 \
  --gamma 0.7 \
  --reset-mode ready \
  --skill-profile aggressive \
  --gantry-speed-scale 1.0 \
  --progress-every 100 \
  --seed 0 \
  --output-csv skill_eval/v3/random_preposition_canonical_decisions.csv \
  --coverage-csv skill_eval/v3/random_preposition_canonical_coverage.csv \
  --output-json skill_eval/v3/random_preposition_canonical_decisions.json \
  --output-pkl data/v3/random_preposition_canonical_decisions.pkl
```

For a small collection smoke test, use `--decisions 100` and
`--max-rallies 50`.

### Label attacker-perspective returns

Use only data produced by the current canonical collector:

```bash
python3 v3/label_preposition.py \
  --input-pkl data/v3/random_preposition_canonical_decisions.pkl \
  --output-pkl data/v3/labeled_preposition_canonical_examples.pkl \
  --output-csv skill_eval/v3/labeled_preposition_canonical_examples.csv \
  --gamma 0.7
```

The labeler rejects data with an incompatible state-encoder name or missing
per-decision metadata. Do not reuse old pre-canonical V3 data for final
training.

### Train the current one-hot Q model

```bash
python3 v3/train_q_preposition_onehot.py \
  --labels data/v3/labeled_preposition_canonical_examples.pkl \
  --epochs 1500 \
  --lr 0.001 \
  --seed 0 \
  --model-out models/v3/q_preposition_onehot_canonical.pth \
  --log-csv logs/v3/train_q_preposition_onehot_canonical.csv \
  --metrics-json skill_eval/v3/q_preposition_onehot_canonical_metrics.json \
  --bins-csv skill_eval/v3/q_preposition_onehot_canonical_bins.csv \
  --q-matrix-csv skill_eval/v3/q_preposition_onehot_canonical_sample_matrix.csv
```

The saved model expects exactly 84 input features: 76 canonical state values,
5 one-hot attack-skill values, and 3 one-hot defender-position values.

### Evaluate rotating attacker/defender Q policies

The evaluator supports `nash_q`, `greedy_q`, `random`, and `fixed` policies.
Roles switch at each net crossing: the receiver becomes the attacker and the
other physical player becomes the defender. It writes episode and decision
CSV/JSON files, Q matrices, selected semantic actions, policy probabilities,
outcome diagnostics, and termination reasons.

Small Nash-Q versus random smoke test:

```bash
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/matplotlib \
venv/bin/python v3/eval_q_policy.py \
  --model models/v3/q_preposition_onehot_canonical.pth \
  --ppo logs/best_model_tracker1/best_model \
  --p1-policy nash_q \
  --p2-policy random \
  --episodes 2 \
  --steps 300 \
  --seed 0 \
  --output-dir skill_eval/v3/q_policy_eval_smoke
```

Rendered one-episode diagnostic:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
venv/bin/python v3/eval_q_policy.py \
  --model models/v3/q_preposition_onehot_canonical.pth \
  --ppo logs/best_model_tracker1/best_model \
  --p1-policy nash_q \
  --p2-policy random \
  --episodes 1 \
  --steps 300 \
  --render \
  --debug-decisions \
  --output-dir skill_eval/v3/q_policy_eval_render_smoke
```

Do not set `MUJOCO_GL=egl` when using `--render`; it requires the live viewer.

The decision logs include both semantic labels and coordinate diagnostics:

```text
attack_semantic_action
attack_player_relative_side
attack_target_local_y
attack_target_world_x / attack_target_world_y
defense_semantic_action
defense_preposition
defense_world_y
defender_raw_qpos_y
defender_actual_world_y
defense_target_player_relative_side
defense_player_relative_side
```

`defense_preposition` is the selected semantic action. It is not a claim that
the robot had already reached the target at the instant the decision was
logged; use the explicit actual-position fields for that distinction.

## V3 dependencies and scope

V3 is logically isolated, but it is not a standalone copy of the repository.
It intentionally imports these shared components:

```text
nash_skills.env_wrapper / nash_skills.skills / nash_skills.eval_matchup
nash_skills.v2.labeling
model_arch.py
MuJoCo model/environment files used by the shared environment
stable-baselines3 PPO checkpoint
PyTorch, SciPy, NumPy, and MuJoCo Python packages
```

Therefore, committing only `v3/` preserves the V3 implementation, but running
it still requires the shared repository files and external artifacts listed in
the commands above. The V3 commit does not include datasets, trained models,
logs, or generated `skill_eval/` outputs unless those paths are staged
separately.
