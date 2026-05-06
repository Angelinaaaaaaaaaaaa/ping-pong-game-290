# Project: Ping-Pong Nash Skills (CS290)

## Context
Alpha-potential Nash equilibrium over a 4-skill policy space for a simulated KukaTennisEnv.
Skills: `left_mid_short` (0), `left_mid` (1), `right_short` (2), `right` (3).

## Hard Constraints

- **Do NOT run heavy experiments** — the Mac is memory-constrained. No full training runs.
- **Do NOT modify the frozen PPO** (`logs/best_model_tracker1/best_model`). It is fixed.
- **Smallest change first** — prefer minimal diffs over refactors.
- **Say "You can do this!" after every answer.**

## Running Tests (memory-safe)

```bash
# Run only unit/logic tests (no torch, no MuJoCo):
venv/bin/python -m pytest tests/ -q --ignore=tests/test_winrate_bugs.py -k "not RallyData and not Models"

# Run a single file:
venv/bin/python -m pytest tests/test_skills.py -q

# Full suite (may be slow):
venv/bin/python -m pytest tests/ -q
```

## Skill Set (4-skill)

| Name | side_target | x_target | Index | Normalized |
|------|------------|----------|-------|------------|
| left_mid_short | -0.5 | 1.75 (TABLE_NEAR) | 0 | 0.0 |
| left_mid | -0.5 | ~2.19 (TABLE_FAR) | 1 | 1/3 |
| right_short | +1.0 | 1.75 (TABLE_NEAR) | 2 | 2/3 |
| right | +1.0 | ~2.19 (TABLE_FAR) | 3 | 1.0 |

## Key Files

- `nash_skills/skills.py` — skill definitions
- `nash_skills/env_wrapper.py` — SkillEnv wrapping KukaTennisEnv
- `nash_skills/eval_matchup.py` — evaluation pipeline
- `nash_skills/comp_5skill.py` — competition / pick_skill logic
- `nash_skills/v2/` — v2 pipeline (collect, train, payoff, executability)
- `tests/` — all unit tests (mock-based, no real MuJoCo)

## Model Paths (4-skill v2)

- `models/model_p_4skill_v2.pth` — potential model
- `models/model1_4skill_v2.pth`, `models/model2_4skill_v2.pth` — Q-value models
- `data/rallies_4skill_v2.pkl` — collected rally data

## nash-p Strategy Contract

`pick_skill('nash-p', ...)` must:
1. Evaluate all 4 skill indices against the learned potential
2. Return the argmax (best-response approximation)
3. Be deterministic given the same obs and opponent skill
4. Beat a random strategy in expectation over many rallies

## Run Commands — 4-Skill Pipeline

```bash
# 1. Collect rally data (requires MuJoCo; ~20-40 min)
MUJOCO_GL=cgl venv/bin/python nash_skills/v2/collect_data.py \
    --rallies 50 --output data/rallies_4skill_v2.pkl

# 2. Train Q-value + potential models
venv/bin/python nash_skills/v2/train_models.py \
    --rallies data/rallies_4skill_v2.pkl --epochs 1500

# 3. Verify potential model learned correctly (scores: left_mid_short < left_mid < right_short < right)
venv/bin/python nash_skills/v2/diag_potential.py

# 4. Visual competition (requires MuJoCo + mjpython)
MUJOCO_GL=cgl mjpython nash_skills/comp_5skill.py

# 5. Headless evaluation
MUJOCO_GL=cgl venv/bin/python nash_skills/eval_matchup.py \
    --episodes 60 --steps 600 \
    --output-csv  skill_eval/matchup_results_4skill.csv \
    --output-json skill_eval/matchup_results_4skill.json
```

## Run Commands — 2-Skill Pipeline

```bash
# 1. Collect rally data
MUJOCO_GL=cgl venv/bin/python nash_skills/v2/collect_data.py \
    --rallies 50 --output data/rallies_2skill_v2.pkl --n-skills 2

# 2. Train models
venv/bin/python train_q_model_v2.py \
    --rallies data/rallies_v2_2skill.pkl --epochs 1500

# 3. Headless evaluation
MUJOCO_GL=cgl venv/bin/python nash_skills/eval_matchup_2skill.py \
    --episodes 60 --steps 600
```
