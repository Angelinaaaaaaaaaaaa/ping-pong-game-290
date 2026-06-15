# 5-Skill Nash Pipeline

This folder extends the original 2-skill (left/right) pipeline with three additional skills, following the plan in `docs/CLAUDE.md`.

## Skills

| Skill | `side_target` | `x_target` | Description |
|-------|--------------|------------|-------------|
| `left`        | -1.0 | ~2.19 m | Deep return to opponent's left  (original) |
| `right`       | +1.0 | ~2.19 m | Deep return to opponent's right (original) |
| `left_short`  | -1.0 | ~1.65 m | Short return, left side |
| `right_short` | +1.0 | ~1.65 m | Short return, right side |
| `center_safe` |  0.0 | ~1.85 m | Conservative center return |

`side_target` controls the lateral aim: the environment passes `y_target = side_target * 0.38` to the racket pose solver.  
`x_target` controls depth: how far into the opponent's table half the ball should land.

## Files

| File | Purpose |
|------|---------|
| `skills.py` | Skill name → `(side_target, x_target)` mapping |
| `env_wrapper.py` | `SkillEnv` wraps the base env with 5-skill support |
| `collect_data_5skill.py` | Roll out PPO policy over all 25 skill combos, save rallies |
| `train_q_model_5skill.py` | Train Q-value estimators + potential function |
| `comp_5skill.py` | Run competition with chosen strategies |

## Full Pipeline — Step by Step

All commands should be run from the **project root** (`ping-pong-game-290/`), not from inside `nash_skills/`.

---

### Step 0 — Prerequisites (do once)

Make sure the environment is set up and the base PPO policy is trained:

```bash
# activate your venv / conda env first, then:
pip install mujoco stable-baselines3 torch gymnasium scipy numpy
```

The PPO policy at `logs/best_model_tracker1/best_model` must already exist.  
If it does not, train it first:

```bash
python train.py          # Ubuntu
mjpython train.py        # macOS  (takes a long time — 500M steps)
```

---

### Step 1 — Collect 5-Skill Rally Data

Rolls out the trained PPO policy for every combination of the 5 skills (25 combos × 20 000 steps each).  
Saves to `data/rallies_5skill.pkl`.

```bash
# Ubuntu
python nash_skills/collect_data_5skill.py

# macOS
mjpython nash_skills/collect_data_5skill.py
```

Expected output:
```
=== Collecting: ego=left  opp=left ===
=== Collecting: ego=left  opp=right ===
...
Saved 312 rallies to data/rallies_5skill.pkl
```

---

### Step 2 — Train Q-Value + Potential Models

Reads `data/rallies_5skill.pkl`.  
Trains `model1_5skill`, `model2_5skill` (Q-value estimators) and `model_p_5skill` (potential function).  
Saves to `models/`.

```bash
python nash_skills/train_q_model_5skill.py
```

No viewer needed — pure training, no MuJoCo rendering.  
Expected output every 100 epochs:
```
Q-epoch 100/1500  loss1=0.04231  loss2=0.03912
...
  [saved] epoch 45  potential_loss=0.00812
```

---

### Step 3 — Run Competition

Edit the strategy constants at the top of `nash_skills/comp_5skill.py`:

```python
STRATEGY1 = 'nash-p'    # ego player
STRATEGY2 = 'random'    # opponent
```

Available strategy values:

| Value | Description |
|-------|-------------|
| `'random'` | Uniformly random skill each rally |
| `'left'` | Always use left skill |
| `'right'` | Always use right skill |
| `'left_short'` | Always use left_short skill |
| `'right_short'` | Always use right_short skill |
| `'center_safe'` | Always use center_safe skill |
| `'nash-p'` | Approximate Nash via potential maximisation |

Then run:

```bash
# Ubuntu
python nash_skills/comp_5skill.py

# macOS
mjpython nash_skills/comp_5skill.py
```

The MuJoCo viewer opens automatically.  
Terminal prints the chosen skill at each rally crossing:
```
Rally crossing — ego: left_short  opp: right
Rally crossing — ego: center_safe  opp: left_short
...
```

---

## Comparing Against Baselines

Run the same script with different strategy pairs to compare:

| Experiment | STRATEGY1 | STRATEGY2 |
|------------|-----------|-----------|
| Nash vs Random | `'nash-p'` | `'random'` |
| Nash vs Nash | `'nash-p'` | `'nash-p'` |
| Nash vs fixed left | `'nash-p'` | `'left'` |
| Random vs Random | `'random'` | `'random'` |
| Old 2-skill baseline | use original `comp.py` | — |

The original 2-skill competition is still in `comp.py` (project root) — it is unchanged.
