# Recommended Training And Evaluation Workflow

Run every command from the repository root. These are the current recommended
simple-model pipelines. Older collectors, trainers, and top-level evaluators
remain for compatibility and experiments, but should not be used for a new
standard run.

Use `MUJOCO_GL=egl` on this Linux environment. Setting `MPLCONFIGDIR` avoids
Matplotlib cache warnings and slow imports.

## 2-Skill Pipeline

### 1. Collect balanced completed rallies

```bash
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/matplotlib \
  venv/bin/python collect_data_v2.py \
  --rallies 100 \
  --output data/rallies_v2_2skill.pkl
```

This collects 100 completed rallies for each of the 4 skill pairs and writes
76-dimensional encoded states.

### 2. Train Q-value estimators and potential function

```bash
venv/bin/python train_q_model_v2.py \
  --rallies data/rallies_v2_2skill.pkl \
  --epochs 1500
```

Outputs:

- `models/model1_76dim.pth`
- `models/model2_76dim.pth`
- `models/model_p_76dim.pth`

Do not use `train_q_model.py` for the current pipeline. Treat
`train_q_model_v3.py` and the factored trainer as experiments.

### 3. Evaluate across multiple seeds

```bash
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/matplotlib \
  venv/bin/python nash_skills/eval_matchup_2skill.py \
  --model 76dim \
  --episodes 60 \
  --steps 6000 \
  --seed 0 \
  --num-seeds 5
```

The evaluator defaults to `--model 76dim`, but specifying it makes the selected
pipeline explicit. Results are written under `skill_eval/`.

## 5-Skill Pipeline

### 1. Collect balanced completed rallies

```bash
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/matplotlib \
  venv/bin/python nash_skills/v2/collect_data.py \
  --rallies 50 \
  --output data/rallies_5skill_v2.pkl
```

This collects 50 completed rallies for each of the 25 skill pairs and writes
76-dimensional encoded states. Step-capped episodes are discarded because
their unknown winner would create zero-valued Q targets.

### 2. Train v3 Q-value estimators and potential function

```bash
venv/bin/python nash_skills/v2/train_q_model_5skill_v3.py \
  --rallies data/rallies_5skill_v2.pkl \
  --epochs 1500
```

Outputs:

- `models/model1_5skill_v3.pth`
- `models/model2_5skill_v3.pth`
- `models/model_p_5skill_v3.pth`

The v3 trainer is preferred over the older 5-skill trainers because it trains
the potential function with same-state, per-sample counterfactual constraints.

### 3. Run a single matchup evaluation sweep

```bash
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/matplotlib \
  venv/bin/python nash_skills/eval_matchup.py \
  --v3-5skill \
  --episodes 60 \
  --steps 6000
```

### 4. Run the recommended multi-seed evaluation

```bash
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/matplotlib \
  venv/bin/python nash_skills/eval_multiseed.py \
  --v3-5skill \
  --seeds 0 1 2 3 4 \
  --episodes 60 \
  --steps 6000 \
  --output-dir skill_eval/multiseed_v3
```

Use the multi-seed results for model comparisons and final reporting.

## Quick Command Check

Before a long run, verify that the environment and command imports work:

```bash
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/matplotlib \
  venv/bin/python collect_data_v2.py --help
venv/bin/python train_q_model_v2.py --help
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/matplotlib \
  venv/bin/python nash_skills/v2/collect_data.py --help
venv/bin/python nash_skills/v2/train_q_model_5skill_v3.py --help
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/matplotlib \
  venv/bin/python nash_skills/eval_multiseed.py --help
```
