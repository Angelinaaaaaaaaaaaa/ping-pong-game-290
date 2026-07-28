# Probabilistic Skill Selection

> No MuJoCo required for any command in this document.

---

## Motivation

The current Nash potential pipeline picks skills deterministically:

```
ego_skill = argmax_s  Φ(state, s, opp_skill)
```

When the Φ surface is flat — i.e., the model has not learned to discriminate between skills — argmax always returns the same skill (the first one after a tie-break). This creates a circular bias: the dominant skill appears most in the collected data → the model learns to assign it the highest Φ → it is chosen most → repeat.

Probabilistic selection breaks the cycle by injecting controlled exploration. Four modes are available, ranging from fully deterministic to fully random.

---

## Modes

| Mode | Description | Parameters |
|---|---|---|
| `argmax` | Deterministic argmax (default). Preserves existing eval behavior including the confidence-margin softmax fallback. | — |
| `softmax` | Sample from `softmax(scores / T)`. Low T → sharp; high T → uniform. | `--temperature T` |
| `epsilon_argmax` | Argmax with ε-greedy uniform noise. With probability ε, pick uniformly at random. | `--epsilon ε` |
| `epsilon_softmax` | Softmax with ε-uniform mixing: `result = (1-ε)·softmax + ε·uniform`. | `--temperature T`, `--epsilon ε` |

All modes reduce to `argmax` when `epsilon=0` and `temperature→0`.

---

## Implementation

### Core module: `nash_skills/v2/skill_selection.py`

Three composable functions:

```python
from nash_skills.v2.skill_selection import (
    softmax_probs,           # values → probability distribution
    epsilon_mix_probs,       # mix any distribution with uniform
    select_skill_from_values,  # values → int skill index
)
```

**`softmax_probs(values, temperature=1.0)`**
- Applies max-shift for numerical stability before computing softmax.
- Raises `ValueError` if `temperature <= 0`.

**`epsilon_mix_probs(base_probs, epsilon, num_skills)`**
- `result = (1 - epsilon) * base_probs + epsilon * uniform`
- Raises `ValueError` if `epsilon` is outside `[0, 1]`.

**`select_skill_from_values(values, mode='argmax', temperature=1.0, epsilon=0.0, rng=None)`**
- Accepts NumPy arrays, lists, and PyTorch tensors.
- Returns a plain `int`.
- Raises `ValueError` for unknown modes.
- If `rng=None`, creates a fresh non-reproducible Generator internally.

### Integration in `eval_matchup.py`

New CLI flags (all additive — existing behavior unchanged when omitted):

```
--selection-mode  {argmax, softmax, epsilon_argmax, epsilon_softmax}  (default: argmax)
--temperature     float  softmax temperature  (default: 1.0)
--epsilon         float  exploration rate [0,1]  (default: 0.0)
--seed            int    RNG seed for reproducibility  (default: None)
```

The dispatch happens inside `make_picker` via a `_dispatch(action_scores)` closure:
- `argmax` mode: calls the existing `_pick_with_softmax_fallback` (confidence-margin logic preserved). **Default behavior is unchanged unless `--selection-mode` is explicitly passed.**
- All other modes: calls `select_skill_from_values` from `skill_selection.py`.

Both `pick1` and `pick2` share the same `selection_mode`/`temperature`/`epsilon` values, but each player now gets an **independent** `numpy.random.Generator` (fixed — previously both players shared one `rng` instance, coupling their stochastic draws). Given `--seed`, `run_matchup` derives `rng1 = seed + 2*matchup_idx` and `rng2 = seed + 2*matchup_idx + 1`, so results are reproducible per matchup without correlating player 1 and player 2's random choices, and without replaying identical draws across different matchups in the same run.

---

## Diagnostic Script

```bash
# Synthetic demo — no model or rally data needed
PYTHONPATH=. venv/bin/python nash_skills/v2/diag_skill_selection_probs.py

# With real model + rally data
PYTHONPATH=. venv/bin/python nash_skills/v2/diag_skill_selection_probs.py \
    --rallies data/rallies_5skill_v2.pkl \
    --model   models/model_p_5skill_v3.pth \
    --n-samples 200 \
    --temperature 0.5 \
    --epsilon 0.1 \
    --strategy hard    # or minimax
```

The script prints a table of average selection probabilities per skill under each mode. A column bias diagnostic is included: `max/min ratio >> 5×` suggests the model is dominated by a single skill and exploration would help.

`_load_model` auto-detects `state_dim` from the checkpoint's `fc.0.weight` shape, so the same script works for both 76-dim (v3, `_1000`, `_discard_30`, `_tie_30`) and 12-dim (gantry, gantry_sym) SimpleModel checkpoints. FactoredModel checkpoints (`*_factored.pth`) use a different architecture (`state_encoder`/`skill_encoder`/`fusion` branches, no `fc.0.weight`) and are not yet supported — the script detects this and falls back to the synthetic demo with a clear warning instead of crashing.

### Measured Collapse — Full Ablation (2026-07, `models_new/` × `data_new/`)

Initial runs against `data/rallies_5skill_v2.pkl` only covered 76-dim models — the 116-dim and 12-dim (gantry) models fell back to the synthetic demo because no matching rally data was available locally. Once `data_new/` was populated with the missing rally pickles (including the gantry-encoded ones), a new ablation runner (`nash_skills/v2/run_selection_ablation.py`) was built to auto-pair every model in a directory with a matching rally file by detected `state_dim`, and sweep the full model set in one command instead of one at a time.

```bash
PYTHONPATH=. venv/bin/python nash_skills/v2/run_selection_ablation.py \
    --models-dir models_new --rallies-dir data_new \
    --n-samples 200 --temperature 0.5 --epsilon 0.1 --strategy hard
```

Full results (strategy=hard, T=0.5, ε=0.1, n=200 sampled states):

| Model | Rallies used | dim | argmax dominant | argmax max/min | softmax max/min | epsilon_argmax max/min |
|---|---|---|---|---|---|---|
| `model_p.pth` | `rallies_5skill.pkl` | 116 | left | **∞** (100% argmax) | 1.0× | 46.0× |
| `model_p_5skill.pth` | `rallies_5skill.pkl` | 116 | right | 27.8× | 1.0× | 16.4× |
| `model_p_5skill_v2.pth` | `rallies_5skill_v2.pkl` | 76 | right | 19.1× | 1.2× | 12.7× |
| `model_p_5skill_v3.pth` | `rallies_5skill_v2.pkl` | 76 | left | 12.4× | 1.0× | 8.0× |
| `model_p_5skill_v3_1000.pth` | `rallies_5skill_v2.pkl` | 76 | left | 12.4× | 1.0× | 8.0× |
| `model_p_5skill_v3_discard_30.pth` | `rallies_5skill_v2.pkl` | 76 | left | 10.2× | 1.0× | 7.2× |
| `model_p_5skill_v3_tie_30.pth` | `rallies_5skill_v2.pkl` | 76 | right | 7.5× | 1.0× | 5.7× |
| `model_p_76dim.pth` | `rallies_5skill_v2.pkl` | 76 | left | **∞** (100% argmax) | 1.0× | 46.0× |
| `model_p_5skill_v3_gantry.pth` | `rallies_v2_1000_gantry.pkl` | 12 | right | **71.5×** | 2.1× | 22.9× |
| `model_p_5skill_v3_gantry_sym.pth` ⚠️ STALE | `rallies_v2_1000_gantry.pkl` | 12 | right | **190.0×** | 2.4× | 35.7× |
| `model_p_5skill_factored.pth`, `model_p_5skill_v3_factored.pth`, `model_p_factored.pth` | — | — | SKIPPED — FactoredModel architecture (no `fc.0.weight`) not yet supported by this script | | |

⚠️ `model_p_5skill_v3_gantry_sym.pth` was trained with the pre-fix, buggy `encode_opp_gantry` (see `lightweight_diagnostics.md` §3.2) — its ratio should be treated as unreliable until it's retrained on correctly re-encoded data. It was included here for completeness, not as a valid signal.

**Caveat on the table above**: `run_selection_ablation.py` pairs each model with *any* rally file of matching `state_dim`, not necessarily the file it was actually trained on — several models (`_1000`, `_discard_30`, `_tie_30`, `_gantry`) were evaluated against `rallies_5skill_v2.pkl` or `rallies_v2_1000_gantry.pkl` by dimension coincidence, not by provenance. See the corrected, name-matched results below for the authoritative numbers.

### Name-Matched Results (authoritative — matched by actual training data)

Cross-referenced each model against the `RALLY_PATH`/`MODEL_P_PATH` constants in its training script (`train_q_model_5skill.py`, `train_q_model_5skill_v2.py`, `train_q_model_5skill_v3.py`, `train_q_model_v3.py`) and the `--output-suffix` naming convention documented in `symmetrize_rallies.py`'s own printed next-step commands, then ran `diag_skill_selection_probs.py` directly against each model's actual training rallies:

```bash
PYTHONPATH=. venv/bin/python nash_skills/v2/diag_skill_selection_probs.py \
    --model models_new/<model>.pth --rallies data_new/<matching_rallies>.pkl \
    --n-samples 200 --temperature 0.5 --epsilon 0.1 --strategy hard
```

| Model | Actual training rallies | dim | argmax dominant | argmax max/min | softmax max/min |
|---|---|---|---|---|---|
| `model_p_5skill.pth` | `rallies_5skill.pkl` | 116 | right | 27.8× | 1.0× |
| `model_p_5skill_v2.pth` | `rallies_5skill_v2.pkl` | 76 | right | 19.1× | 1.2× |
| `model_p_5skill_v3.pth` | `rallies_5skill_v2.pkl` | 76 | left | 12.4× | 1.0× |
| `model_p_5skill_v3_1000.pth` | `rallies_v2_1000.pkl` | 76 | right | 15.3× | 1.0× |
| `model_p_5skill_v3_discard_30.pth` | `rallies_v2_30_discard.pkl` | 76 | left | 6.5× | 1.0× |
| `model_p_5skill_v3_tie_30.pth` | `rallies_v2_30_tie.pkl` | 76 | right | 6.0× | 1.0× |
| `model_p_5skill_v3_gantry.pth` | `rallies_v2_1000_gantry.pkl` | 12 | right | 71.5× | 2.1× |
| `model_p_5skill_v3_gantry_sym.pth` ⚠️ STALE | `rallies_v2_1000_gantry_sym.pkl` | 12 | right | **∞** (500,000,000×, i.e. ~100% argmax) | 1.3× |
| `model_p_76dim.pth` — **legacy 2-skill model** | `rallies_v2_2skill.pkl` | 76 | left | **∞** (100% argmax) | 1.1× |
| `model_p.pth` — **legacy 2-skill model** | `rallies.pkl` | 116 | — | SKIPPED: `rallies.pkl` is a flat list of raw state arrays (pre-dict-schema format from the original `train_q_model.py`), not compatible with the dict-based rally format this diagnostic (and every other v2 script) expects. Not a bug — this is a deprecated, incompatible data format. | — |

**Findings (corrected):**
1. **Column bias is real and holds up under name-correct pairing** — every 5-skill model still shows meaningful argmax collapse (6.0× to 71.5×), confirming the meeting's concern independent of the earlier dimension-only pairing mistake.
2. **The `_discard_30` and `_tie_30` models have the lowest bias of any 5-skill model (6.5× and 6.0×)** when evaluated on their own actual training data — noticeably lower than the same models evaluated (incorrectly) against the much larger `rallies_5skill_v2.pkl` in the dimension-matched run (10.2× and 7.5× there). This is consistent with the meeting's hypothesis that different long-rally labeling treatments (discard vs. tie) affect skill preference — though dataset size (750 rallies vs. 1250+) remains a confound, and `tie0` is not clearly better than `discard` here (6.0× vs 6.5×, not a large gap).
3. **`model_p_5skill_v3_gantry_sym.pth`'s bias is far worse (~∞) than initially estimated (190×)** once evaluated against its own correct training data (`rallies_v2_1000_gantry_sym.pkl`) rather than the non-symmetrized gantry file. Combined with the known stale skill-slot-swap bug, this model's results should not be used for any decision — it needs retraining on correctly re-encoded data before its true bias can be assessed.
4. **Both legacy 2-skill models (`model_p.pth`, `model_p_76dim.pth`) are out of scope for this diagnostic** — `model_p_76dim.pth` was trained on `rallies_v2_2skill.pkl` with a 2-value skill encoding `{-1, +1}`, not the 5-skill normalized-index scheme `{0, 0.25, 0.5, 0.75, 1.0}` this script assumes; its "100% argmax" result is likely an extrapolation artifact from evaluating skill values (0.5, 0.75, 1.0) the model never saw during training, not a real finding. `model_p.pth` uses an entirely different, non-dict rally schema and cannot be run through this diagnostic at all.

---

## Suggested Ablation Workflow

All steps are read-only (no MuJoCo, no retraining):

```bash
# Step 1: Check current column bias
PYTHONPATH=. venv/bin/python nash_skills/v2/diag_skill_selection_probs.py \
    --rallies data/rallies_5skill_v2.pkl \
    --n-samples 200

# Step 2: Inspect skill-pair win rates
venv/bin/python nash_skills/v2/plot_winrate_heatmap.py \
    --input data/rallies_5skill_v2.pkl \
    --output logs/winrate_heatmap.png

# Step 3: Compare selection probabilities at different temperatures
PYTHONPATH=. venv/bin/python nash_skills/v2/diag_skill_selection_probs.py \
    --temperature 0.1   # near-argmax
PYTHONPATH=. venv/bin/python nash_skills/v2/diag_skill_selection_probs.py \
    --temperature 2.0   # soft exploration
```

When running a real eval with probabilistic selection (requires MuJoCo):

```bash
# Softmax exploration (temperature 0.5) — compare with argmax baseline
venv/bin/python nash_skills/eval_matchup.py \
    --v3-5skill \
    --selection-mode softmax \
    --temperature 0.5 \
    --seed 0 \
    --episodes 60

# Epsilon-greedy (10% random) — lighter-weight than full softmax
venv/bin/python nash_skills/eval_matchup.py \
    --v3-5skill \
    --selection-mode epsilon_argmax \
    --epsilon 0.1 \
    --seed 0 \
    --episodes 60
```

---

## Expected Effect of Each Mode

| Mode | Expected change vs. argmax |
|---|---|
| `softmax` (T=0.5) | Increases skill diversity; reduces column bias; may lower raw win rate if argmax was genuinely best |
| `softmax` (T=2.0) | Near-uniform; useful as a baseline to confirm skill-conditional Φ values are real |
| `epsilon_argmax` (ε=0.1) | 90% of decisions unchanged; adds light random probing |
| `epsilon_softmax` (ε=0.1) | Mix of soft exploration + random noise; most flexible |

**Warning**: if the model's Φ surface is genuinely informative (right skill really is best), softmax exploration will lower win rate vs. argmax. The goal of probabilistic modes is diagnosis and data collection diversity, not necessarily higher win rate.

### Recommended Starting Point

Given the measured Φ-surface flatness above (softmax already near-uniform at T=0.5, meaning the model has weak skill discrimination), **`epsilon_argmax` with `ε=0.05`** is a conservative starting point for later evals — it preserves the model's existing (weak but nonzero) preference 95% of the time while adding just enough random probing to prevent total collapse and to diversify future data collection. This should be treated as empirical, not fixed: run the ablation matrix below (`ε ∈ {0.02, 0.05, 0.10, 0.20}`) once MuJoCo access is available and pick based on measured win rate + skill usage spread, per the meeting note.

```bash
# ε sweep (requires MuJoCo — run later)
for eps in 0.02 0.05 0.10 0.20; do
  venv/bin/python nash_skills/eval_matchup.py --v3-5skill \
      --selection-mode epsilon_argmax --epsilon $eps --seed 0 --episodes 60
done
```

---

## Files

| File | Purpose |
|---|---|
| `nash_skills/v2/skill_selection.py` | Core module — 3 public functions |
| `nash_skills/v2/diag_skill_selection_probs.py` | Probability diagnostic (no MuJoCo); auto-detects state_dim, fails gracefully on FactoredModel |
| `tests/test_skill_selection.py` | 33 unit tests (all passing) |
| `nash_skills/eval_matchup.py` | CLI flags wired: `--selection-mode`, `--temperature`, `--epsilon`, `--seed`, `--model-dir` |
| `nash_skills/v2/state_encoder_gantry.py` | Skill-slot swap fix in `encode_opp_gantry` (see `lightweight_diagnostics.md` §3.2) |
| `nash_skills/v2/inspect_truncated_rallies.py` | Per-skill-pair truncation stats (no MuJoCo) |
| `nash_skills/v2/labeling_ablation.py` | Prepares relabeled datasets (discard/tie0/asym_small) for future retraining |
| `nash_skills/v2/run_selection_ablation.py` | Sweeps the diagnostic across every compatible (model, rally-data) pair in two directories; auto-pairs by state_dim |
| `tests/test_state_encoder_gantry.py`, `tests/test_symmetrize_rallies.py`, `tests/test_inspect_truncated_rallies.py`, `tests/test_labeling_ablation.py`, `tests/test_run_selection_ablation.py` | New unit tests for the above |
