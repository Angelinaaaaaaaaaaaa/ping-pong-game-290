# Lightweight Diagnostics — Nash Skills Pipeline

> Generated from code inspection only. No MuJoCo runs, no model retraining.

---

## 1. Discount Factor (Gamma)

### Current Setting

| Location | Value | Notes |
|---|---|---|
| `nash_skills/v2/labeling.py:53` | **`GAMMA = 0.7`** | Canonical constant, imported by all v2 trainers |
| `train_q_model_v3.py:128` | `gamma=GAMMA` | **Fixed** — now imports `GAMMA` instead of hardcoding `0.7` |
| `train.py:47` | `gamma=0.99` | PPO low-level controller only (unrelated to meta-policy) |

**Current default is 0.7.** The stale comment claiming *"0.9 per mentor guidance"* has been corrected — `labeling.py` now states that 0.7 is the current setting and that 0.9 (and other values) is a candidate for a **future ablation**, not the current default. Do not change `GAMMA` to 0.9 without running the ablation below and comparing results.

### What Gamma Controls

`compute_returns` in `labeling.py` produces discounted returns for Q-model training:

```
G1[T] = ±1    (winner gets +1, loser gets -1 at terminal crossing)
G1[t] = γ · G1[t+1]   for t < T
```

A lower gamma (0.7) makes the training signal decay faster from the terminal crossing backward — the model is encouraged to attend to the **immediately pre-terminal** crossing rather than the entire rally. This is intentional: early crossings have high noise (either player can still win), so discounting them reduces label noise.

### Where to Plug in a Future Ablation

```python
# nash_skills/v2/labeling.py
GAMMA: float = 0.7    # ← change this for ablation, or add CLI arg
```

All downstream trainers (`train_q_model_5skill_v3.py`, `train_models.py`,
`train_q_model_5skill_factored.py`, `train_q_model_v3.py`) import `GAMMA`
from this file and will automatically pick up any change.

**Suggested ablation values:** 0.5, 0.7 (current default), 0.9, 1.0 (undiscounted). This requires retraining on the server — not yet run.

---

## 2. Max Steps / Truncation

### Where Truncation Is Controlled

| Script | Variable | Default | CLI flag | Context |
|---|---|---|---|---|
| `nash_skills/v2/collect_data.py:68` | `MAX_STEPS_PER_EPISODE = 800` | 800 | `--max-steps` | **Collection-time** — physical env step cap per rally attempt |
| `nash_skills/eval_matchup.py:970` | (CLI default) | **600** | `--steps` | **Eval-time** — step cap per episode |
| `nash_skills/eval_multiseed.py:77` | `DEFAULT_MAX_STEPS = 6000` | 6000 | `--steps` | **Eval-time** — TOTAL steps across all episodes (not per-episode) |
| `diagnostic_fixed_skill.py` | (arg `--steps`) | varies | `--steps` | Per-episode step cap for the fixed-skill diagnostic |

Key observation: **collection uses 800 steps/episode, but evaluation uses 600 steps/episode by default**. Episodes that would terminate naturally between step 601 and 800 during collection will be truncated during evaluation, inflating the truncation rate in eval versus train.

### How Truncated Episodes Are Handled

**Collection (`collect_data.py:207-212`):**
> Truncated episodes are **discarded** — they are not saved to the rally pickle at all. Only episodes with a real winner (`done=True`) are stored. This means the training data has `winner ∈ {1, 2}` only; no `winner=0` labels exist.

**Evaluation (`eval_matchup.py`, `eval_matchup_2skill.py`):**
> Truncated episodes ARE tracked separately (`truncated_episodes` counter) but are **excluded from win-rate denominators**. Both `win_rate` and `win_rate_clean` use `done_episodes` (non-truncated) as the denominator.

```python
# eval_matchup_2skill.py:66 — win_rate denominator
def win_rate(self) -> Optional[float]:
    ...   # ego_wins / done_episodes

def win_rate_clean(self) -> Optional[float]:
    return self.ego_wins / self.done_episodes   # currently identical to win_rate
```

The comment at `eval_matchup.py:189` notes: *"This is currently identical to `win_rate` because truncated episodes are not counted in the denominator."*

---

## 3. Truncation Bias — Why It Matters

### The Bias Mechanism

Suppose ego tends to **win quickly** (1–2 crossings) but **lose slowly** (long rallies that often hit the step cap). Then:

- **Without truncation correction**: truncated episodes (long losses) are excluded → denominator is small → win rate looks high → **optimistic bias**.
- **With truncation correction**: if we included truncated episodes as losses → win rate would be lower → but this is wrong too, because we don't know who would have won.

There is no universally correct treatment. The current approach (exclude truncated from denominator) is unbiased IF the probability of truncation is independent of who would win. In practice, it is not — long rallies tend to be ones where neither player dominates.

### Decided-Only Win Rate vs. Reported Win Rate

| Metric | Denominator | Bias |
|---|---|---|
| `win_rate` | `done_episodes` (non-truncated decided) | Excludes long-running ambiguous episodes |
| `wr_dec` (computed by `analyze_existing_results.py`) | `ego_wins + opp_wins` | Same as `win_rate` here; useful cross-check |
| `win_rate_clean` | `done_episodes` | Currently identical to `win_rate` |

**Selection bias of decided-only rate**: If one skill matchup produces short, decisive rallies (high proportion decided) while another produces long, truncated ones (low proportion decided), the decided-only rate over-represents the decisive matchups. This makes some skill-pair comparisons unfair.

**Recommendation for future work**: track truncation rate per matchup and flag pairs where `tr% > 20%` as unreliable data points.

**Note on `wr_dec` / decided-only win rate**: it is the cleanest available metric (matches `win_rate` exactly when both use `done_episodes`), but it is *selection-biased* per the paragraph above — pairs that rally for a long time are systematically under-represented in the decided-only counts. Treat cross-pair win-rate comparisons cautiously when truncation rates differ a lot between pairs (see Section 3.1 below — they differ by 100 percentage points across pairs in the current dataset).

### 3.1 Truncation Rate by Skill Pair (measured, no MuJoCo)

Ran `nash_skills/v2/inspect_truncated_rallies.py` on `data/rallies_5skill_v2.pkl` (1250 rallies, 74.0% overall truncation rate). Findings:

- **Opposite-lateral matchups truncate almost 100% of the time**: `left` vs `right`, `left` vs `right_short`, `left_short` vs `right` all hit 100% truncation (avg rally length ~12-13 states). This strongly suggests mirror-image skills produce long, physically stable rallies that rarely terminate within the step cap — not a collector bug, but a genuine property of the skill geometry.
- **`right_short`-involved matchups truncate rarely**: `right` vs `right_short` = 0% truncation, `right_short` vs `right_short` = 18% truncation, `right_short` vs `right` = 2%. Rally lengths are much shorter (~4-5 states) for these pairs.
- **Practical implication**: the 325 decided rallies in the dataset are not a random sample of skill pairs — they are concentrated in matchups involving `right_short` and same-side pairs. Any Q-model or potential-function trained on this data has much less signal for opposite-lateral matchups, which may explain part of the argmax collapse toward `left`/`right` seen in Section 5 of `probabilistic_skill_selection.md`.

Run it yourself:
```bash
venv/bin/python nash_skills/v2/inspect_truncated_rallies.py \
    --input data/rallies_5skill_v2.pkl --top-k 10
```

---

## 3.2 Gantry Encoder Skill-Slot Fix (STALE MODEL WARNING)

`nash_skills/v2/state_encoder_gantry.py::encode_opp_gantry` had a bug: when building the opponent's-perspective state, it swapped the ego/opp **gantry** positions correctly but did **not** swap the ego/opp **skill indices** to match. This means every opp-perspective training sample had its skill labels attached to the wrong player.

**Fixed.** `encode_opp_gantry` now places the opponent's own skill in the "ego skill" slot (index 10) and the original ego's skill in the "opp skill" slot (index 11), mirroring the gantry swap.

**Models trained before this fix are STALE and must be retrained:**
- `models_new/model_p_5skill_v3_gantry.pth`
- `models_new/model_p_5skill_v3_gantry_sym.pth`

Any rally pickle produced via `symmetrize_rallies.py` (which calls `encode_opp_gantry`) before this fix should also be regenerated before retraining.

---

## 4. Fixed-Player / Randomized-Opponent Diagnostics

**Existing tool**: `diagnostic_fixed_skill.py` — fixes player 2's skill and sweeps player 1 over all 5 skills. Generates per-matchup stats, step-limit CI, ball trajectory plots, and a markdown report.

**Run command** (requires MuJoCo):
```bash
MUJOCO_GL=egl venv/bin/python diagnostic_fixed_skill.py \
    --fixed-player 2 --fixed-skill center_safe \
    --episodes-per-matchup 30 --steps 600 \
    --seeds 0 1 2 \
    --output-dir skill_eval/fixed_center_safe/
```

**Interpreting results**: If Q1(state, ego_skill, fixed_opp_skill) shows no variation across ego skills (always picks the same skill regardless of opp), this is column-bias — the model has not learned to condition on the opponent's choice. See `nash_skills/v2/diag_q_fixed_opp.py` for the lightweight (no MuJoCo) version.

---

## 5. Suggested Future Ablations (Not Yet Run)

| Ablation | What to Change | Expected Effect |
|---|---|---|
| Gamma 0.5 | `labeling.py:53 GAMMA = 0.5` | More local signal; may underfit long rallies |
| Gamma 0.9 | `labeling.py:53 GAMMA = 0.9` | Longer-horizon signal; risk of label noise |
| Gamma 1.0 | `labeling.py:53 GAMMA = 1.0` | Undiscounted; treats all crossings equally |
| Max steps 400 | `collect_data.py:68` + `eval_matchup.py --steps 400` | Lower truncation rate; faster collection |
| Max steps 1200 | Same | Fewer truncations; longer wall time |
| Epsilon exploration | Add ε-greedy to skill selection in `eval_matchup.py` | Mixes Nash policy with random; diagnoses skill coverage |
| Gantry-only state | Use `state_encoder_gantry.py` (12-dim, fixed skill-slot swap) | Less feature noise; tests spatial sufficiency of state. Requires retraining — existing gantry models are stale (see 3.2) |
| Symmetrized training | Run `symmetrize_rallies.py` (post-fix) then retrain | Forces balanced ego/opp labels; should reduce column-bias |
| Long-rally labeling | `discard` (current) vs `tie0` vs `asym_small` via `labeling_ablation.py` | Compare Q-value variance and downstream win rate; `asym_small` needs an `initiator` field not currently collected |

---

## 6. Quick Analysis Commands (No MuJoCo)

```bash
# Summarise all existing result files
venv/bin/python nash_skills/v2/analyze_existing_results.py

# Summarise a specific file
venv/bin/python nash_skills/v2/analyze_existing_results.py \
    --file skill_eval/matchup_results_5skill.json

# Regenerate win-rate heatmap from rally data (no MuJoCo)
venv/bin/python nash_skills/v2/plot_winrate_heatmap.py \
    --input data/rallies_5skill_v2.pkl \
    --output logs/winrate_heatmap.png

# Lightweight Q-model diagnostic: fix opp skill, check ego argmax pattern
venv/bin/python nash_skills/v2/diag_q_fixed_opp.py \
    --model1 models/model1_5skill_v3.pth \
    --rallies data/rallies_5skill_v2.pkl

# Truncation rate by skill pair (no MuJoCo, no model)
venv/bin/python nash_skills/v2/inspect_truncated_rallies.py \
    --input data/rallies_5skill_v2.pkl --top-k 10

# Prepare relabeled datasets for a future long-rally labeling ablation
# (dry-run first — writes nothing)
venv/bin/python nash_skills/v2/labeling_ablation.py \
    --input data/rallies_5skill_v2.pkl --dry-run
venv/bin/python nash_skills/v2/labeling_ablation.py \
    --input data/rallies_5skill_v2.pkl --modes discard tie0

# Probabilistic skill-selection bias diagnostic (see probabilistic_skill_selection.md)
venv/bin/python nash_skills/v2/diag_skill_selection_probs.py \
    --rallies data/rallies_5skill_v2.pkl \
    --model models_new/model_p_5skill_v3.pth --n-samples 200
```
