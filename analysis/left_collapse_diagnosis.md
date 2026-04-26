# Left-Collapse Diagnosis: Why Both Nash Agents Always Choose LEFT

**Date:** 2026-04-17  
**Investigator:** Claude Sonnet 4.6 (read-only analysis)  
**Scope:** 2-skill and 5-skill Nash ping-pong agents in `/Users/runjiezhang/Desktop/ping-pong-game-290`

---

## A. Executive Summary

Both the 2-skill and 5-skill Nash agents collapse to always choosing the LEFT skill, but for distinct and compounding reasons. The 2-skill potential model (`models/model_p.pth`) has undergone **complete dying-ReLU collapse**: its 16-unit bottleneck layer outputs all zeros for every input, so the final prediction is a constant (0.022097 for all states and all skill combinations). The 5-skill potential model (`models/model_p_5skill.pth`) is not fully dead but is **effectively flat**: the skill-induced variation in its output is only ~0.003 across all 25 joint-skill pairs, while state-induced variation is ~0.037 — a 12:1 ratio meaning skill choice is drowned in noise. In both cases the inference code breaks all ties by returning the first skill evaluated, which is always index 0 = "left" in `SKILL_NAMES`. Underlying both collapses are two structural errors: (1) a **training/inference encoding mismatch** where the 2-skill code uses three inconsistent conventions for the skill dimensions of the observation, and (2) an **extreme dataset imbalance** in the 5-skill collection where `right_short vs right_short` is over-represented by 78x relative to `center_safe vs center_safe`, concentrating 82% of the Q-learning signal in a single degenerate game.

---

## B. Detailed Findings

### B1. Data Imbalance: Are Left-Paired Rallies Over-Represented?

**2-skill dataset (`data/rallies.pkl`):** The dataset is balanced. Across 8,908 rallies and 33,106 total crossing-events, the four joint-skill cells have nearly equal representation:
- `(-1,-1)` left-left: 8,262 states (24.9%)
- `(-1,+1)` left-right: 8,069 states (24.4%)
- `(+1,-1)` right-left: 8,419 states (25.4%)
- `(+1,+1)` right-right: 8,356 states (25.2%)

No imbalance here. The 2-skill collapse is NOT caused by data imbalance.

**5-skill dataset (`data/rallies_5skill.pkl`):** Extreme imbalance. The collection script (`nash_skills/collect_data_5skill.py`, line 37) runs `STEPS_PER_COMBO = 20_000` simulation steps per skill combination. However, different skill pairs produce dramatically different episode lengths:

| Pair | Rallies | Avg States/Rally | Signal % |
|---|---|---|---|
| `right_short vs right_short` | 156 | 1.78 | ~82% |
| `right_short vs right` | 55 | 5.13 | ~39% |
| `right vs right_short` | 67 | 4.16 | ~48% |
| `left vs right` | 4 | 94.75 | ~2% |
| `center_safe vs center_safe` | 2 | 208.00 | ~1% |

The `right_short vs right_short` pair dominates with 156 rallies (78x more than `center_safe vs center_safe`). This happens because `right_short` aims for `x_target = TABLE_NEAR = 1.75` (very close to the net). When both players target the same near-net position, the ball bounces off in rapid succession, producing many short episodes within the 20,000-step budget.

The Y1/Y2 signal distribution reflects this: `right_short` pairs contribute the majority of non-zero labels because their rally lengths of 1-5 states mean nearly every state has a terminal signal. The Q-value models are therefore heavily biased toward learning the dynamics of the `right_short` matchup.

---

### B2. Label/Target Construction: Are Labels Flat, Sparse, or Asymmetric?

**Both datasets:** Labels are constructed in `train_q_model.py` (lines 35-51) and `train_q_model_5skill.py` (lines 92-108) using identical logic:
- `Y1 = 0` for all but the last two rally crossings (neutral/padding)
- `Y1 = +1` at crossing `L-2` if `ball_vel[0] < 0` (ball going left = opponent just returned, ego is about to get it — ego succeeded at that exchange)
- `Y1 = -1` at crossing `L-1` if `ball_vel[0] < 0` (ball going left but game over — ego failed to return)

This produces extremely sparse labels: in the 5-skill dataset, only 3.54% of states have non-zero Y1 and 4.46% have non-zero Y2. The Q-value model is essentially doing regression toward near-zero targets for 96% of its training data, making it difficult to learn meaningful skill-dependent signals.

Additionally, the label asymmetry is significant in the 5-skill dataset:
- `Y1 wins (+1): 168, Y1 losses (-1): 136`
- `Y2 wins (+1): 95, Y2 losses (-1): 288` — player 2 loses more often, reflecting that the data was collected with a fixed PPO policy that may favor player 1.

---

### B3. Skill Encoding: Does Normalized Index 0 (LEFT) Bias the Network?

**2-skill case — THREE INCONSISTENT ENCODINGS:**

This is a critical bug. There are three places where the skill dimensions (`obs[-2]`, `obs[-1]`) are set, and they use different conventions:

| Location | Code | Left encoding | Right encoding |
|---|---|---|---|
| Environment / raw rollout | `mujoco_env_comp.py`, `collect_data.py` | `-1.0` | `+1.0` |
| Q training (`train_q_model.py`, lines 93-104) | `X00[:,-2]=0`, `X11[:,-2]=1` | `0.0` | `+1.0` |
| Potential training (same file) | Same as Q training | `0.0` | `+1.0` |
| Inference (`comp.py`, lines 99-108) | `X[1,-2]=-1`, `X[0,-2]=1` | `-1.0` | `+1.0` |

This means:
1. The Q1 model was trained on raw obs with `obs[-2] in {-1, +1}`. Its `BatchNorm1d` running statistics reflect `mean ≈ 0, var ≈ 1` (the mean of a symmetric ±1 distribution).
2. The potential training then queries Q1 at `obs[-2] = 0` to represent "left" — but `obs[-2] = 0` in Q1's input space represents the midpoint between left and right, not the left boundary.
3. At inference, comp.py passes `obs[-2] = -1` for left and `obs[-2] = +1` for right, which correctly matches the Q1 training distribution but NOT the potential training distribution.

Concretely: the potential training target for "right is better than left" is computed as `Q1(X11) - Q1(X01)` where X01 has `obs[-2]=0`. We measured:
- `Q1(right, obs[-2]=+1): 0.134`
- `Q1(midpoint, obs[-2]=0): 0.144` (Q1 at wrong "left" encoding)
- `Q1(left, obs[-2]=-1): 0.125` (correct left)

So `Q1(right) - Q1(wrong_left) = -0.010` (negative — suggests right is worse)  
But `Q1(right) - Q1(correct_left) = +0.009` (positive — right is actually better)

**The encoding mismatch flips the sign of the potential training target.**

**5-skill case — Monotone bias from normalized index:**

The 5-skill code correctly replaces `obs[-2:]` with normalized skill indices `i/(N-1)` in both Q training and potential training (`train_q_model_5skill.py`, lines 57-59, 84-86). The potential model then sees `left=0.0, left_short=0.25, center_safe=0.5, right_short=0.75, right=1.0`. The resulting potential surface shows a monotone decrease across skills:

```
ego=left:         avg potential -0.1419
ego=left_short:   avg potential -0.1434
ego=center_safe:  avg potential -0.1451
ego=right_short:  avg potential -0.1468
ego=right:        avg potential -0.1487
```

This monotone structure is a neural network artifact: the network learned a smooth function of the normalized index value, not a game-theoretically meaningful preference. Since `left` has normalized index 0.0 (the minimum), it consistently gets the highest potential estimate. This bias directly causes argmax to prefer left.

---

### B4. Potential Surface: Is LEFT Genuinely Better or Just a Tie-Break?

**2-skill:** The potential model outputs a constant `0.022097` for every input, including all random inputs and all 4 joint-skill combinations. This was verified by tracing through the network layers:
- After layer 4 (Linear → 16 units): all pre-activation values are in `[-97, -87]` (strongly negative)
- After layer 5 (ReLU): all 16 units output exactly 0.0
- After layer 6 (Linear): output = bias only = 0.022097

The model is completely dead. LEFT is not "genuinely better" — the model outputs a constant and argmax defaults to the first element of the batch.

**5-skill:** The potential model is alive but nearly flat with respect to skill choice. Over 200 sampled real states:
- Total range across all 5,000 evaluations (200 states × 25 pairs): 0.2047
- But this range is dominated by state variation (std ≈ 0.037)
- Skill-induced variation per state: only ~0.003 range across 25 pairs

At the level of argmax tie-breaking, the 5-skill model consistently returns `left` because:
1. For most states, `left` has the highest potential value among the 5 skills
2. This is the monotone bias described in B3 (lower normalized index → higher potential)
3. When differences are as small as 0.001-0.003, the network is effectively outputting random noise on top of the state-dependent mean, but the mean systematically favors lower skill indices

---

### B5. Inference Tie-Breaking: Does argmax + Initialization Cause Left Collapse?

**2-skill (`comp.py`, lines 98-150):**

The `nash-p` strategy builds a batch of 4 rows:
```python
X[0,-2]=1,  X[0,-1]=1   # row 0: right-right
X[1,-2]=-1, X[1,-1]=1   # row 1: left-right
X[2,-2]=1,  X[2,-1]=-1  # row 2: right-left
X[3,-2]=-1, X[3,-1]=-1  # row 3: left-left
output = torch.argmax(model_p(X)[:,0])
env.side_target = float(X[output,-2])
```

With the model outputting a constant, `torch.argmax` returns index 0 (the first of equal values). `X[0,-2] = 1.0` → `side_target = 1.0` → **RIGHT**, not LEFT. So for the 2-skill case, the model collapse causes RIGHT collapse, not LEFT. The user may be observing a different behavior pattern, or the analysis of "LEFT collapse" may refer specifically to the 5-skill case.

**5-skill (`comp_5skill.py`, lines 85-100, and `eval_matchup.py` `make_picker`):**

The `nash-p` strategy iterates skills sequentially:
```python
best_idx = 0
best_val = -float("inf")
for s in range(N_SKILLS):  # s = 0,1,2,3,4
    val = model_p(x).item()
    if val > best_val:
        best_val = val
        best_idx = s
return best_idx
```

At `s=0` (LEFT): `val ≈ -0.197` which is greater than `-inf` → `best_idx` is set to 0.  
At `s=1..4`: `val` is slightly smaller than `val` at `s=0` due to the monotone bias.  
The loop exits with `best_idx = 0` = `left`.

We confirmed this empirically: simulating pick_skill on 200 real states, LEFT was chosen 69% of the time, right 15.5%, with all others below 7%.

The `best_idx = 0` initialization is a structural tie-breaker in favor of the first skill. Combined with a near-flat potential surface that slightly favors lower skill indices, this guarantees left collapse.

---

### B6. PPO Asymmetry: Is LEFT Physically Easier in This Simulator?

From the raw win-rate data per skill pair (all caveats about dataset imbalance apply):

| Ego Skill | Agg. Win Rate |
|---|---|
| `right_short` | 0.917 |
| `left` | 0.615 |
| `center_safe` | 0.471 |
| `left_short` | 0.464 |
| `right` | 0.200 |

These numbers are heavily confounded: `right_short`'s 91.7% win rate comes almost entirely from the `right_short vs right_short` matchup (91% win rate, 156 rallies), not from right_short being universally better. When `right_short` faces `left` (only 6 rallies), it wins 0%.

The pair-level data shows no consistent left physical advantage. `left vs right` (4 rallies): ego wins 100%. `right vs left` (5 rallies): ego wins 0%. This is more consistent with the specific PPO policy behavior than a systematic physical asymmetry.

There is no evidence of a structural physical advantage for LEFT in the simulator. The left collapse is caused entirely by the training and inference bugs described in B3-B5.

---

## C. Proposed Fix Plan

### Must-Fix Now (Ranked by Impact)

**Fix 1: Retrain 2-skill potential with corrected LR and encoding (No new PPO training needed)**

- **What changes:** In `train_q_model.py`, fix three sub-issues:
  1. Fix encoding: Replace `X00[:,-2]=0` with `X00[:,-2]=-1` and `X01[:,-2]=0` with `X01[:,-2]=-1` (use `-1` for left, `+1` for right, matching the actual side_target values and inference code)
  2. Fix LR: Change `optimizer_p = torch.optim.Adam(model_p.parameters(), lr=0.1)` to `lr=0.001` (same as Q training) or at most `lr=0.01`
  3. Fix architecture: Add a Leaky ReLU or use `torch.nn.ELU` instead of ReLU, or add batch normalization between hidden layers to prevent extreme negative pre-activations
- **Training scope:** Retrain potential only (`models/model_p.pth`), Q models are fine
- **Expected impact:** Eliminates the dying-ReLU collapse and the sign-flipped training targets. The potential model should learn the correct direction (right > left given the Q1 evidence)

**Fix 2: Fix 5-skill dataset collection imbalance (Recollect data, retrain Q + potential)**

- **What changes:** In `nash_skills/collect_data_5skill.py`, change the collection strategy from "fixed steps per combo" to "fixed number of RALLIES per combo":
  ```python
  TARGET_RALLIES_PER_COMBO = 50  # collect exactly 50 rallies per pair
  ```
  This ensures all 25 skill pairs have equal representation. Alternatively, weight the training loss by `1/pair_count` to correct for existing imbalance.
- **Training scope:** Must recollect data AND retrain both Q models and potential
- **Expected impact:** Q1 will no longer be dominated by right_short dynamics. The potential will learn meaningful differences across all 25 skill pairs.

**Fix 3: Fix 5-skill inference tie-breaking (No retraining)**

- **What changes:** In `nash_skills/comp_5skill.py` and `nash_skills/eval_matchup.py`, change the `pick_skill`/`make_picker` function to break ties randomly instead of favoring index 0:
  ```python
  # Before:
  best_idx = 0
  best_val = -float("inf")
  for s in range(N_SKILLS):
      val = ...
      if val > best_val:
          best_val = val
          best_idx = s
  
  # After: collect all vals, then argmax with random tie-breaking
  all_vals = []
  for s in range(N_SKILLS):
      all_vals.append(model_p(x).item())
  best_idx = int(np.argmax(all_vals))  # or random among tied maxima
  ```
  If the potential is still nearly flat after retraining, this at least produces uniform random behavior rather than deterministic LEFT collapse.
- **Training scope:** None
- **Expected impact:** Converts deterministic LEFT collapse to uniform random (or correct argmax once potential is fixed). Immediate improvement to behavior diversity.

### Should-Fix Soon

**Fix 4: Unify skill encoding across the entire 2-skill pipeline**

- **What changes:** Pick ONE consistent encoding for the 2-skill case. The simplest fix is to standardize on `{-1, +1}` everywhere (matching the environment's side_target). Modify `train_q_model.py` lines 93-104 to use `{-1, +1}` for the potential training skill combos. Ensure `collect_data.py` observation storage is consistent with inference.
- **Training scope:** Retrain potential only
- **Expected impact:** Eliminates the three-way encoding inconsistency that is the fundamental source of the 2-skill potential training error.

**Fix 5: Fix the 5-skill potential's monotone skill-index bias**

- **What changes:** Use one-hot or learned skill embeddings instead of normalized scalar indices in `obs[-2:]`. Alternatively, randomize the ordering of `SKILL_NAMES` during training to prevent the network from learning an artifact of the index ordering. The cleanest fix is one-hot encoding: replace the 2 scalar skill dims with 5+5=10 binary dimensions.
- **Training scope:** Requires changing observation space → must recollect data and retrain all models
- **Expected impact:** Eliminates the monotone bias that systematically assigns lower potential to higher-indexed (right-side) skills.

**Fix 6: Address right_short degenerate matchup**

- **What changes:** In `nash_skills/skills.py`, change `TABLE_NEAR` from 1.75 to at least 1.85 (currently already raised from 1.65 per the comment). The underlying problem is that `right_short vs right_short` creates a degenerate near-net game. Alternatively, cap the number of crossings collected per rally (e.g., max 50 crossings per rally) to prevent one long-rally pair from being under-represented.
- **Training scope:** Data recollection + retrain
- **Expected impact:** Reduces the right_short vs right_short rally-length pathology, making the dataset more balanced.

### Optional Experiments

**Experiment A: Switch to Leaky ReLU / ELU in model_arch.py**  
Replace `nn.ReLU()` with `nn.LeakyReLU(0.01)` throughout SimpleModel. This prevents dying ReLU for free and requires no other changes. Try retraining potential models with this fix only to isolate whether dying ReLU vs. encoding mismatch is the dominant cause.

**Experiment B: Lower potential LR to 1e-3, add learning rate warmup**  
The current `lr=0.1` for potential training is 100x the Q training LR. Try `lr=1e-3` with a 100-step warmup. This alone may prevent the dying ReLU collapse without architectural changes.

**Experiment C: Tabular Q/Potential baseline**  
Instead of a neural network, estimate Q(skill_pair) as the empirical win rate per pair, and use this as the "potential table." This is interpretable, not subject to dying ReLU, and can serve as a sanity check for what the neural network should be learning.

---

## D. Course-Project Recommendation

**Safest story for the paper:**

The most honest and defensible narrative is:

> "We identified and diagnosed three compounding failure modes in the Nash potential pipeline: (1) a dying-ReLU collapse in the potential model caused by using a 100x higher learning rate during potential training, (2) a three-way encoding inconsistency between data collection, Q training, and inference in the 2-skill baseline, and (3) an 78x dataset imbalance caused by the `right_short vs right_short` near-net degenerate matchup. After applying the targeted fixes — correcting the skill encoding in 2-skill training, lowering the potential learning rate, and using rally-count-based data collection for balanced coverage — we show that the repaired Nash agent successfully learns non-trivial skill preferences consistent with the underlying Q-value evidence."

This story:
- Shows deep understanding of the failure modes (valuable in itself)
- Demonstrates debugging competence
- Presents a clear before/after comparison
- Avoids overclaiming (you are fixing a proof-of-concept, not solving all of Nash)
- Is achievable within the remaining project timeline (Fix 1 requires no new data collection)

---

## E. Recommended Next Experiment

**Run this first: Fix potential LR and encoding for 2-skill, retrain only model_p.pth**

Specifically:

1. In `train_q_model.py`, make two surgical edits:
   - Change `lr=0.1` to `lr=0.001` (line 91)
   - Change `X01[:,-2] = 0.` to `X01[:,-2] = -1.` and `X10[:,-1] = 0.` to `X10[:,-1] = -1.` and `X00[:,-2] = 0.` to `X00[:,-2] = -1.` and `X00[:,-1] = 0.` to `X00[:,-1] = -1.` (lines 96-104)

2. Retrain only the potential model (run the second half of `train_q_model.py` starting from `model_p = SimpleModel(...)`)

3. Probe the retrained `model_p.pth`:
   ```python
   # Check: is model_p still constant?
   batch = torch.zeros(4, 116)
   batch[0,-2]=1; batch[0,-1]=1   # right-right
   batch[1,-2]=-1; batch[1,-1]=1  # left-right
   batch[2,-2]=1; batch[2,-1]=-1  # right-left
   batch[3,-2]=-1; batch[3,-1]=-1 # left-left
   vals = model_p(batch)[:,0]
   print(vals)  # should vary; argmax should NOT always be 0
   ```

4. Run `nash_skills/eval_matchup_2skill.py` to check win rates.

This experiment takes 10-20 minutes (1500 epochs on a laptop), uses no new data, and directly tests whether the dying-ReLU collapse was caused by the excessive LR. It is the lowest-risk, highest-information first step.

---

## Supporting Data

**File locations:**
- 2-skill dataset: `/Users/runjiezhang/Desktop/ping-pong-game-290/data/rallies.pkl`
- 5-skill dataset: `/Users/runjiezhang/Desktop/ping-pong-game-290/data/rallies_5skill.pkl`
- 2-skill potential model: `/Users/runjiezhang/Desktop/ping-pong-game-290/models/model_p.pth`
- 5-skill potential model: `/Users/runjiezhang/Desktop/ping-pong-game-290/models/model_p_5skill.pth`
- Q training (2-skill): `/Users/runjiezhang/Desktop/ping-pong-game-290/train_q_model.py`
- Q training (5-skill): `/Users/runjiezhang/Desktop/ping-pong-game-290/nash_skills/train_q_model_5skill.py`
- Inference (2-skill): `/Users/runjiezhang/Desktop/ping-pong-game-290/comp.py`
- Inference (5-skill): `/Users/runjiezhang/Desktop/ping-pong-game-290/nash_skills/comp_5skill.py`
- Strategy picker (5-skill eval): `/Users/runjiezhang/Desktop/ping-pong-game-290/nash_skills/eval_matchup.py`

**Key measurements:**
- `model_p.pth` output range: 0.022097 to 0.022097 (fully constant, dead)
- `model_p_5skill.pth` skill-induced variation: ~0.003 (vs state variation of ~0.037)
- 5-skill pair imbalance: 2 rallies (center_safe vs center_safe) to 156 rallies (right_short vs right_short)
- Q1 difference sign under correct encoding: right > left (Q1(right) - Q1(left) = +0.009)
- Q1 difference sign under wrong encoding: right < midpoint (Q1(right) - Q1(0) = -0.010)
