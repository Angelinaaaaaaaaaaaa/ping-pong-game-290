# Experiment plan: Nash-Φ vs self-play

Tracking doc for the full set of experiments toward the L4DC 2027 framing:
same low-level controller and MuJoCo game, compare offline potential-game
learning (Nash-Φ) against online self-play, evaluate both the same way.

Status legend: ✅ done  🚧 in progress / blocked  ⬜ not started

---

## 1. Main pipeline (Nash-Φ) track

### ✅ Retrain the low-level PPO controller
- File: `logs/best_model_tracker1/best_model.zip` (main checkout, server)
- 500M timesteps (the script's real hardcoded target — `TRAINING.md` documents
  100M and a configurable `--total-timesteps`, but `train.py` has no argparse
  at all; every flag in that doc is aspirational).
- Already the default `PPO_MODEL_PATH` everywhere, so it's live.

### ✅ Fresh data collection with the new controller
- File: `data/rallies_5skill_v2_new_ppo.pkl` (1103 rallies, main checkout)
- Collected post winner-assignment fix, with the corrected 1000-step cap.
- Known issue in this dataset: `center_safe` vs `center_safe` never produces
  a decisive winner — 1 accepted rally out of 300 attempts (previously hung
  the collector indefinitely before `--max-attempts-per-pair` was added).

### ✅ Retrain Q/Φ v3 model on the fresh data
- Files: `models/model1_5skill_v3_updated.pth`, `model2_5skill_v3_updated.pth`,
  `models/model_p_5skill_v3_updated.pth`

### ✅ Re-check Φ-vs-empirical-win-rate correlation
- **Pearson r = -0.147** — essentially unchanged from the original ~-0.15
  finding, despite the retrained controller and fresh data.
- Φ's value spread is now much wider (1.91–2.41 vs. the original ~0.17
  range) — it's not flat anymore, it's just not correlated with anything
  strategically real.
- **Conclusion this supports**: the bottleneck isn't staleness (data,
  controller, truncation) — those were all real and worth fixing, but fixing
  them didn't fix Φ. Points at the state encoder itself (skill signal is
  2/76 dims, plus encoded as a single ordinal scalar, not one-hot).

### ✅ Win rate vs random (current `_updated` checkpoint, all 6 decision rules)
- Fresh run via `eval_matchup.py --v3-5skill` pointed at the actual current
  reference checkpoints (`models/model{1,2,p}_5skill_v3_updated.pth`, staged
  under `models_eval_updated/` with the plain `v3` filenames the `--v3-5skill`
  flag expects by default -- its hardcoded default path is the OLDER,
  pre-retrain `model_p_5skill_v3.pth`, not `_updated`; easy to silently
  evaluate the wrong model here). 60 episodes/matchup, `--matchup STRATEGY
  random` for each strategy (not the full 36-matchup default sweep, to stay
  targeted on what was asked for). Local copy:
  `analysis/server_results/vs_random_legacy/`.

  | strategy | win rate vs random |
  |---|---|
  | nash-p-hard | 58% |
  | nash-p-br | 60% |
  | nash-p-minimax | 72% |
  | nash-p-adaptive | 55% |
  | ibr | 68% |
  | ibr-q | 55% |

  All 6 beat random (55-72%). **Important caveat**: a stale local CSV found
  during this check (`ping-pong-game-290-selfplay/matchup_results_5skill.csv`,
  dated Jul 31, predating the Sept retrain) shows very different, much worse
  numbers for the same strategies (25-50%, several AT OR BELOW chance) --
  that file is from the OLD pre-retrain model and should not be cited as
  current. The numbers above are the trustworthy, up-to-date ones.
- **Reconciling this with the weak Φ correlation (r=-0.147 / r=0.040)**:
  not a contradiction. The correlation diagnostic tests fine-grained,
  cell-by-cell structure (24 specific skill-pair cells); beating a uniformly
  random opponent only requires a much coarser signal (roughly which skills
  tend to be stronger overall). Φ can carry that coarse signal while still
  failing to resolve the fine matchup-specific structure the correlation
  diagnostic is sensitive to -- which is also exactly why the argmax-ablation
  finding from earlier in this project (84.0% vs 84.2% average, but real
  left/right asymmetry) matters: aggregate win rate alone, including vs
  random, can hide exploitable weaknesses the correlation/matchup-level
  diagnostics are needed to catch.
- **One-hot model (84-dim) vs random -- run, and it's a concerning result.**
  `scripts/eval_one_hot_vs_random.py`: reuses `eval_matchup.py`'s
  `run_matchup`/`make_picker` as a library (that file itself untouched)
  with an injected one-hot `state_encoder_fn`, deliberately replicating
  a pre-existing quirk in `_v2_state_encoder` (player 2's Q/Φ evaluation
  uses `encode_ego`, not `encode_opp` -- `encode_opp` is present but
  commented out in the source) for comparability with the legacy number
  above, via a `USE_ENCODE_OPP=1` flag to toggle it off for a follow-up.

  | strategy | legacy (76-dim) | one-hot (84-dim), quirk replicated |
  |---|---|---|
  | nash-p-hard | 58% | **42%** |
  | nash-p-br | 60% | **40%** |
  | nash-p-minimax | 72% | **37%** |
  | nash-p-adaptive | 55% | **45%** |
  | ibr | 68% | 67% |
  | ibr-q | 55% | 53% |

  All four Φ-based rules (nash-p-*) drop BELOW chance with one-hot,
  despite one-hot's Φ-correlation being slightly better (r=-0.147 ->
  +0.040). Only the Q-model rules (ibr/ibr-q, which don't use Φ
  directly) hold steady.
- **Root cause found and fixed: this was a bug, not a real result.**
  `eval_matchup.py`'s `_build_phi_table` and the inline `ibr`/`ibr-q`
  counterfactual loops hardcode `x[-2] = ego_s/(N-1)`, `x[-1] =
  opp_s/(N-1)` for EVERY skill-pair query they evaluate -- correct for
  the 76-dim encoded state and the raw 116-dim observation (both end
  with exactly this scalar convention), but wrong for the 84-dim
  one-hot state, where `[-2]`/`[-1]` are just 2 of a player's 5-dim
  one-hot block. Writing a fraction into 2 of 5 one-hot slots while
  leaving the other 3 untouched corrupted every single counterfactual
  query all four `nash-p-*` decisions depend on -- the exact same bug
  category already caught and fixed in `train_q_model_5skill_v3_one_hot.py`'s
  `apply_skill_pair`, just not yet applied to the decision-time code.
  The encode_ego/encode_opp quirk suspected earlier was ruled out
  empirically first (`USE_ENCODE_OPP=1` rerun gave nearly identical,
  still-broken numbers) before finding this bigger, real cause by
  reading `make_picker`'s actual code.
- **Fix**: `_write_skill_pair(state_tensor, ego_skill_idx, opp_skill_idx)`,
  a new pure helper in `eval_matchup.py` that auto-detects which
  convention applies from the tensor's own last dimension -- 84-dim
  gets the one-hot-correct zero-block-then-set-one-index treatment;
  every other dimension (76, 116) falls through to the ORIGINAL scalar
  behavior unchanged, so legacy and raw-obs evaluation are untouched.
  Wired into all 11 call sites across `_build_phi_table`,
  `_q_values_for_player`, `_phi_values_for_player`, the duplicate
  decision-log `ibr-q` helper, and both `make_picker` closures
  (`ibr`/`ibr-q`). 11 new unit tests
  (`tests/test_eval_matchup_skill_pair_write.py`), TDD, green; full
  suite 725 passed (714->725), same 6 pre-existing MuJoCo-only
  failures, including every EXISTING `eval_matchup.py` test
  (`TestMakePickerNoLeftBias`, `TestMinimaxPlayer2SignConvention`, etc.)
  still passing unchanged -- confirms the legacy path is bit-for-bit
  preserved.
- **Also checked and ruled out as affected**: `correlate_phi_winrate.py`
  (the Φ-correlation diagnostic, r=-0.147/r=0.040) reads states directly
  from the rally pickle at ACTUALLY-OBSERVED skill pairs -- it never
  constructs synthetic counterfactual states, so it never hit this bug.
  Those correlation numbers remain valid as reported. V3 (`v3/q_policy.py`)
  has its own separate decision logic and never calls `make_picker` --
  also unaffected.
- **Corrected result** (legacy regression-checked in the same run --
  58-65% range, unchanged from before the fix, confirming no regression):

  | strategy | legacy (76-dim) | one-hot, buggy | one-hot, FIXED |
  |---|---|---|---|
  | nash-p-hard | 65% | 42% | **65%** |
  | nash-p-br | 63% | 40% | **63%** |
  | nash-p-minimax | 58% | 37% | **67%** |
  | nash-p-adaptive | 53% | 45% | **70%** |
  | ibr | 60% | 67% | 67% |
  | ibr-q | 58% | 53% | **77%** |

  With the fix, all six one-hot strategies clearly beat random (63-77%),
  matching or exceeding legacy for every strategy -- `nash-p-adaptive`
  and `ibr-q` in particular now look better than their legacy
  counterparts. One-hot's main-pipeline story is good news after all;
  the earlier "below chance" result was purely a harness bug, now fixed
  and confirmed.

### ⬜ Gantry (12-dim) Φ retrain
- The encoder function exists (`state_encoder_gantry.py`, bug-fixed earlier
  this project), but no training script wires it into `train_q_model_5skill_v3.py`
  yet — that script hardcodes the 76-dim encoder with no flag to switch.
- Deferred once already in favor of the standard v3 retrain above. Given that
  retrain didn't move the correlation number, this is now the better-motivated
  next step, not just an alternative.
- Needs: a small training-script adaptation (TDD-able) + a fresh collection
  pass re-encoded through the gantry function.

### ✅ One-hot skill encoding — built as a new versioned encoder, result: small improvement, not a fix
- Built as `nash_skills/v2/state_encoder_one_hot.py` (`STATE_DIM_ONE_HOT=84`)
  rather than mutating the shared scalar encoder in place -- the legacy
  76-dim encoder and every checkpoint trained on it are untouched.
- Full pipeline run end to end: `collect_data.py --encoder one_hot` (1105
  rallies, server) → `train_q_model_5skill_v3_one_hot.py` (clean
  convergence, Q-loss 0.173→0.021, Φ-loss 10.85→~5.0-5.9, exit 0) →
  `correlate_phi_winrate.py` against the new checkpoint.
- Result: **Pearson r=0.040 (p=0.854), Spearman r=0.090 (p=0.676)** --
  both statistically indistinguishable from zero. Compare to the legacy
  scalar encoder's **r=-0.147**. The sign flipped (negative → small
  positive), which is mildly encouraging and rules out "one-hot made it
  actively worse," but r≈0.04 is not a fix by any reasonable standard.
- Φ spread (not collapsed): std=0.132, range=0.499 (24 cells, -0.095 to
  -0.594 -- notably ALL 24 cells negative, a systematic offset, but with
  real cell-to-cell variance, not a flat constant). Win-rate spread for
  comparison: 6%-92%, std=22% -- the empirical side has plenty of real
  signal to correlate against, so the flat correlation isn't because
  there's nothing there to find.
- **Conclusion: one-hot skill encoding was not the (or not the only) root
  cause of the Φ-vs-win-rate correlation problem.** Switching
  representations alone doesn't fix it. The gantry 12-dim retrain (below)
  or a deeper look at the same-state counterfactual-constraint training
  methodology itself are the remaining live leads, not further encoding
  tweaks on their own.

### ⬜ Per-decision Φ visualization
- For a given state: show all 5 predicted Φ values, which one argmax picks,
  and the actual outcome. Distinguishes "Φ itself is wrong" from "the
  decision rule reads a correct Φ badly" — meeting-note-driven diagnostic,
  not yet built as a standalone tool.

---

## 2. Self-play track

### ✅ First full training run + external eval
- Checkpoint: `models/selfplay_5skill_full1.pth` (feature-selfplay, server)
- Result vs fixed baselines: beats random (74.0%) and all 5 fixed skills
  (58.9–95.0%). Truncation high specifically vs `left`/`left_short`
  (44.5%/23.5%), near-zero elsewhere.
- Skill usage shows real opponent-conditioning: favors `right` against
  left-leaning opponents and shifts toward `left`/`left_short` against
  right-leaning ones — "hit away from the opponent," not a single global
  preference.
- **Caveat**: evaluated using a PPO checkpoint state that didn't exactly
  match what the policy trained against (the controller retrain was
  overwriting the shared checkpoint file concurrently).

### ✅ Second training run with checkpoint history + sweep eval
- Checkpoint + 10 snapshots: `models/selfplay_5skill_full2.pth`,
  `_iter{50,100,...,500}.pth`
- Sweep results: `logs/checkpoint_sweep/iter{50,...,500}.log`
- Win rate vs. random climbed 46.0% (iter 50) → 80.0% (iter 300), plateaued
  72–80% through iter 500 — real improvement over a *fixed* baseline, not
  just the usual self-referential ~50%.
- Weakest/noisiest case: vs. `right` (its own favored skill mirrored back),
  stays in the 49–72% band with no clean trend.
- **Known issue with these numbers**: measured under a stale gantry-x clamp
  (`[-1.0,-0.5]`) that was narrower than what the low-level controller
  actually trained on (`[-1.2,-0.3]`) — a teammate fixed this on
  `feature-selfplay` (commit `3a4b337`, landed after this sweep ran).
  `main` already had the correct range, so the Φ-correlation numbers above
  aren't affected — only self-play's numbers are.

### 🚧 Re-run checkpoint sweep under the corrected gantry range
- Blocked on the jump host being reachable (down 3+ days as of now).
- Same 10 snapshots, same baseline — this replaces the sweep numbers above
  with ones measured in an environment that actually matches what the
  controller trained on.
- Teammate's own measurement (150 rallies/skill, low-level controller only):
  win rate near-identical (49.1% vs 48.2%), but truncation rises
  substantially (8.1%→19.1%) and rallies run longer with fewer unforced
  errors. Expect the self-play trend to hold; don't expect the truncation
  numbers to match the old sweep.

### ✅ Argmax / deterministic eval mode (teammate, local run, not yet pushed)
- Added a deterministic mode (argmax instead of sampling) and ran it on the
  existing checkpoint.
- **Aggregate win rate is a wash**: 84.0% deterministic vs 84.2% sampled,
  within noise. But the average hides a real, consistent split: stronger vs
  `left` (+1.5 to +2.0 pts) and weaker vs the two short skills (−3.5 to
  −5.5 pts). A left-right win-rate gap of +18.5% opens early and stays
  positive the entire run.
- **The framing that matters**: "total score is fine; exploitability is the
  cost, and none of our current metrics measure it." Going deterministic
  doesn't cost win rate on average, but it may create a fixed, learnable
  weak spot (a pure strategy has a well-defined best response; a mixed one
  doesn't) — and nothing we currently log would catch that.
- Confirmed no infinite-loop/hang issue running the checkpoint deterministically.
- **Answers the original question, partially**: the stochasticity isn't
  costing win rate, so it's plausibly doing real hedging work — but we don't
  yet have a metric that would prove it either way. See the new item below.

### ⬜ Exploitability metric
- Directly motivated by the argmax finding above: aggregate win rate can't
  tell "robust mixed strategy" apart from "lucky on average, exploitable in
  specific matchups." Needs a metric that targets the policy specifically —
  e.g. train or hand-craft a best-response opponent *against* the
  deterministic version and see how much worse it does compared to a
  best-response trained against the stochastic version. Not yet built.

### ⬜ Retrain self-play under the corrected gantry range (contingent)
- Only worth doing if the re-run sweep (above) shows the environment fix
  meaningfully changes optimal positioning-dependent strategy, not just
  truncation/rally-length bookkeeping. Decide after that result lands.

### ⬜ Multiple seeds (≥3)
- Currently one (unseeded) run for `full1`, one (unseeded) run for `full2`.
- Needed before citing any self-play number as a stable finding rather than
  one trajectory — report mean ± CI across seeds, not a single run.

---

## 3. Skill target design (teammate, local run, not yet pushed)

New track, not something we'd identified — a teammate found and partly
fixed a *physical* skill-definition problem, separate from anything above.

### ✅ Diagnosed the fault mode, then recalibrated the skill targets
- Found out-of-bounds faults were ~100% lateral (wide), never long — depth
  targeting was fine, left-right aim was systematically off.
- Retuned: lateral target 0.646→0.589, short 1.80→1.75, deep held at 2.50,
  plus a refit aim→landing calibration (`0.893·aimed + 0.271`, r=0.997).
- Result, old vs. new design:

  | | old | new |
  |---|---:|---:|
  | out-rate | 10.3% | 5.4% |
  | win rate | 81.0% | 84.2% |

  All 6 opponents improved (+1.1 to +9.0 pts).
- New eval capability that comes with this: out / wide / long / net-fault
  rates, all reported over one shared denominator (avoids the
  decided-vs-all-rallies denominator inconsistency flagged earlier in this
  project).

### 🚧 Attribute the improvement to a specific constant
- Four constants moved together (lateral, short, deep, calibration fit), so
  which one(s) actually drove the win-rate/out-rate improvement is still
  unknown. Isolation runs (changing one constant at a time) are in progress
  on the teammate's end.

### ⬜ Open question: does the skill design itself need to change?
- Side effect of the recalibration: self-play's usage of `left_short`/
  `right_short` climbed to ~41% combined by iteration 500 under the new
  targets — a real shift in what "optimal" looks like once the skills do
  what they were meant to. Not yet resolved whether this is "a bug fix that
  changes nothing conceptually" or "a redefinition that makes prior
  self-play/Φ results not directly comparable to future ones." Worth
  deciding explicitly rather than let it pass silently, since it affects
  whether `full1`/`full2` checkpoints are still comparable to whatever
  trains under the new targets.
- **Not yet on the server** — this is all local on the teammate's laptop;
  code lands once they have stable wifi.

---

## 4. Comparative / final (both tracks together)

### ⬜ Role-swapped head-to-head: self-play vs Nash-Φ strategies
- `eval_selfplay_5skill.py --vs-strategy` only supports self-play as ego
  (P1) right now — no built-in role swap. Needs a small script extension to
  also run self-play as P2 / Nash-strategy as P1 and aggregate both
  directions.
- Low priority until Φ itself correlates with real outcomes — right now a
  self-play win here would only show "beats a currently-flat Φ baseline,"
  not "beats a strong game-theoretic policy."

### ⬜ Shared benchmark table across both methods
- One table, same fixed opponents (random + 5 skills), same metrics (win
  rate, truncation, avg/median rally length, skill entropy, dominant-skill
  fraction), with confidence intervals, both methods evaluated identically.
- This is the actual paper table — everything else feeds into it.

### ⬜ Behavioral / strategic analysis
- Attack landing-position heatmaps, defender positioning heatmaps,
  state-conditioned (not just opponent-conditioned) skill distribution.
- The self-play checkpoint sweep already shows *opponent*-conditioned
  behavior in aggregate (per-opponent skill-usage tables); this would go
  further to genuine per-state behavior, closer to what a paper reviewer
  would want as evidence of "learned strategy" rather than "learned a
  correlation."
- No instrumentation for this exists yet — real new-code effort, not a
  rerun of something that already exists.

---

## 5. Attacker/defender extension (Design A/B)

Two competing designs, both for adding an explicit defend-position choice
on top of the existing 5 attack skills: **Design A** — role determined by
ball direction (symmetric at game level), formulated as `Q(state,
A_attack, A_defend)` with attacker maximizing / defender minimizing,
structurally analogous to `nash-p-minimax`. **Design B** ("minimal
change") — both players always choose an attack-skill (5-way) and a
defend-position (3-way) every rally; the environment applies whichever
half is relevant based on current ball direction. Agreed approach: build
and validate the *mechanism* (defend-position steering) in isolation
before committing to either Q-formulation, and keep it a **new versioned
pipeline** (separate encoder/checkpoints) rather than mutating the shared
76-dim state encoder — growing `STATE_DIM` would break every existing
trained checkpoint's `load_state_dict` shape.

### ✅ Defend-position steering mechanism, built and live-verified
- `nash_skills/defend_positions.py` — the 3 named gantry targets (left/
  right/center), **not mirrored** for player 2 (confirmed empirically
  correct via live test, unlike `skills.world_target_xy`'s racket-aim
  mirroring).
- `nash_skills/defend_steering.py` — pure steering math: given who's
  defending and their chosen target, compute the gantry action that
  steers toward it using the existing per-step `gantry_step` scale.
- `nash_skills/env_wrapper.py` — opt-in wiring (`set_defend_positions`);
  zero behavior change for any caller that never calls it.
- Live PPO-driven trace (`scripts/trace_defend_engagement.py`) confirms:
  steering converges smoothly to the unmirrored target with no overshoot,
  correctly re-engages across multiple defend phases within one rally.

### ✅ Found and fixed: test harness never drove the PPO controller
- First verification attempt used an all-zeros action for the entire
  step, so neither robot ever swung a racket — "rallies" were single
  uninterrupted ballistic flights, incapable of a role flip. Root-caused
  via a 400-step trace (`ball_vel_x` constant at exactly ±4.5 for an
  entire rally, final `ball_x` far outside table bounds). Fixed in both
  `scripts/test_defend_physics.py` and `scripts/trace_defend_engagement.py`
  by driving the actual PPO controller (`_build_ppo_obs` /
  `ppo.predict`), matching `collect_data.py`'s established pattern. This
  invalidated the original win-rate numbers from that harness (80%/50%/40%
  by defend choice) — not a real result, just distance-to-flightpath with
  no hitting.

### ✅ Found and fixed: naive instantaneous `ball_vel_x` role check
- Original role logic recomputed "who's defending" from `ball_vel_x >= 0`
  every step, with no persistent state. Live trace with real hitting
  caught it misbehaving exactly as anticipated: during one real contact
  event, velocity oscillated -3.81 → +3.81 → -3.82 → +3.48 → -3.82 across
  5 consecutive steps (not just a near-zero wobble — a full-amplitude
  multi-step contact-resolution artifact), which the naive check read as
  5 role flips in 5 steps. A second case showed a small +0.35 velocity
  blip (ball grazing near a boundary) also misread as a flip.
- Fixed with `nash_skills/defend_role.py`'s `DefenderRoleTracker`: latches
  the initial defender only once `|ball_vel_x|` clearly exceeds a
  threshold (0.5; observed contact noise ~0.08, cruising speed ~3.5-4.6),
  then only flips when the ball's x-position actually crosses the net
  plane (`TABLE_SHIFT`=1.5) — immune to velocity-sign noise right at a
  hit. Side-by-side re-trace confirms the fix: tracker holds steady
  through both noisy episodes above while the naive column flip-flops;
  gantry convergence remains clean (final `gantry2_y` within 0.0005 of
  target in all 4 traced rallies). 14 new regression tests
  (`tests/test_defend_role.py`) cover the 4 required cases (ambiguous
  near-zero doesn't latch a wrong defender, clear +/- threshold latches
  correctly, flip only after a real net crossing) plus boundary/reset/
  multi-flip cases.

### 🚧 6-combination mechanistic validation — inconclusive, NOT clean
- Ran the agreed small experiment (attack={left,right} × defend={left,
  center,right}, 10 rallies each, real PPO hitting, fixed role logic) via
  `scripts/test_defend_physics.py` on the server.
- Result: `opp_wr` is 0 or 1 out of 10 in **every** combination —
  player 1 (fixed attacker in this test) wins almost every rally
  regardless of which defend position player 2 pre-steers to. This is
  **not** the "attack-left→defend-left best, attack-right→defend-right
  best" pattern that would count as strong validation per the agreed
  criterion — so per that same agreed criterion, this is not yet clean
  enough to justify building the new encoder/data-collection pipeline.

### ✅ Baseline control (defend=None, no override) — root cause confirmed
- Ran the same attack skills with `set_defend_positions(None, None)` --
  no gantry override at all, player 2 entirely under its own PPO policy.
- Result is a complete reversal from the defend-active rows:
  - Baseline: attack=left -> `ego_wins=0/10, opp_wr=2/10, trunc=8/10`;
    attack=right -> `ego_wins=0/10, opp_wr=0/10, trunc=10/10`.
  - Defend-active (all 6 combos): `ego_wins=9-10/10`, `trunc=0` (one row
    `trunc=1`).
  - Player 1's skill-based attack essentially **never** wins outright
    against a free player 2 (0/10 both attack skills) -- rallies either
    run to the step cap or go to player 2. Turning the override on flips
    this to player 1 winning 90-100% of the time with no truncations.
- This rules out the "pre-existing baseline asymmetry" hypothesis and
  confirms the override-timing one: the defend-steering override isn't
  merely failing to help, it is actively destroying player 2's ability
  to play at all, by holding their gantry locked onto the static defend
  target continuously through the one moment (the return swing itself)
  when the PPO policy needs free gantry control. This is not a coding
  bug -- the code does exactly what the original spec said (override
  gantry for the whole defending phase, release only after the ball
  crosses back over the net) -- it's a finding that the spec's own
  assumption (hold the override for the *entire* defending phase) is
  wrong in practice.
- Open design question, not yet decided: when should the override
  release gantry control back to the PPO policy? Candidates: (a) once
  the defend target is reached (converged) -- matches the real-world
  intuition of "get into position, then swing freely"; (b) once the ball
  crosses some proximity threshold to the defender's side; (c) a hybrid
  of whichever comes first. Needs a decision before re-running the
  6-combination validation a second time.

### ✅ Discovered: Kexin's independent "V3" rotating attacker/defender pipeline
- Commit `3b7bff6` ("Add rotating attacker defender V3 pipeline", 47
  files, ~11.7k lines) landed on `origin/main` mid-session -- a complete,
  independent, much more thorough implementation of the same idea:
  `v3/env_wrapper.py` (defender gantry pre-positioning), `v3/state_encoder.py`
  (84-dim canonical: 76 role-based state + 5-d attack one-hot + 3-d
  defense one-hot), `v3/collect_random_preposition.py`,
  `v3/eval_fixed_preposition.py` (supports all 15 attack x defense
  combinations), `v3/eval_q_policy.py`, `v3/train_q_preposition_onehot.py`,
  plus extensive `diagnose_*.py` scripts and rendered-image verification
  of the P1/P2 coordinate mapping.
- Copied to the server (`v3/`, not merged into the server's messy local
  `master` checkout, to avoid any risk to in-progress binary checkpoints).
- Ran V3's own documented smoke test (`eval_fixed_preposition.py
  --attack-skill left --defense-position right --episodes 10
  --skill-profile aggressive --reset-mode ready`): `attacker_wr=1.000`
  at the role level, BUT `player_return_success_rate=1.0` for both
  physical players (26-27/26-27 successful returns when contact was
  made) and `avg_rally_length=5.6` decisions/rally -- i.e. real,
  sustained back-and-forth, not instant collapse. V3's "attacker always
  wins" is a near-tautological role-level statistic in a ROTATING-role
  framework (whoever fails to return last is, by construction, the
  "defender" at that instant) -- not comparable to this project's own
  FIXED-role metric (same physical player always attacks/defends for the
  whole rally), which is what actually isolates defend-steering's effect.
  Conclusion: V3's defender gantry override (freeze command at exactly
  (0,0) once within tolerance, via a proportional gain controller,
  `position_gain=8.0`) does NOT appear to suffer the same severe
  suppression this project's own override did.
- Mirroring-convention note: V3 labels defend positions from the
  defender's own physical perspective (mirrored for P2: "P2 left" ->
  world y -0.35 -> raw qpos y +0.35), whereas this project's
  `defend_positions.py` uses fixed world-frame labels (same raw qpos
  regardless of player). The underlying RAW QPOS COMMANDS end up
  identical either way (no raw-qpos mirroring in either system) -- the
  difference is naming convention only, not physics.

### 🚧 Override-timing fix re-run under real PPO hitting -- still NOT clean
- Re-ran the exact same 6-combination experiment plus the no-override
  baseline with the target-reached-release fix
  (`nash_skills/defend_engagement.py`) in place.
- Baseline (unseeded, so run-to-run noise expected at n=10): attack=left
  `ego_wins=1/10 opp_wins=1/10 trunc=8/10 rally_len avg=840`; attack=right
  `ego_wins=1/10 opp_wins=0/10 trunc=9/10 rally_len avg=920`. Consistent
  with the earlier baseline run: attacker essentially never wins
  decisively; rallies run nearly to the 1000-step cap.
- Defend-active, all 6 combos: `ego_wins` 9 or 10 out of 10 in every
  single combination, `trunc=0` in every combination, rally_len avg
  150-375 (far short of the step cap). `target_reached` (i.e. the
  override actually released before the rally ended) ranged from 1/10 to
  10/10 depending on combination -- in the two combos where it reached
  100% of the time (`right`/`right`, `right`/`center`), the defender
  STILL lost 9-10/10. The override-release-on-reach fix did NOT restore
  return ability in this project's own harness, despite passing all 18
  of its own unit tests and behaving correctly in the pure-Python logic.
- Ruled out as explanations (checked directly, not assumed): skill
  profile ("current", this project's default, vs V3's "aggressive" --
  aggressive is a MORE extreme/harder-to-return target, so if profile
  mattered it would cut the opposite direction from what's observed);
  reset_mode ("clean" vs V3's "ready" -- confirmed in
  `mujoco_env_comp.py` that `"ready"` is simply aliased to `"clean"`,
  not a different mode at all).
- Leading candidate mechanism, NOT yet confirmed: this project's
  `_steer_toward` is bang-bang (saturates at full speed, i.e.
  `gantry_step` per step, for the entire approach, then an EXACT,
  INSTANT stop the moment `|error| <= DEFEND_REACHED_TOLERANCE`), vs
  V3's proportional-gain controller (`command = clip(error * 8.0, -1,
  1)`), which decelerates smoothly over roughly the last 0.125m before
  settling. An abrupt full-speed-to-zero transition at the exact instant
  of release could leave residual gantry velocity that fights the PPO's
  own control right as the stroke begins, whereas a smooth settle hands
  off cleanly before that moment. Not yet instrumented (would need qvel
  logged at the instant of release across several rallies to confirm or
  refute) -- this is the next diagnostic step per the user's own Phase 3
  gate instruction ("diagnose the remaining physical/control interaction
  first... instrument the timing around target reached, override
  release, ball position, racket contact, and PPO gantry command").
- **Gate status (superseded by the controller A/B below): NOT passed** at
  the time this was written. Per the user's explicit instructions, this
  means STOP before building the attacker/defender versioned encoder or
  the small-scale data-collection mode -- not done yet.

### ✅ Strict controller A/B + release-velocity instrumentation
- Built `nash_skills/defend_controllers.py` (pure logic, 12 unit tests):
  a 2x2 factorial of (approach law: bangbang vs proportional-gain-8.0) x
  (post-tolerance behavior: release-to-PPO vs freeze-at-zero), isolating
  the two differences between this project's design and the teammate's
  V3 design instead of guessing at one fix. `scripts/ab_defend_controllers.py`
  runs all 4 named controllers through the same 6-combo experiment (real
  PPO hitting, `skill_profile="aggressive"`, `DefenderRoleTracker`
  unchanged), logging gantry velocity at the exact step each rally's
  defender first reaches tolerance.
- Aggregate result (10 rallies/combo, 240 rallies total):

  | controller | ego_wr | opp_wr | trunc | reached | avg release |v| | max release |v| |
  |---|---|---|---|---|---|---|
  | bangbang_release (current production) | 0.67 | 0.12 | 0.22 | 0.75 | 0.092 | 0.818 |
  | proportional_release | 0.75 | 0.12 | 0.13 | 0.62 | 0.116 | 1.222 |
  | bangbang_freeze | 0.90 | 0.05 | 0.05 | 0.85 | 0.096 | 0.939 |
  | proportional_freeze (closest replica of V3's design) | 0.92 | 0.03 | 0.05 | 0.63 | 0.136 | 1.327 |

- **Finding 1 (clear, consistent, contradicts the original hypothesis):**
  release vs freeze is the dominant factor, not approach-law smoothness.
  Release beats freeze by ~0.17-0.23 in ego_wr at BOTH fixed approach
  laws (bangbang: 0.67 vs 0.90; proportional: 0.75 vs 0.92). Bangbang
  vs proportional only changes ego_wr by ~0.02-0.08 at fixed post-
  tolerance behavior, with bangbang consistently slightly better. The
  "abrupt full-speed-to-instant-stop leaves residual velocity" theory is
  not supported by the release-velocity numbers either: release_velocity
  tracks approach law (bangbang ~0.09-0.10, proportional ~0.12-0.14,
  nearly identical whether paired with release or freeze), not the
  release/freeze choice itself, and release-velocity differences this
  small don't explain a 0.17-0.23 swing in outcome.
- **Finding 2 (good news):** the project's EXISTING production design
  (`bangbang_release`, i.e. `defend_steering.py` + `defend_engagement.py`
  as already built and wired into `env_wrapper.py`) is the best of the 4
  tested, not the worst -- no controller change is indicated.
- **Finding 3 (why this still isn't a clean pass):** even the best
  variant leaves `ego_wr=0.67` against a no-override baseline of
  `ego_wr≈0.05-0.10` (from the earlier baseline-control run) -- still a
  large gap. Defend-steering, even done as well as these 4 variants
  allow, continues to measurably suppress the defender's return ability
  relative to no override at all. This does not meet "return ability
  restored" per the user's own Phase-3 gate wording.
- **Finding 4 (mixed matchup signal, within bangbang_release specifically):**
  attack=right shows a clean, sensible match effect -- `opp_wr` jumps to
  0.40 (the single highest cell in the whole table) when defend=right
  matches, vs 0.00-0.20 for mismatched/center. attack=left shows no such
  pattern -- defend=left and defend=right both give `opp_wr=0.00`, and
  defend=center (not a "match" under any naming convention) is actually
  best at `opp_wr=0.10`. At n=10/cell this could be noise, or could
  reflect a genuine left/right asymmetry -- the codebase already has
  other flagged left/right asymmetries (e.g. a teammate's
  `diagnose_short_skill_asymmetry.py` in V3) elsewhere, so this would not
  be a new category of problem if confirmed.
- **Gate status: still NOT passed**, for a different and more specific
  reason than before -- not "the mechanism is completely broken" but
  "even the best available controller still leaves a large, unexplained
  gap vs the no-override baseline, and the matchup signal is mixed
  rather than clean." Per the user's own Phase 3 instructions this stays
  in STOP-and-diagnose territory, not proceed. Leading open question,
  not yet investigated: this script only ever overrides player 2's
  gantry (player 1 plays pure PPO in every role, including when P1 is
  defending) -- same asymmetric structure as every fixed-role experiment
  this session, inherited from the original design, not yet examined as
  a possible contributor to the remaining gap.

### ✅ P1/P2 symmetry diagnostic (`scripts/ab_defend_symmetry.py`)
- Three conditions, all under IDENTICAL settings in one script (fixing a
  confound discovered while planning this: `test_defend_physics.py`'s and
  `ab_defend_controllers.py`'s earlier P2-only numbers used different
  `skill_profile` settings and are not directly comparable to each
  other): `none` (`set_defend_positions(None, None)`), `p2_only`
  (`set_defend_positions(None, defend_choice)`), `symmetric`
  (`set_defend_positions(defend_choice, defend_choice)`), all via the
  unmodified production `bangbang_release` path. Tracks both win/loss AND
  `nash_skills.eval_matchup._parse_contact_lines`-based contact/
  return-success counts, per explicit request to avoid a "50/50 but both
  bad" false read.
- **Important boundary, confirmed with the user before running this**:
  this only changes which physical player's gantry gets steering
  assistance at a given instant (ball-direction physics, unavoidable in
  any design) -- it does NOT change Design B's action-space (both
  players always hold both an attack-skill and a defend-position choice
  every decision; environment picks which is physically live). Not a
  step toward Design A's role-gated action space, even informally.
- Result: `none` ego_wr=0.50 (the only apples-to-apples baseline that
  exists so far -- NOT comparable to the earlier "~0.05-0.10" number,
  which used a different skill_profile); `p2_only` ego_wr=0.87,
  avg_len=263.5, opp_contacts=0.85/rally; `symmetric` ego_wr=0.48,
  avg_len=203.8 (shortest of all three), opp_contacts=0.70/rally (lower
  than p2_only, not higher). Symmetric override balances the win rate but
  makes contact rate and rally length WORSE, not better -- the exact
  "bad controller vs bad controller gives 50/50 too" pattern the user
  warned against, not a genuine fix. The P1/P2 test asymmetry is real but
  is not, on this evidence, the dominant explanation for the remaining
  gap from the controller A/B.

### ✅ New one-hot encoder for the MAIN (non-attacker/defender) pipeline
- `nash_skills/v2/state_encoder_one_hot.py`, `STATE_DIM_ONE_HOT = 84`
  (74 shared dims, byte-for-byte identical to the legacy 76-dim encoder
  minus its 2 scalar skill slots, reused by slicing the legacy encoder's
  own output -- not reimplemented -- plus a 5-d one-hot block per player
  in place of each scalar skill slot). Does NOT modify
  `nash_skills/v2/state_encoder.py`; `STATE_DIM` there stays 76 and every
  existing checkpoint keeps loading unchanged. 13 new tests
  (`tests/test_state_encoder_one_hot.py`), TDD, all green. This is
  independent of the attacker/defender gate above -- it only touches
  skill-identity representation, not defend-position at all.
- Explicitly chosen over reviving one-hot in the SHARED encoder (which
  would break every existing checkpoint) per the user's own standing
  instruction not to do that without documented justification; this is
  the "new versioned encoder only" option.
- ✅ Data collected: `data/rallies_5skill_v2_one_hot.pkl` (server), ~1155
  rallies across the 25 skill pairs via `collect_data.py --encoder
  one_hot`, balance ratio 2.94 (comparable scale/shape to the original
  76-dim dataset's 1103 rallies, same historically-hard pairs falling
  short of the 50/pair target). Ready for
  `train_q_model_5skill_v3_one_hot.py` whenever training is requested --
  not yet launched.

### ✅ Confound-free matched-pair mechanism validation — GATE FAILED, clearly
- `scripts/mechanism_validation_matched.py`: attack skill held fixed and
  identical for both players (isolates defend-steering from attack-skill
  asymmetry); `tested_player ∈ {1,2}` (the P1/P2 orientation swap,
  controls for physical-player-specific effects); matched `defend_on`/
  `defend_off` pairs sharing the EXACT same starting conditions.
  Mechanism: `KukaTennisEnv.reset()`'s `seed` parameter is accepted but
  never used internally (confirmed by reading the source before
  building this) -- all real reset-time randomness draws from the
  global `np.random` state inside `reset_ball_throw()`, so true matching
  is via `np.random.seed(x)` called globally right before each `reset()`
  call, not via `reset(seed=x)`, which would have silently done nothing.
  2 attack × 3 defend_target × 2 tested_player = 12 cells, 50 matched
  pairs/cell, 1200 total rally-attempts.
- **Result: every single one of the 12 cells shows a negative delta for
  both tested-player win rate and contact rate.** No exceptions.
  Aggregate: `avg_delta_tested_wr = -0.243`, `avg_delta_contacts =
  -2.065`. Per-cell deltas range from -0.10 to -0.42 (win rate) and
  -1.16 to -3.52 (contacts) -- consistently and substantially negative
  across both attack skills, all three defend targets, and both player
  orientations.
- Rules out the "just hasn't released yet" explanation specifically:
  even in cells where the override released before the rally ended 94%+
  of the time (e.g. attack=right/defend=right/tested_player=1,
  reached=0.94), the harm is still large (delta_wr=-0.22,
  delta_contacts=-1.16). Something about having gone through the
  override leaves the player worse off even after control returns to
  the PPO -- plausible explanation (not yet directly tested): the PPO
  was trained under its own continuous, reactive gantry-tracking policy
  and never on recovering from being artificially placed at one of 3
  fixed discrete targets, so the detour pushes it out of its training
  distribution in a way release alone doesn't undo.
- **Gate status: FAILED, per the user's own explicit pre-committed
  rule** ("如果 defend_on 相比 matched defend_off 仍然大幅降低 return
  success / rally length，那么先别收 225-grid 数据"). Stopping here:
  the full 225-grid collection, Design B value-model training, and
  policy evaluation are NOT started, and should not be until the
  mechanism itself is redesigned -- this is now a well-evidenced
  conclusion, not a cautious guess, given the matched-pair rigor and the
  consistency across all 12 cells. Likely needs rethinking the mechanism
  itself (e.g. a soft bias blended with the PPO's own gantry output
  instead of a hard override-then-release scheme, or learned/continuous
  targets instead of 3 fixed points) rather than further parameter
  tuning (gain, tolerance) on the current override design -- this is a
  design-level finding, not a bug to patch.

### ✅ Soft-bias alpha sweep — GATE FAILED, pivoting to low-level PPO retraining
- `nash_skills/defend_soft_bias.py`: `final_command = clip(ppo_command +
  alpha * Kp * (target - current), -1, 1)` -- ADDS a gentle pull to the
  PPO's own gantry command instead of REPLACING it (the hard-override
  design above). Kp=8.0 reused from the already-validated proportional
  approach law, not a second new free parameter. 9 tests, TDD, green.
- `scripts/soft_bias_sweep.py`: same confound-free, matched-seed design
  as the hard-override validation (fixed/identical attack skill,
  tested_player P1/P2 orientation swap), swept across alpha in {0, 0.1,
  0.25, 0.5} with seeds shared across ALL FOUR alpha values per cell.
  12 cells x 50 seeds x 4 alphas = 2400 rally-attempts.
- **Result: harm is monotonic in alpha, present even at the gentlest
  tested setting.** alpha=0 (baseline): tested_wr=0.335. alpha=0.1:
  tested_wr=0.255 (delta -0.080), already a ~24% relative drop at the
  smallest nonzero alpha. alpha=0.25: -0.165. alpha=0.5: -0.210, with
  avg_contacts also collapsing (-1.07) and rallies shortening
  (624→441 steps) at that point -- the same signature as the hard
  override, just reached more gradually.
- **Important nuance at alpha=0.1**: contacts and win rate diverge --
  avg_contacts actually rises slightly (+0.27) while tested_wr falls
  (-0.080), alongside longer rallies (624→677) and more truncation
  (0.34→0.41). A contacts-only read would have missed the harm; win
  rate caught it. Exactly the "good-looking proxy metric, bad real
  outcome" trap flagged earlier in this track.
- **Not perfectly uniform like the hard-override result (12/12)**: one
  cell (`attack=right, defend=right, tested_player=1` -- a MATCHED
  attack/defend pairing) shows improvement at alpha=0.1 (+0.14).
  Suggestive -- a soft bias might help specifically when well-matched
  to the incoming attack and hurt when mismatched, which would be the
  qualitatively right signature for a working mechanism -- but it's one
  cell at n=50, not yet confirmed, not chased further in this pass.
- **Gate status: FAILED**, applying the user's own pre-committed rule
  for "even alpha=0.1 stably hurts": both the hard-override AND the
  soft-bias designs are now tested and found net-harmful against a
  frozen PPO controller. **Conclusion: this PPO's gantry control is too
  sensitive to any external intervention (hard or soft) to accommodate
  defensive pre-positioning without retraining/conditioning the
  low-level controller itself.** The correct next step per the user's
  own stated architecture is NOT another wrapper-level fix -- it's
  conditioning the low-level PPO's observation on a defend-target/intent
  signal during training, so attack-skill-and-defend-target selection
  and stroke execution become a genuine hierarchical control problem
  (high-level Design B policy chooses intent -> low-level PPO,
  conditioned on that intent, jointly figures out movement + stroke) --
  a substantially larger undertaking (a real PPO retraining run, new
  observation space, new training infrastructure) than anything built
  in this track so far. Not started; needs explicit direction before
  committing to that scope.

### ✅ Design B data-collection pipeline (versioned, small-scale validated)
- `nash_skills/v2/state_encoder_attack_defend.py`:
  `STATE_DIM_ATTACK_DEFEND = 90` (the 84-dim one-hot encoder's output
  unchanged at `[0:84]`, reused by delegation not reimplementation, plus
  a 3-d one-hot defend-position block per player appended at
  `[84:87]`/`[87:90]`). One-hot (not the scalar y-coordinate considered
  in an earlier draft) to stay consistent with the project's current
  direction. Does not touch `state_encoder.py` or
  `state_encoder_one_hot.py`. 13 tests, TDD, all green.
- `nash_skills/v2/collect_data_attack_defend.py`: a new, separate
  collector (not a mode on `collect_data.py`, for the same
  don't-destabilize-the-working-legacy-file reason as every other
  versioned piece this session). Random joint-action mode only, per the
  original phased plan (small subset before the full 15x15=225 grid):
  at each net crossing -- detected via `SkillEnv`'s own
  `DefenderRoleTracker` changing, reused unmodified, not a cruder
  raw-ball-x check -- records the state against the (attack, defend)
  choices active during the segment that just concluded, credited to
  whichever physical player was defending then, and samples a fresh
  joint action for all four values (ego_attack, ego_defend, opp_attack,
  opp_defend) for the next segment. Saved schema: `states` (90-dim),
  the four choice lists, `defending_player`, `winner`, `encoder`
  (`"attack_defend"` provenance tag), plus `steps`/`ego_contacts`/
  `opp_contacts`/`ego_successes`/`opp_successes` (via
  `nash_skills.eval_matchup._parse_contact_lines`, reused not
  reinvented) for the mechanism-health check. 8 tests, TDD -- including
  an explicit no-regression check that the legacy `collect_data.py`
  collector still produces 76-dim `encoder="legacy"` output side by side
  in the same test session.
- Small-scale validation run (`--decisions 150`, server): 71 rallies
  attempted, **71 accepted, 0 discarded** -- random joint-action play
  resolves far more readily than the legacy grid collector's fixed
  skill-pairs (which has several historically-hard pairs). 151 decisions
  recorded, 116 of 225 possible joint combinations hit at least once
  (min=1, max=4, mean=1.30/combination) -- expected, partial coverage at
  this sample size, not meant to be balanced yet. Schema, labels, and
  role-crediting all confirmed correct both by the unit tests and by
  inspecting this live run.
- **Mechanism-health re-check: inconclusive, confounded, not swept under
  the rug.** avg_steps/rally=147.2, avg_ego_contacts/rally=0.42,
  avg_opp_contacts/rally=0.51 -- lower than ALL THREE conditions in the
  symmetry diagnostic above (even `symmetric`'s 203.8 steps / 0.70
  contacts). This is NOT a clean comparison: this collector randomizes
  BOTH attack skill AND defend position for BOTH players at EVERY
  decision, whereas the symmetry diagnostic held attack skill fixed and
  deliberate ("left"/"right") for the whole rally. Random, often-poor
  attack shots plausibly explain most or all of the shorter rallies here
  on their own, independent of defend-steering's health. The honest
  conclusion is that **this run cannot answer the open mechanism-health
  question one way or the other** -- a clean answer needs a collection
  mode that holds attack skill fixed while varying only defend position,
  which is not what "random joint-action, Phase-5-style" was designed to
  test. Flagging this explicitly rather than reporting the raw numbers
  as if they were comparable.
- **What this validates**: the Design B data-collection MACHINERY
  (encoder, collector, schema, role-crediting, no regression to legacy)
  is correct and ready to scale up. **What it does not yet validate**:
  whether the defend-steering mechanism itself is net-helpful, net-
  harmful, or neutral -- that question is still open, same as before
  this pipeline existed. The full 225-grid collection is not enabled
  yet, per the original phased plan.

### ✅ Win rate vs random -- filled in, confirms the hard-override finding generalizes
- `scripts/attack_defend_vs_random.py`: minimal adaptation of
  `mechanism_validation_matched.py` -- same matched seeds, same
  tested_player P1/P2 orientation swap, same hard-override controller,
  but the attack skill is now drawn uniformly at random per rally
  (shared by both players, same symmetric-attack convention) instead of
  fixed to "left"/"right". 6 cells (`defend_target x tested_player`) x
  50 matched pairs x 2 conditions = 600 rally-attempts.
- **Tested player's raw win rate vs random attacker, no defend
  override: 36.7%.** That's the actual vs-random baseline for this
  track -- pure PPO defense alone doesn't beat a random attacker even
  half the time, a separate, more fundamental finding about the PPO's
  defensive strength that has nothing to do with defend-steering.
- **Every one of the 6 cells is negative again** when the override is
  turned on: `avg_delta_tested_wr=-0.160`, `avg_delta_contacts=-2.083`.
  Smaller magnitude than against a fixed, skilled attacker (-0.243) but
  the same direction, no exceptions. **This closes the possibility that
  the hard-override's harm was somehow specific to facing a deliberate,
  skilled attacker -- it generalizes to a random, unpredictable one
  too.** The mechanism-level conclusion (needs low-level PPO
  conditioning, not another wrapper fix) stands on firmer ground now
  than it did with only the fixed-attack result.

### ✅ Target-pose redirect ("Option B") -- root-caused in full, GATE FAILED, pivoted to a reactive mechanism that passes
- Direction confirmed explicitly, repeatedly, by the user: Design B only
  (both players always hold a joint attack+defend choice every rally;
  environment picks which half is live from ball direction) -- not
  Design A (role alternation). The pasted L4DC paper draft's own formal
  model (Section 3.2, Eq. 6/7, the "active distribution determined by
  current role" policy-parameterization sentence) is written as Design
  A -- a confirmed, still-unresolved mismatch between the paper text and
  the implementation direction. Proposed fix (not yet applied to any
  file, paper exists only as pasted chat text, no local source):
  generalize Eq. 7's role-gated $\mathcal{K}$-or-$\mathcal{M}$ action set
  to $\mathcal{K}\times\mathcal{M}$ for every player every exchange;
  Sections 4-5's actual machinery ($\theta^i=(w^{attack,i},w^{defend,i})$,
  the 15-cell static-baseline set, the two-headed self-play actor) is
  already representation-agnostic and needs no change.
  **SUPERSEDED (2026-10-18)**: this proposal read our implementation as the
  reference. Checked against the paper's role definitions, the paper's
  "defender" is the player who just HIT the ball (it prepares during the
  opponent's return), while our `defending_player` is the RECEIVER. Keep
  Eq. 6/7 as written and fix the code instead. See "Paper review" below.
- `nash_skills/defend_target_pose.py` (static racket-pose redirect, built
  earlier) got two more real, empirically-grounded fixes this round:
  (1) `DEFEND_POSE_Y` -- the lateral offset was borrowed from
  `defend_positions.py`'s gantry-qpos dict (±0.35), never checked against
  the racket-pose space it was actually used in; measured via
  `scripts/trace_defend_y_sign.py` (40 real rallies): sign was correct,
  magnitude needed is ~0.63/-0.75, roughly double. (2) The engagement
  latch (`DefendPoseEngagementTracker.should_steer`) compared GANTRY QPOS
  against an unrelated gantry-qpos target while the mechanism was
  actually commanding a RACKET POSE -- fixed to compare the real racket
  end-effector position against the actual commanded pose. Verified
  effect: `reached` went from ~0% to ~90%+ of cells.
- **Despite both real fixes landing, the gap barely moved**
  (avg_delta_tested_wr: -0.308 -> -0.292 -> -0.271, vs hard-override's
  -0.243) -- traced directly (`scripts/trace_handoff_moment.py`) to the
  root cause: this static mechanism replaces the racket-pose TARGET
  itself (not just the action), which is the ONLY channel this low-level
  PPO ever receives ball information through (confirmed by reading
  `update_target_racket_pose_opp`'s actual math this session: its
  x_target/y_target parameters select racket ORIENTATION only -- the
  interception POSITION is 100% ball-trajectory-determined, with no
  bias hook at all). Freezing the target to a static point blinds the
  policy to the ball for the whole pre-position phase; at handoff the
  real target can jump 0.5+ units (ten times the 0.05 convergence
  tolerance) with too little remaining flight-time to recover -- true
  even once the latch bug above was fixed and handoff started firing
  correctly. **Gate status: FAILED**, a third distinct no-retrain
  mechanism now confirmed broken, this time with a precise mechanistic
  account rather than a null result.
- **Fix: `nash_skills/defend_reactive_pose.py`** -- ports the SAME
  ball-trajectory interception math `update_target_racket_pose_opp`
  already uses (verified bit-for-bit against the real engine's own
  output across 472 compared steps, `scripts/verify_reactive_pose_port.py`)
  and adds a lateral bias toward the named position that fades to zero
  as the ball approaches (`BIAS_MAGNITUDE=0.15`, fade window
  `ball_pos[0]` in `[0.3, 1.5]` -- first-guess constants, not yet tuned).
  No freeze, no static point, no handoff, no discontinuity by
  construction: the target is never ball-independent in the first
  place. `SkillEnv.set_defend_reactive_positions` (new, alongside the
  now-superseded `set_defend_target_positions`).
- **SUPERSEDED first validation (kept for the record, numbers NOT
  valid):** an initial full-scale run reported avg_delta_tested_wr=-0.068,
  avg_delta_contacts=+9.483 (contacts 3-7x higher, rallies running to
  1700-2500 steps). These were produced under a second bug, found later
  (below) -- the inflated contacts were an artifact, not a real effect.
- **Second real bug, found when Phase 2/4 stalled:** with the reactive
  mechanism active for BOTH players (Design B's actual usage --
  the matched-pair validation above only ever activated it for one
  side), collection acceptance collapsed to ~0.5% and self-play hit
  trunc=1.00 every iteration (no learning signal at all). Systematic
  isolation, in order: per-decision vs per-rally resampling (fixed,
  no effect), skill-pair diversity (`scripts/sweep_both_sided_skill_pairs.py`:
  24/25 pairs at trunc=1.00 -- not skill-specific), bias magnitude
  (`scripts/sweep_bias_magnitude_both_sided.py`: still 1.00 even at
  BIAS_MAGNITUDE=0.0 -- falsified). Real cause: `reactive_defend_pose()`
  solved orientation toward a hardcoded NEUTRAL aim instead of the
  defending player's own attack skill. The role tracker still labels a
  player "defending" at the instant of contact (it flips only after the
  hit), so this neutral aim WAS every player's outgoing shot -- every
  exchange converged on the same safe, maximally-returnable placement
  regardless of anyone's choice. Fix: x_target/y_target are now REQUIRED
  parameters, supplied by `SkillEnv._reactive_pose_for_player` from that
  player's own current skill (`test_orientation_responds_to_the_given_aim_not_a_fixed_default`
  is the regression guard).
- **Valid results after the orientation fix:**
  - Both-sided 25-pair sweep: overall truncation 0.997 -> **0.245**; 16/25
    pairs at trunc=0.00. Only `center_safe` vs `center_safe` stays at
    1.00 -- the same pre-existing degenerate matchup already documented
    for the main pipeline (section 1: "1 accepted rally out of 300
    attempts"), unrelated to this mechanism.
  - **One-sided matched-pair, full scale (1200 rally-attempts, 2500-step
    cap): avg_delta_tested_wr=-0.018, avg_delta_contacts=-0.225,
    trunc=0.00 in all 12 cells.** Per-cell deltas -0.14..+0.12 with no
    systematic direction -- noise around zero. This is the honest
    headline: the mechanism is essentially NEUTRAL (does not hurt the
    defender vs no preference at all), the first of four no-retrain
    designs to cleanly pass the original gate. It does NOT show that
    choosing the RIGHT defend position beats choosing the WRONG one --
    that is a separate, not-yet-run experiment (see the remaining-work
    table at the end of this file).
- **Gate status: PASSED (neutral, not beneficial).** Pipeline re-wired:
  `nash_skills/v2/collect_data_attack_defend.py` (also changed to sample
  once per rally, `TestFixedPerRallySampling` guards it) and
  `nash_skills/eval_matchup_attack_defend.py` both use
  `set_defend_reactive_positions`; FakeEnv regression guards reject
  either older mechanism. Phase 2 re-verified: 25/33 rallies accepted
  (76%), 150 decisions in 71 s, avg 6 decisions / 452 steps per rally.

### ✅ Mixed-strategy (θ) data collection (paper Sec. 4.1, small-scale validated)
- Paper Sec. 4.1 asks for continuations from "deterministic choices as well
  as mixed strategies". New `--sampling mixed --p-vertex 0.3` mode in
  `collect_data_attack_defend.py`: per rally, each player draws
  θ^i=(w^A,w^D)∈Δ5×Δ3 (a vertex w.p. `p_vertex`, else Dirichlet(1)); the
  four action slots of the 90-dim state hold θ instead of one-hots
  (`write_mixed_strategy_pair`), and the joint action is redrawn from θ at
  every decision. Rallies carry `ego_theta`, `opp_theta`, `theta_mode`.
  Helpers in `nash_skills/v2/mixed_strategy.py` (simplex projection,
  sampling, PGA). `random`/`grid` stay fixed-per-rally (static profiles).
- Local smoke run: **65/66 rallies accepted (98.5%), 302 decisions, 4.6
  decisions/rally, median 203 steps.** P1 won 37/65 (16/25 in the pure
  set); not significant at this n, recheck side bias at scale.

### ✅ Design B α-potential trainer, paper-aligned (`train_q_model_5skill_v3_attack_defend.py`)
- 90-dim input; Q1/Q2 value nets and Φ, unilateral deviations over the full
  15-way joint (attack, defend) choice per player (6300 constraint triples).
- Paper Sec. 4.1/5.2 items now implemented as flags:
  - rally-level train/held-out split (`--holdout-frac 0.2`, `--seed`), so
    correlated states of one rally never straddle the split;
  - held-out unilateral residuals |e| = |ΔΦ − ΔQ| written to
    `logs/heldout_attack_defend.json`; **held-out max|e| is the empirical
    α in Prop. 1's (α+λ) bound**, reported with p95/mean;
  - max-discrepancy objective `--phi-loss lse` (smooth max,
    (1/β)[logsumexp(β|e|) − log N], β=10) beside the default MSE;
  - nested data-volume subsets `--train-rally-limit M`, `--gamma`;
  - `--theta-mode mixed`: deviations drawn in θ-space (needs mixed data;
    fails fast on pure-only data).
- Toy-scale results (local, NOT final; datasets of 25 and 65 rallies):

  | data | setting | held-out max\|e\| | p95 | mean |
  |---|---|---|---|---|
  | pure, 25 rallies | MSE | 1.57 | - | 0.21 |
  | pure, 25 rallies | LSE | 1.19 | - | 0.25 |
  | mixed, 65 rallies | θ-mode pure | 1.512 | 0.751 | 0.297 |
  | mixed, 65 rallies | θ-mode mixed | 1.422 | 0.500 | 0.178 |

  LSE trades mean for max as designed; θ-mixed training fits mixed data
  better. Held-out Q MSE 0.58 / 0.50. At this size α≈1.2-1.6 is large next
  to Φ's range, so no Table 1 conclusion can rest on these models.

### ✅ Design B self-play, moved to the paper's PPO recipe (`nash_skills/v2/selfplay_attack_defend.py`)
- Two-headed actor (attack Δ5, defend Δ3) + value head, shared trunk,
  90-dim state; both heads sampled at every net crossing.
- REINFORCE → actor-critic → **PPO** (clipped surrogate ε=0.2, GAE λ=0.95,
  γ=0.99, 4 epochs, minibatch 64, value coef 0.5, entropy coef 0.05),
  frozen-snapshot opponent pool (`--snapshot-prob 0.5`), as in Sec. 4.2.
- **Critic bootstrap at truncation**: an unfinished rollout is valued by the
  critic, not scored as a lost point (paper: "bootstrapped from its critic
  rather than treated as a lost point"); verified: truncated reward 0.0,
  bootstrap value ≈ −0.20 on a forced truncation.
- **Policy entropy by role** logged (`entropy_attack`, `entropy_defend`), as
  Sec. 5.3 asks.
- Smoke run (2 iters × 4 rallies): trunc 0.00, ent_a 1.605 (max ln5=1.609),
  ent_d 1.095 (max ln3=1.099), 3-4.5 crossings and 210-310 steps per rally.
  22 unit tests.

### ✅ Design B decision rules incl. continuous-θ PGA (`nash_skills/eval_matchup_attack_defend.py`)
- `nash-p-hard/br/minimax/adaptive` on the 15×15 Φ table, `ibr`/`ibr-q`
  (Q-based), `random`, and any fixed (attack, defend) pair (the 15 static
  opponents). Runs through `set_defend_reactive_positions`.
- New `nash-p-pga`: projected gradient ascent on Φ over Δ5×Δ3 per player
  (paper Eq. 18), warm start from the previous decision, 3 restarts, then
  samples the action from θ.
- Smoke test on 8 real states with the mixed-trained toy Φ:
  - interior θ in 4/8 states (attack mixes two skills; defend always at a
    vertex); PGA's modal action = exhaustive 225-profile argmax in 7/8;
  - **PGA value − best pure profile: max +0.0098, mean −0.0009**, below
    the best pure profile in 3/8 states — about 150× smaller than the
    held-out α (1.42), i.e. not distinguishable from model error;
  - restart spread mean 0.056; **61 ms per decision** vs one batched
    forward pass for the discrete table.
  - Reading: at this scale PGA adds nothing measurable over discrete
    argmax and costs more. Keep it for paper fidelity, report `nash-p-hard`
    as the primary rule, re-check at full scale.

### 🚧 Paper review (Sections 3-5 vs code) — one serious mismatch found
- **Role semantics are inverted relative to the paper (found 2026-10-18,
  not yet fixed).** Paper Sec. 3.2: the *attacker* ι_t "receives the
  currently incoming ball" and picks the return target; the *defender* ȷ_t,
  "having produced the preceding shot", picks a preparation position and
  "moves toward [it] during the opponent's return phase", **without**
  seeing the attacker's new target. Our code: `DefenderRoleTracker.
  defending_player` = the player whose half the ball is in = the
  **receiver** (`defend_role.py`), and every defend mechanism we built
  (hard override, target pose, reactive bias) acts only on that player
  (`env_wrapper.py` `use_defend_reactive1 = defending_player == 1`).
  Consequences:
  1. The defend choice of the player who just hit is never applied.
     During the opponent's return the interception guard returns None
     (ball moving away), so that racket target simply **freezes at its
     last contact pose**; the paper's preparation window is unmodeled.
  2. The receiver's defend choice is drawn at the net crossing, when the
     incoming ball is already in the state, so there is no "preparation
     under uncertainty". Both live choices in an exchange belong to the
     receiver; the attacker-vs-defender tension within an exchange is gone.
  3. Likely explains the neutral gate result (−0.018): a bias of 0.15 that
     fades to zero before contact can barely matter by design.
- **Proposed paper-faithful mechanism (still Design B):** both players
  keep holding (k, m) every exchange; the environment makes the
  **receiver's attack** choice live (as now) and the **non-receiver's
  defend** choice live — set its racket target to the preparation pose
  d_m while the ball moves away from it (exactly the window where the
  guard already returns None and nothing tracks the ball), and release
  to normal interception the moment the opponent's return is incoming.
  This avoids Option B's blinding/handoff failure because nothing is being
  tracked in that window. Decisions at net crossings then match the
  paper's information structure: the attacker has seen the incoming ball,
  the defender has not seen the attacker's new target.
- Other gaps (paper text or not-yet-run experiments):
  - Sec. 4.2 "Only the head corresponding to the active role selects an
    action": under the fix above, credit only the live head's log-prob in
    PPO (summing both heads is unbiased but noisier).
  - Sec. 4.1/5.3 horizon H (continuation cutoff) is not a flag yet; γ is.
  - Sec. 5.1 simulation-based IBR (rollouts over H_IBR) is not built; the
    `ibr`/`ibr-q` rules are Q-based proxies — label them as such.
  - Sec. 5.1 unilateral improvement R_i (best response trained against
    each frozen method) — not built.
  - Sec. 5.3 stochasticity 2×2 (nominal vs perturbed transitions ×
    sampled vs argmax) — not built; needs a perturbation model.
  - Sec. 5 setup "balancing table sides and initial roles" —
    `run_matchup_joint` always plays strategy 1 as P1; the Table 1 driver
    must run both orientations.
  - Sec. 5.4 human-video comparison (Table 2) — out of scope for this code.
- Already consistent: zero-sum ±1 point reward, Φ unilateral-deviation
  definition, 15 static pairings, random baseline, PPO + critic + snapshot
  pool + entropy, critic bootstrap, mixed θ + PGA (Eq. 18).
- **Update 2026-10-18:** the proposed mechanism is now built and compared
  head-to-head with the reactive one (next entry).

### ✅ Preparation mechanism (paper roles) built and compared with the reactive one
- **What it is.** Still Design B (both players hold (k, m) every exchange),
  with the roles alternating at every hit: the receiver's attack choice is
  live, and the player who just hit uses its defend choice. Opt-in
  `SkillEnv.set_defend_preparation_positions`
  (`nash_skills/defend_preparation.py`): while the ball moves away from a
  player (±0.5 velocity dead-band), its racket target is its calibrated
  defend pose (`defend_target_pose`); once the opponent's return is
  incoming, the normal interception update runs unchanged.
  `--defend-mechanism reactive|preparation` in the collector,
  `run_matchup_joint`, and self-play (default `reactive`, recorded per
  rally); switch in `nash_skills/defend_mechanisms.py`. 35 new tests.
- **Baseline trace before building it** (20 rallies, 141 hits): the
  direction latch flips within one step of every recorded hit, with no
  stray flips. The window lasts a median 62 steps. Without a mechanism the
  hitter's racket moves a median 1.1 cm in it, because its target is frozen
  at the last contact pose. All 16 misses had a closest racket–ball distance
  of 0.25–1.72.
- **Live check** (9 rallies): the target is the preparation pose for 100%
  of every window. The racket gets there (median distance 0.11–1.37 → 0.03–0.08
  by the opponent's hit, even across the full table width) and is handed back
  to interception after every window. Smoke runs: collector 40/50 rallies kept
  (80%); self-play 2×4, truncation 0.
- **Methodology bug found along the way — earlier "matched pairs" were not
  matched.** `KukaTennisEnv.reset()` never resets `curr_target` /
  `curr_target_opp` (its `reset_target()` call is commented out), so every
  rally starts by chasing the previous rally's leftover target. Same-seed
  replays of the *same* condition diverged on 3/3 seeds, once from step 0.
  `scripts/compare_defend_mechanisms.py` restores the construction-time home
  target before each reset; with that, replays are identical and reactive
  "center" reproduces no-mechanism exactly on 100% of seeds.
  - Consequence: the earlier gates (hard override −0.243, target pose
    −0.27..−0.31, reactive −0.018) were effectively unpaired, so they were
    noisier than reported. The large, all-cells-negative results keep their
    direction. The reactive result is re-measured below with exact pairing.
  - The production pipelines still carry the target over between rallies.
- **One-sided comparison** (`--mode one-sided`, 800 seeds × 7 rallies,
  symmetric attack skills left / left_short / right_short / right, tested
  player P1 or P2, 2500-step cap, 0 truncations, 95% paired-bootstrap CIs):

  | | reactive (current) | preparation (paper) |
  |---|---|---|
  | avg over positions − no mechanism (pre-set gate, floor −0.05) | +0.002 [−0.018, +0.025] PASS | −0.057 [−0.084, −0.028] **FAIL** |
  | matched − mismatched position | −0.013 [−0.052, +0.026] — no effect | **+0.130 [+0.091, +0.166]** |
  | matched − no mechanism | −0.003 | −0.035 [−0.069, −0.003] |
  | center − no mechanism | 0.000 (identical rallies) | +0.030 [−0.003, +0.061] |
  | mismatched − no mechanism | +0.010 | **−0.165 [−0.205, −0.125]** |

  - Under reactive, the defend choice is inert.
  - Under preparation, it is strongly consequential:
    - preparing on the wrong side costs 16.5 points;
    - center is best or near-best in almost every cell, the classic "recover
      to the middle";
    - the attacker can punish a side commitment (e.g. vs left_short, preparing
      right wins only 0.08 / 0.15 for P1 / P2).
  - P1 and P2 show the same pattern cell by cell, so there is no frame error.
  - Changing only the position flips the winner in ~46% of seeds under both
    mechanisms. That alone measures chaotic sensitivity, not strategic
    content, so the contrast is the meaningful number.
- **Both-sided comparison** (`--mode both-sided`, Design B's real usage: 400
  seeds, random static profiles over all 5 skills, both players use the
  mechanism):

  | | truncated | median steps | returns/rally |
  |---|---|---|---|
  | no mechanism | 0.215 | 325 | 12.9 |
  | reactive | 0.245 (paired McNemar p=0.09) | 323 | 13.7 |
  | preparation | **0.145 (p=0.0003)** | 241 | 9.4 |

- **Reading.** For the research question (a defend axis that carries
  strategy, as Table 1's 5×3 static matrix and the paper assume),
  preparation is the better mechanism:
  - it matches the paper's roles;
  - it gives the choice a large, systematic effect;
  - it produces significantly more decided rallies.

  It fails the gate as pre-registered. That gate averages a good choice
  with a bad one, so a consequential choice fails it by construction. The
  check the gate was meant to stand in for — does the low-level controller
  still work when the choice is good — passes: center +0.030, zero
  truncations. This re-reading of a pre-set criterion is flagged
  explicitly rather than silently adopted.
- **Open points:**
  - The matched side is not better than center (0.470 vs 0.535). The side
    poses (±0.63 / −0.75) are contact poses, and may be too extreme as
    waiting poses; a side-offset sweep could restore a center-safe vs
    side-gamble trade-off.
  - Center dominance may make optimal defend play nearly pure.
  - **Answered on the server (next entry):** under the retrained controller
    the matched side beats center.

### ✅ Server runs 2026-10-20: Design B + preparation, the paper's pipeline
Decision (user, 2026-10-20): Design B with the preparation mechanism is the
method; self-play second. Full details, commands and checksums:
`analysis/server_results/designb_preparation/README.md`.

- **Setup.**
  - Fresh server folder `~/angelina/ping-pong-game-290-designb` and a Python
    env pinned to the local versions.
  - The **retrained** low-level controller (the server's live, uncommitted
    checkpoint `db5df579…`); git and local machines carry the April one.
  - Per-rally target reset; 2500-step cap.
- **Two harness fixes landed first** (tests included):
  - `run_matchup_joint` used to hand player 2's picker player 1's *newly
    chosen* action, contradicting the paper's simultaneous-move timing.
    Both now see only the other's previous choice.
  - New `selfplay:` / `selfplay-argmax:` strategies, plus
    `scripts/eval_table1_attack_defend.py`, which plays both side
    orientations and reports truncations separately.
- **Mechanism comparison, retrained controller:**
  - Preparation's matched − mismatched is **+0.219 [+0.179, +0.257]**
    (April: +0.129). Matched − none +0.039, mismatched − none −0.177.
  - Reactive's defend choice is not a coherent strategic variable: −0.042,
    the wrong direction.
  - Both players active: truncation none 0.295 / reactive 0.290 /
    preparation **0.138**.
  - Server and local agree to 0.001 on the April controller.
  - Side-offset sweep: 0.25× → +0.035, 0.5× → +0.016, 0.75× → +0.095
    (also passes the pre-set rule), 1.0× → +0.219 (fails it). 1.0×
    (pre-specified) is kept; 0.75× is the alternative that satisfies both.
- **Data:** pure 1256 rallies (85.7% kept) and mixed θ 1684 (98.4%), 18,209
  decisions in total.
- **Φ fit** (held-out, rally-level split):
  - Main run (smooth-max loss, mixed θ, 3 seeds): max|e| = α ≈ 1.93–2.22,
    p95 0.87–0.91, mean 0.29–0.30.
  - The data-volume curve falls with data (α 2.60 → 2.08 → ~2.0; Q MSE
    0.40 → 0.33 → 0.25–0.30).
  - Pure-only data is worse (mean 0.41).
  - α ≈ 2 is large next to ±1 utilities, so Prop. 1's bound is loose.
- **Table 1, α-potential row** (Φ seed 0, 200 decided rallies per opponent,
  both sides):

  | rule | static mean | static min | random | IBR (Q-based) |
  |---|---|---|---|---|
  | `nash-p-pga` (paper) | **0.698** [0.682, 0.714] | 0.555 | **0.640** [0.571, 0.703] | 0.535 [0.466, 0.603] |
  | `nash-p-hard` | 0.661 [0.645, 0.678] | 0.495 | 0.520 [0.451, 0.588] | 0.425 [0.359, 0.494] |

  - **This overturns the toy-scale reading above** ("PGA adds nothing
    measurable"). That check compared Φ values only; in play, sampling the
    PGA mixed strategy wins more and is harder for IBR to exploit.
  - The hardest static opponents prepare at center.
  - **Over Φ seeds 0/1/2 (mean, range):**
    - PGA: static mean 0.666 (0.647–0.698), static min 0.470, random 0.620,
      IBR 0.535.
    - Argmax: 0.635 / 0.402 / 0.580 / 0.463.
    - PGA is better on the static mean and vs IBR in every seed.
- **Self-play, final** (3 seeds, 500 iterations; mean, range):
  - Sampled: static mean 0.621 (0.601–0.650), min 0.448, random 0.665,
    IBR 0.537, vs α-potential PGA 0.517 (0.455–0.600).
  - Argmax: static mean 0.670 (0.614–0.715), min 0.467, random 0.698,
    IBR 0.573, vs PGA 0.618 (0.570–0.680).
  - Reading: the two methods are close. α-potential is better against static
    strategies, self-play better against random, they tie against IBR, and
    head-to-head is about even.
  - Deployment cuts in opposite directions: mixing helps α-potential (PGA >
    argmax), argmax helps self-play.
  - Full table: `analysis/server_results/designb_preparation/table1_summary.csv`.

### ✅ State-encoder layout bug found and fixed (2026-10-22)
- **The bug.** KukaTennisEnv's raw obs is concat(qpos[:18], qvel[:18], ...)
  (mujoco_env_comp.py:601/738), i.e. [P1 qpos, P2 qpos, P1 qvel, P2 qvel]. The
  legacy `nash_skills/v2/state_encoder.py` was written for [P1 qpos, P1 qvel, P2
  qpos, P2 qvel]:
  - P1's state actually held both robots' positions and no velocities;
  - P2's state held both robots' velocities and no positions.

  V3's `_role_raw_slices` reads the same slices. Live proof:
  `scripts/check_obs_layout.py`.
- **The fix (Design B only).** `state_encoder_attack_defend` is rebuilt as a joint
  state (still 90 dims, choice slots unchanged at [74:90]):
  - both robots' qpos + qvel;
  - the ball;
  - both racket targets and tracking errors;
  - the attacker one-hot (paper's iota);
  - an acting-player one-hot;
  - the choices.

  A real-env test pins each slot to MuJoCo's qpos/qvel.
- **Guards.** The new encoder tag `attack_defend_joint_v2` is required by:
  - the trainer (old data rejected);
  - the Table 1 driver (old Phi/Q dirs refused);
  - the self-play loader (checkpoints carry a `.json` sidecar).
- **Legacy encoder.** Behaviour is kept for existing main-pipeline checkpoints;
  only its docstring and labels are corrected.
- **Impact.** The Table 1 numbers above were produced with the old encoder. For
  α-potential it was a consistent P1 view (both positions); self-play's P2 got no
  positions. Rerun pending.

### ✅ Paper experiments rerun with the fixed encoder (2026-10-23)
Full details: `analysis/server_results/designb_joint/README.md`.

| Method (3 seeds) | Static mean | Static min | Random | IBR | Other proposed |
|---|---|---|---|---|---|
| α-potential, PGA, γ 0.99 | **0.696** | **0.487** | **0.673** | **0.540** | 0.530 |
| Self-play, sampled | 0.611 | 0.422 | 0.592 | 0.468 | 0.470 |
| α-potential, PGA, γ 0.7 | 0.649 | 0.422 | 0.597 | 0.432 | 0.467 |
| Self-play, argmax | 0.636 | 0.403 | 0.607 | 0.468 | 0.513 |

- With the discount matched to self-play (γ 0.99), α-potential leads on every
  column, and beats sampled self-play head-to-head 0.530.
- At γ 0.7 the two methods are close.
- Both are slightly below the old-encoder numbers, within the seed spread.
- Still not run: Sec. 5.2 best-response gains, the horizon H sweep, and the
  stochasticity 2×2.

### ✅ α-potential with ~3.5× more data (2026-10-24)
Full details: `analysis/server_results/designb_big/README.md`.
- Data: 63,881 decisions, matching self-play's budget.
- **PGA γ 0.99: static mean 0.722, static min 0.572, random 0.713, IBR 0.540.**
  It beats sampled self-play head-to-head 0.562.
- Data-volume curve: the static mean rises 0.666 → 0.723 from 580 to 8,151 training
  rallies, and Q MSE falls 1.10 → 0.63. α (max) stays at about 2.2–2.5.
- γ 0.7 at full data is worse (0.628 static mean).

---

## Suggested order, once the server is back and the teammate's code lands

**Superseded by explicit user direction**: proceeding now with the
attacker/defender Design B Table 1 pipeline (data collection -> Phi/Q
training -> self-play -> Table 1 assembly) on top of the just-passed
reactive-mechanism gate, ahead of the numbered list below. That list
remains the plan for the main (non-attacker/defender) pipeline tracks
once this work lands. The Design B order now lives in section 6 below;
its item 0 (defend semantics) should be settled before the full-scale
collection, since every dataset and model depends on it.

1. Pull the skill-recalibration + deterministic-mode code once pushed;
   confirm it applies cleanly on top of the corrected gantry range.
2. Re-run the self-play checkpoint sweep under the corrected gantry range
   (already queued, blocked on connectivity).
3. Resolve the skill-design question explicitly (section 3) before training
   anything new against the recalibrated targets — decide whether `full1`/
   `full2` remain a valid baseline or whether the comparison resets.
4. Gantry Φ retrain (the best-motivated fix for the Φ-correlation problem;
   independent of the skill-target question).
5. Build the exploitability metric — the argmax finding is the strongest
   open lead right now and current metrics can't confirm or rule it out.
6. Decide on self-play retrain under the corrected range / recalibrated
   targets based on #2 and #3.
7. Multiple seeds, once the above have stabilized the recipe worth
   repeating.
8. Role-swapped head-to-head + shared benchmark table, once Φ is trustworthy.
9. Behavioral analysis (heatmaps) — the distinct, additional contribution
   beyond win-rate numbers, worth its own dedicated pass.

---

## 6. Remaining work for Table 1 (Design B), with expected results

Updated 2026-10-18. Everything in section 5 runs locally; nothing below
has a full-scale result yet. "Expected" = the prediction we commit to
before running, with the reason; "Changes the plan if" = the result that
would make us stop and rethink.

| # | Experiment | Where / blocker | Expected result | Changes the plan if |
|---|---|---|---|---|
| 0 | ✅ Preparation mechanism built beside the reactive one (`--defend-mechanism`) | done 2026-10-18 | — | — |
| 0a | ✅ Gate, exact matched pairs | done | was: neutral or better. **Got:** avg over positions −0.057 (fails the pre-set floor); center +0.030; mismatched −0.165; both-sided truncation 0.215 → 0.145 | — |
| 0b | ✅ Does the choice matter | done | was: uncertain, useful if ≥5-10 pt spread. **Got:** matched − mismatched +0.130 [+0.091, +0.166] for preparation, −0.013 (none) for reactive | — |
| 0c | ✅ Mechanism for Table 1: **preparation** (user decision, 2026-10-20) | done | — | — |
| 0d | ✅ Side-offset sweep done 2026-10-20: 0.25× +0.035, 0.5× +0.016, 0.75× +0.095 (passes the pre-set rule), 1.0× +0.219 (kept) | done | — | — |
| 0e | ✅ `SkillEnv(reset_targets=True)` / `--reset-targets`, used in every server run | done | — | — |
| 1 | ✅ Synced to `~/angelina/ping-pong-game-290-designb` (fresh folder + pinned env) | done | — | — |
| 2 | ✅ Data: pure 1256 rallies (85.7% kept), mixed 1684 (98.4%), 18,209 decisions | done | — | — |
| 3 | ✅ Φ/Q: main run (3 seeds) + MSE, pure-only, 25%/50% data ablations; α ≈ 2, falls with data | done (H flag still open) | — | — |
| 4 | ✅ Self-play PPO, 3 seeds × 500 iterations: trunc 0, entropies fell (attack 0.87–1.13, defend 0.83–0.91) | done | — | — |
| 5 | ✅ Self-play strategies + `scripts/eval_table1_attack_defend.py` (both sides, truncations separate) | done | — | — |
| 6 | ✅ **Table 1** complete: α-potential PGA 0.666 / 0.470 / 0.620 / 0.535; self-play sampled 0.621 / 0.448 / 0.665 / 0.537, vs PGA 0.517 (static mean / min / random / IBR) | done | — | — |
| 7 | Unilateral improvement R_i: PPO best response vs each frozen method (reuse self-play trainer, frozen opponent) + compact-family response (15 vertices + PGA) | server | both gains > 0 (finite search = lower bound); which method is less exploitable is the open question | — |
| 8 | Stochasticity 2×2: train nominal vs perturbed transitions × deploy sampled vs argmax | server, needs a perturbation model | argmax more exploitable than sampled; perturbed-trained more robust under perturbed eval | — |
| 9 | Simulation-based IBR (rollouts over H_IBR), as the paper defines it | server, costly | stronger opponent than the Q-based `ibr`/`ibr-q` proxies | — |
| 10 | Paper text: role/preparation semantics per #0, PPO + critic bootstrap details, truncation handling, label Q-based IBR | after #0 | — | — |

---

## 7. Paper-faithful rerun, both skill profiles (2026-10-25 → 26)

Folder: `analysis/server_results/paper_runs/` (README, `summary_tables.md`, launch scripts).
Controller db5df579 for everything; calibrated preparation poses (`preparation_cal`);
decisions at racket contacts (`--decision-timing contact`); per-player, role-gated
self-play (`selfplay_per_player.py`); Φ chosen on a validation split; simulated IBR.

| # | Experiment | Result |
|---|---|---|
| 1 | Preparation matters (matched − mismatched) | aggressive +0.39 [+0.32, +0.46], moderate +0.47 [+0.40, +0.53] (was +0.22 with crossing timing) |
| 2 | α-potential Table 1 (3 seeds) | aggressive 0.728 / 0.525 / 0.632 / Q-IBR 0.610 / sim-IBR 0.258; moderate 0.712 / 0.345 / 0.707 / 0.575 / 0.438 |
| 3 | Self-play sampled | aggressive 0.761 / 0.565 / 0.752 / 0.742 / 0.317; moderate 0.792 / 0.432 / 0.777 / 0.738 / 0.542 (31% of static rallies unresolved) |
| 4 | Self-play argmax | aggressive 0.797 / 0.568 / 0.823 / 0.810 / 0.492; moderate 0.850 / 0.540 / 0.875 / 0.823 / 0.642 (52% unresolved) |
| 5 | Head-to-head | α-potential wins 0.31 vs sampled, 0.25 vs argmax (aggressive); 0.31 / 0.19 (moderate) |
| 6 | γ (α-potential) | aggressive flat (0.7 0.708, 0.9 0.697, 0.95 0.687, 0.99 0.705–0.746); moderate prefers high γ |
| 7 | Data volume (α-potential) | aggressive weakly up with data; moderate not monotone (open) |
| 8 | Φ selection / loss | validation selection raises static min (0.505 vs 0.435; 0.455 vs 0.250); LSE beats MSE on α max (2.36 vs 3.60) |
| 9 | Computation per decision | PGA ~0.22 s, simulated IBR ~27 s |

Caveats: decided-only win rates hide self-play's truncations on the moderate profile
(counting them as half a point: static mean 0.768 sampled, 0.771 argmax); self-play saw
~32k training rallies vs ~3.4k for α-potential; `tests/test_skills.py` fails since the V3
commit 5db8b1e (not this work).

Not run: Sec 5.2 horizon H, Sec 5.3 stochasticity 2×2, best-response gains R_i, Q3 patterns.

---

## 8. α-potential implementation check and ablations (2026-10-27, after the mentor meeting)

Meeting asks: γ sweep 0.5 → 1 and training-data volume (~1/10, 1×, 10×) for α-potential,
scored against random and the 15 static strategies only (plots for the paper). Teammate's
diagnostic: Φ/Q sign agreement for unilateral deviations (their V3: attacker 87%, defender 58%).

### ✅ Implementation check — found and fixed: the Q models overfit
- Q1/Q2 were trained full-batch for 1500 epochs with no stopping rule. Aggressive γ 0.99:
  train MSE 0.22 vs held-out 0.75 (predicting 0 gives 0.875); at 1/8 data, 0.013 vs 1.05.
- Consequence: one player's single choice change moved its Q by up to ±2, the learned Φ was
  no better than a constant (held-out mean |e| 0.603 vs 0.616; α max 2.36 vs 2.00), and Φ moved
  ~0.15× as much as Q. Explains the earlier puzzles (α never shrinking, less data looking better
  on moderate, γ 0.7's "lower error").
- Fix: `--q-select val` (early stopping on the validation split). Main setting, seed 0:
  held-out Q MSE 0.55 / 0.53 (aggressive / moderate); Φ now beats the constant potential
  (mean |e| 0.270 vs 0.345, 0.319 vs 0.414); α p95 0.74 / 0.88 (was 1.85 / 1.92); sign
  agreement 0.62–0.76 with correlation up to +0.61. `scripts/phi_sign_agreement.py` and the
  trainer report (`heldout_residuals_constant_phi`, `heldout_q1_mse_predict_zero`) track this.
- Our preparation changes agree with Q as often as attack changes (unlike V3's 58%): our
  preparation moves the whole racket target, V3's only the gantry.

- Win rates with the fix (seed 0, Table 1 static + random; "half" counts an unresolved rally as half a point):

  | Profile | Q selection | static mean (half) | static min | random | unresolved | P1 / P2 |
  |---|---|---|---|---|---|---|
  | aggressive | last epoch | 0.705 (0.704) | 0.505 | 0.620 | 2.1% | 0.738 / 0.646 |
  | aggressive | early-stopped | **0.800 (0.784)** | 0.545 | 0.660 | 9.9% | 0.787 / 0.796 |
  | moderate | last epoch | 0.694 (0.694) | 0.455 | 0.685 | 1.2% | 0.603 / 0.768 |
  | moderate | early-stopped | **0.794 (0.782)** | 0.380 | 0.760 | 6.9% | 0.804 / 0.780 |

  Self-play seed 0 for comparison (half): aggressive sampled 0.766, argmax 0.802; moderate
  sampled 0.753, argmax 0.731. The earlier "self-play beats α-potential" was mostly the Q bug.
  The fixed α-potential Table 1 rows (3 seeds, Q-IBR, simulated IBR, head-to-head) are rerun by
  `scripts/launch_alpha_main_fixed.sh` into `runs_paper/alpha_fixed/`.
- 10× data: 47.5k / 47.8k rallies (362k decisions) per profile, P1 wins 0.50.

### 🚧 Ablations (`scripts/launch_alpha_ablations.sh`, QSEL=val, `runs_paper/ablations_qval/`)
- γ ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0} × 3 seeds × 2 profiles, base data.
- Data volume: nested subsets of a 10× data set (≈340, 1.1k, 3.4k, 11k, 34k train rallies), γ 0.99, 3 seeds.
- Summaries: `scripts/summarize_alpha_ablations.py` (tables, CSVs, plots).

### ✅ γ sweep done (fixed Q, 3 seeds, 60 decided rallies per side per opponent)
Tables/plots: `analysis/server_results/paper_runs/ablations_qval/summary/` (gamma_ablation.png,
gamma_*.csv, ablation_tables.md).

| γ | aggressive static mean / min / random | moderate static mean / min / random |
|---|---|---|
| 0.5 | 0.734 / 0.425 / 0.692 | 0.693 / 0.306 / 0.728 |
| 0.6 | 0.734 / 0.411 / 0.686 | 0.702 / 0.247 / 0.725 |
| 0.7 | 0.736 / 0.436 / 0.667 | 0.714 / 0.208 / 0.761 |
| 0.8 | 0.769 / 0.436 / 0.708 | 0.750 / 0.283 / 0.742 |
| 0.9 | 0.761 / 0.497 / 0.622 | 0.744 / 0.297 / 0.744 |
| 0.95 | 0.757 / 0.478 / 0.711 | 0.756 / 0.300 / 0.736 |
| 0.99 | 0.764 / 0.525 / 0.656 | 0.790 / 0.292 / 0.742 |
| 1.0 | 0.750 / 0.453 / 0.650 | 0.784 / 0.278 / 0.744 |

- Static mean rises with γ (aggressive +0.03, moderate +0.10 from 0.5 to 0.99); γ 0.99 is best
  or tied on every column except random (flat, noisy). Static minimum also best near 0.99 on
  aggressive. Unresolved rallies barely matter for α-potential (half-point counting moves < 0.01).
- Φ's gain over a constant potential shrinks as γ grows (mean |e| ratio 0.73 at 0.5 → 0.80 at
  0.99 → 0.83 at 1.0): long-horizon values are harder to fit from the same data (mentor's point).
- Weak spot, moderate profile: opponents who always attack right_short and prepare right/center
  (0.35 / 0.39 at γ 0.99; the minimum in 18 of 24 runs). PGA's strategy on held-out states
  prepares center 81–97% of the time and right only 3–19%; right_short is the shot this controller
  returns worst. α-potential plays one fixed equilibrium-style strategy per state and does not
  adapt to an opponent who repeats one shot.

### ✅ Data volume done (fixed Q, γ 0.99, 3 seeds; nested subsets of the 10× set)
`ablations_qval/summary/volume_ablation.png`, `volume_*.csv`.

| train rallies | aggressive static mean / min / random (unresolved) | moderate static mean / min / random (unresolved) |
|---|---|---|
| 340 | 0.645 / 0.200 / 0.486 (60%) | 0.752 / 0.204 / 0.725 (37%) |
| 1.1k | 0.719 / 0.414 / 0.636 (15%) | 0.762 / 0.285 / 0.750 (43%) |
| 3.4k (base) | 0.778 / 0.428 / 0.708 (3%) | 0.758 / 0.171 / 0.756 (40%) |
| 11k | 0.785 / 0.431 / 0.742 (7%) | 0.759 / 0.292 / 0.781 (14%) |
| 34k (10×) | 0.745 / 0.302 / 0.722 (16%) | 0.818 / 0.300 / 0.844 (25%) |

- Static mean and random improve with data (aggressive up to ~11k rallies, moderate mainly at 10×);
  the static minimum does not (weak spot is the method, not data).
- Held-out Q error falls monotonically (0.77 → 0.52 aggressive, 0.72 → 0.45 moderate), but α rises
  (p95 0.24 → 0.94/1.05) and Φ's gain over a constant potential shrinks (ratio 0.40 → 0.93): sharper,
  more accurate values expose the game's zero-sum structure, which a single potential cannot fit.
  At 10× data Φ barely beats a constant on aggressive, consistent with the dip there. Next: a larger
  Φ network / longer Φ training at 10×.

### ✅ Table 1 with the fixed Q (`runs_paper/alpha_fixed/`, `alpha_fixed/table1_updated.md`)

| Profile | Method | Static mean | Static min | Random | IBR (Q) | IBR (sim) | vs other |
|---|---|---|---|---|---|---|---|
| aggressive | α-potential, fixed Q | 0.760 | 0.495 | 0.658 | 0.542 | 0.304 | 0.31 / 0.23 |
| aggressive | Self-play sampled | 0.761 | 0.565 | 0.752 | 0.742 | 0.317 | 0.69 |
| aggressive | Self-play argmax | 0.797 | 0.568 | 0.823 | 0.810 | 0.492 | 0.78 |
| moderate | α-potential, fixed Q | 0.794 | 0.313 | 0.775 | 0.663 | 0.550 | 0.33 / 0.22 |
| moderate | Self-play sampled | 0.792* | 0.432* | 0.777 | 0.738 | 0.542 | 0.67 |
| moderate | Self-play argmax | 0.850* | 0.540* | 0.875 | 0.823 | 0.642 | 0.78 |

Correction to the seed-0 reading above: the fix brings α-potential level with sampled self-play
on static mean, random and simulated IBR, but self-play still wins head-to-head (0.67–0.78) and
has the higher static minimum. Q-based IBR got harder for α-potential on aggressive (0.610 → 0.542)
because that opponent uses the same, now better, Q models.
