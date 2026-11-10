# Design B: α-potential and self-play for the attacker–defender table-tennis game

This is the pipeline behind the L4DC α-PLAYER results: a two-player MuJoCo table-tennis game where, at every
exchange, each player holds a **return target** (5 options) and a **preparation position** (3 options), and two
ways of choosing them — the **α-potential** method (Sec. 4.1 of the draft: Q → Φ → online argmax) and
**self-play RL** (Sec. 4.2). It runs on top of the existing low-level PPO racket controller; nothing here changes
the physics.

Everything below assumes you run from the repo root with

```bash
export MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONPATH=.
```

(`MUJOCO_GL=cgl` on macOS.) The Python environment is the project's usual one (`venv/`, or
`~/angelina/envs/designb` on the server).

## 1. The game in one paragraph

Both players are driven step by step by the same low-level controller (`logs/best_model_tracker1/best_model`;
md5 `db5df579…` is the retrained tracker the reported numbers use — locally that file may be under
`logs/best_model_tracker2`). On top of it, each player holds a pair `(target, preparation)`:

- **Targets** (`nash_skills/skills.py`, profile `aggressive`): `left`, `left_short`, `center_safe`, `right_short`,
  `right`. Short shots land 0.40 m past the net, deep shots 1.00 m, sides at y ≈ ±0.65 m.
- **Preparations** (`nash_skills/defend_positions.py`): `left`, `right`, `center`. With the mechanism
  `preparation_cal` these are calibrated racket poses (`nash_skills/prep_pose_tables.py`), the same three for every
  method and opponent.

A **decision** is when both players (re)pick their pair. `--decision-timing contact` decides at each racket
contact (the draft's "exchange begins after the preceding racket contact"); `--decision-timing crossing` decides
when the ball crosses the net. The receiver is the attacker (its target is live), the player who just hit is the
defender (its preparation is live). Models, checkpoints and evaluations all record the timing, mechanism and
profile they were made with, and the evaluator refuses to mix them.

## 2. Files

### Game layer (`nash_skills/`)

| File | What it is |
|---|---|
| `skills.py` | The five targets per profile; `aggressive` is the paper's. |
| `defend_positions.py` | The three preparation names and indices. |
| `prep_pose_tables.py`, `defend_mechanisms.py` | Calibrated preparation poses and how a preparation choice is applied (`preparation_cal`). |
| `decision_clock.py` | When a decision fires (`contact` / `crossing`) and who the attacker is. |
| `env_wrapper.py` | `SkillEnv`: the MuJoCo env plus target/preparation application; adds ball spin and rally status to `info`. |
| `rally_status.py`, `winner_inference.py` | Rally bookkeeping and who won a finished rally. |
| `v2/joint_action.py` | Joint index ↔ (target, preparation) pair; 15 pairs per player. |
| `v2/state_encoder_attack_defend.py` | The state vector. `attack_defend_joint_v3` (97 dims) = robots, ball, target poses, spin, rally status, plus both players' pair one-hots in slots `[74:90]`. |

### α-potential (`nash_skills/v2/`, `scripts/`)

| File | What it is |
|---|---|
| `v2/collect_data_attack_defend.py` | Collects rallies. `--sampling grid` cycles through all 225 joint profiles with each player holding its pair (Eq. 7); `--sampling mixed` draws a randomized θ per player per rally. |
| `scripts/merge_rally_shards.py` | Merges collector shards into one pickle. |
| `v2/train_q_model_5skill_v3_attack_defend.py` | Trains Q₁, Q₂ (returns with horizon `--horizon`, Eq. 9) and the potential Φ (`--phi-loss lse`: smooth max of the unilateral-deviation residuals, Eq. 11). Writes `model1/model2/model_p_attack_defend.pth` and `heldout_attack_defend.json` (settings + held-out fit, α statistics). |
| `table_rules.py` | Online decision rules on the 15 × 15 tables: `nash-p-argmax` (**Eq. 12**), `nash-p-ibr`, `nash-p-br`, `nash-p-adaptive`, `nash-p-minimax`, `nash-q`, `br-q`, and the exploitation ablation. |
| `v2/mixed_strategy.py` | `nash-p-pga`: projected gradient ascent on Φ over mixed θ, then a sampled action. |
| `v2/matrix_game.py` | Zero-sum LP solver used by `nash-q`. |
| `v2/potential_lp_joint.py`, `scripts/fit_potential_lp.py` | Alternative Φ fit (exact per-state LP, then regression) — an ablation, not the default. |
| `eval_matchup_attack_defend.py` | Plays matches: `make_picker_joint` turns a strategy spec into a picker; `run_matchup_joint` runs rallies and scores them. |
| `scripts/eval_table1_attack_defend.py` | Table 1 for one method: the 15 static opponents, random, IBR opponents, any other strategy spec. |
| `scripts/eval_response_gains.py`, `response_gains.py` | Table 2: unilateral response gains by simulated continuations. |
| `scripts/diagnose_unfinished_rallies.py` | Replays stalled cells with a long step cap, so win rates don't depend on the cap. |
| `ibr_sim.py` | The simulated iterated-best-response opponent (slow: ~44 s per decision). |
| `scripts/summarize_paper_alpha.py` | Builds Tables 1–3, Figure 2/3 data and plots from a run directory. |

### Self-play (`nash_skills/v2/`)

| File | What it is |
|---|---|
| `v2/selfplay_attack_defend.py` | `MetaPolicy` (trunk (64, 32); target head, preparation head, value head), GAE and the PPO clipped loss. |
| `v2/selfplay_rollout.py` | One rally with role-gated decisions; only the live role's head acts and gets credit. `save_per_player` / `load_per_player` for checkpoints (`.pth` + `.pth.json` sidecar with the settings). |
| `v2/selfplay_per_player.py` | Training: one actor-critic per player, PPO, alternating update rounds against a frozen snapshot pool. |
| `scripts/plot_selfplay_training.py` | Training curves from the CSV log. |

Strategy specs for `--method` / `--opponents`: `selfplay:<ckpt>` (sampled) and `selfplay-argmax:<ckpt>`.

### Cross-checks against the V3 pipeline (Kexin's)

| File | What it is |
|---|---|
| `kexin_v3_player.py` | Her Φ and selection rules playing inside this game (`kexin-v3:<rule>:<model>` spec). |
| `v3_selfplay/` | Self-play retrained inside **her** game and scored by her evaluator (`python -m v3_selfplay.train`, `python -m v3_selfplay.her_eval`). Needs her `v3/` package. |
| `scripts/summarize_vs_kexin.py` | Tables for those comparisons. |

### Launchers (`scripts/launch_*.sh`)

Server scripts that run a whole pipeline; each is idempotent (skips finished outputs, lock file) and writes a
`MANIFEST.txt`. The ones that produced the current results:

| Launcher | Produces |
|---|---|
| `launch_paper_v3.sh` | Contact timing, pure θ: data, Q/Φ (γ sweep, data-volume sweep), all rules, self-play, Tables 1–3, figures → `runs_v3/paper`. |
| `launch_paper_v3_mixed.sh` | Randomized-θ data + PGA → `runs_v3/paper_mixed`. |
| `launch_paper_v3_crossing.sh` | The same pipeline at **net-crossing** timing, Eq. 12 / Adaptive Match / PGA / self-play, no IBR → `runs_v3_crossing/`. |
| `launch_paper_lp.sh`, `launch_paper_v3_exploit.sh` | Ablations (LP Φ fit; exploitation rule). |
| `launch_vs_kexin_v3_ours.sh`, `launch_selfplay_kexin_v3.sh` | Cross-pipeline comparisons. |

Results copied from the server live under `analysis/server_results/paper_runs/`.

## 3. Running the α-potential pipeline

```bash
SET="--defend-mechanism preparation_cal --decision-timing contact --skill-profile aggressive --encoder attack_defend_joint_v3"

# 1. data: shards of fixed-pair rallies (run many in parallel with different seeds), then merge
python nash_skills/v2/collect_data_attack_defend.py --sampling grid --decisions 1500 --max-rallies 3000 \
    --max-steps 2500 $SET --reset-targets --seed 5000 --output runs/data/grid_0.pkl
python scripts/merge_rally_shards.py --output runs/data/rallies.pkl runs/data/grid_*.pkl

# 2. Q and Phi (paper protocol: H = 10, smooth-max loss, validation checkpoints)
python nash_skills/v2/train_q_model_5skill_v3_attack_defend.py --rallies runs/data/rallies.pkl \
    --theta-mode pure --pure-data-only --horizon 10 --gamma 0.99 --phi-loss lse \
    --phi-select val_residual --q-select val --seed 0 --out-dir runs/phi/s0

# 3. Table 1 for the online rule (Eq. 12 = nash-p-argmax)
python scripts/eval_table1_attack_defend.py --defend-mechanism preparation_cal --decision-timing contact \
    --skill-profile aggressive --method nash-p-argmax --model-dir runs/phi/s0 --episodes 100 --workers 8 \
    --max-attempt-factor 3 --opponents static random nash-p-adaptive --output runs/table1/argmax_s0.json

# 4. (optional) replay the cells that stall at the step cap, and Table 2 gains
python scripts/diagnose_unfinished_rallies.py --method nash-p-argmax --model-dir runs/phi/s0 \
    --opponents center_safe/center --rallies-per-side 30 --max-steps 20000 --output runs/replay/argmax_s0.json
python scripts/eval_response_gains.py --method nash-p-argmax --model-dir runs/phi/s0 --skill-profile aggressive \
    --points 32 --workers 8 --output runs/gains/argmax_s0.json
```

For the mixed-θ variant: collect with `--sampling mixed` (seeds distinct from the grid shards), merge grid +
mixed shards, train with `--theta-mode mixed` (drop `--pure-data-only`), evaluate `--method nash-p-pga`.

Important flags: `--decision-timing`, `--defend-mechanism` and `--skill-profile` must be the same in collection,
training and evaluation — the evaluator checks the model's `heldout_attack_defend.json` and refuses mismatches.
`--max-attempt-factor 3` caps re-draws of rallies that hit the step limit (the stalled cells are replayed
separately instead). Use 3 seeds and report mean ± sd.

## 4. Running self-play

```bash
python nash_skills/v2/selfplay_per_player.py --iterations 1000 --rallies-per-iter 32 --round-iters 25 \
    --defend-mechanism preparation_cal --decision-timing contact --skill-profile aggressive --reset-targets \
    --gamma 0.99 --state-mode no_choices --encoder attack_defend_joint_v3 --seed 0 \
    --output runs/selfplay/s0.pth --log runs/selfplay/s0.csv
```

- `--state-mode no_choices` is the paper's π(x, role): the state without the two players' pair one-hots.
- PPO defaults: Adam 1e-3, clip 0.2, 4 epochs, minibatch 64, GAE λ 0.95, entropy 0.05, value coef 0.5.
- Alternating rounds: player 1 learns for `--round-iters` iterations against player 2's frozen copy (plus older
  snapshots drawn with `--snapshot-prob`), then the roles swap.
- Reward is ±1 at the end of the rally; a rally cut by the step cap bootstraps from the critic.

Evaluate it exactly like a rule, with the checkpoint as the method (the sidecar's settings are checked):

```bash
python scripts/eval_table1_attack_defend.py --defend-mechanism preparation_cal --decision-timing contact \
    --skill-profile aggressive --method selfplay-argmax:runs/selfplay/s0.pth --model-dir runs/phi/s0 \
    --episodes 100 --workers 8 --max-attempt-factor 3 \
    --opponents static random nash-p-adaptive nash-p-argmax --output runs/table1/sp_argmax_s0.json
```

(`--model-dir` is needed whenever an opponent is a Φ/Q rule.) Head-to-head α vs self-play is the
`nash-p-argmax` cell of that file, or `selfplay:<ckpt>` in the α method's opponent list.

## 5. Reading results

```bash
python scripts/summarize_paper_alpha.py --root runs_v3/paper --out-dir runs_v3/paper/summary --profiles aggressive
```

expects `<root>/aggressive/{phi,table1,gains,diag_unfinished,ibr_sim}/` with files named `<method>_s<seed>.json`
and writes the markdown tables (`paper_tables.md`), figure data and plots to `--out-dir`. Win rates are percentages of **finished** rallies, both table
sides pooled, mean ± sd over seeds; the static columns use the replayed (played-to-the-end) values where a replay
exists.

## 6. Tests

```bash
PYTHONPATH=. python -m pytest tests -q
```

Pure-logic tests run without MuJoCo (fake environments drive the real role trackers); the few that need the
simulator skip when it is missing. Please add a test next to any behaviour you change.

## 7. Conventions that keep results comparable

- Report the `aggressive` profile only.
- Keep a run's three settings (timing, mechanism, profile) identical across its data, models and evaluations.
- Never run paper jobs from a checkout whose `nash_skills/skills.py` still has the old short-shot depth (1.60);
  the current targets are 1.90 (short), 2.50 (deep), 2.0 (center), the same points as the V3 pipeline.
- Re-draw/replay policy for stalled rallies as above; do not count unfinished rallies as losses.
