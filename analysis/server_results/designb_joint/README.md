# Design B + preparation — paper experiments with the fixed joint-state encoder (2026-10-23)

## Setup
- Encoder `attack_defend_joint_v2`: both robots' qpos/qvel, ball, both racket targets,
  attacker and acting-player one-hots, choices. Replaces the old encoder with the
  raw-observation layout bug (see `scripts/check_obs_layout.py`).
- Server `bluesclues`, `~/angelina/ping-pong-game-290-designb/runs_joint/`.
- Low-level controller: the retrained `best_model_tracker1` (md5 `db5df579…`).
- Per-rally target reset; 2500-step cap; skill profile `aggressive`.
- Full commands: `launch_joint.sh`, `launch_joint_h2h099.sh`; versions in `runs/MANIFEST.txt`.

## Data (`runs/rallies_prep_*.pkl`)
| | rallies | decisions | P1 win share | profiles covered |
|---|---|---|---|---|
| pure (all 225 fixed profiles) | 1191 | 9198 | 0.505 | 208 / 225 |
| mixed θ | 1709 | 9106 | 0.504 | 225 / 225 |

## Table 1 (`paper_table1_final.csv`)
Mean over 3 seeds (range in brackets); 200 decided rallies per opponent; both sides pooled.

| Method | Static mean | Static min | Random | IBR* | Other proposed |
|---|---|---|---|---|---|
| **α-potential, PGA, γ = 0.99** | **0.696** (0.679–0.705) | **0.487** (0.415–0.570) | **0.673** (0.650–0.720) | **0.540** (0.480–0.610) | **0.530** vs self-play |
| **Self-play, sampled, γ = 0.99** | 0.611 (0.586–0.633) | 0.422 (0.380–0.450) | 0.592 (0.515–0.640) | 0.468 (0.390–0.540) | 0.470 vs α-potential |
| α-potential, PGA, γ = 0.7 | 0.649 (0.623–0.695) | 0.422 | 0.597 | 0.432 | 0.467 |
| α-potential, argmax, γ = 0.7 | 0.607 (0.576–0.644) | 0.382 | 0.598 | 0.412 | 0.457 |
| Self-play, argmax | 0.636 (0.595–0.687) | 0.403 | 0.607 | 0.468 | 0.513 vs α-potential |

- *IBR is the Q-based stand-in. It uses the Q models of the row's own Φ run (the
  self-play rows use γ = 0.7 seed 0), so the IBR opponent is not identical across rows.
- Head-to-heads use the PGA opponent from Φ seed 0.
- The 5×3 static matrices are in `table1_static_matrix.csv`; per-run numbers are in
  `table1_summary.csv` and `gamma099/table1_summary.csv`.

## Φ fit (`paper_table_ablations.csv`)
| Setting | Train rallies | Held-out Q MSE | α = max\|e\| | p95 | mean |
|---|---|---|---|---|---|
| 25% / 50% / 100% data | 580 / 1160 / 2320 | 0.37 / 0.32 / 0.26 | 2.15 / 2.18 / 2.12 | 0.99 / 1.00 / 0.92 | 0.33 / 0.33 / 0.30 |
| MSE loss (vs smooth max) | 2320 | same | 2.14 | 0.80 | 0.27 |
| pure-only data | 953 | 0.21 | 2.10 | 1.10 | 0.41 |
| γ = 0.99 | 2320 | 0.95 | 2.38 | 1.91 | 0.56 |

Values at γ = 0.99 are near ±1 everywhere, so its residual scale is not comparable to γ = 0.7.

## Self-play training (`runs/selfplay/`)
3 seeds × 500 iterations × 32 rallies, PPO. At iteration 500:
- truncation 0;
- attack entropy 0.75–1.10 (max 1.61), defend entropy 0.76–0.93 (max 1.10).

## Compared with the old (buggy-encoder) runs (`../designb_preparation/`)
- α-potential PGA γ = 0.7: 0.666 → 0.649 static mean.
- Self-play sampled: 0.621 → 0.611 static mean, 0.665 → 0.592 vs random.

Both moved slightly down, within the seed spread. The new state drops the low-level
action history and adds unnormalized velocities, which may matter. The γ = 0.99
α-potential is the clear improvement.
