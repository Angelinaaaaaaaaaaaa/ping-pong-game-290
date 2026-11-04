# Paper-faithful Design B runs (2026-10-25 → 26)

Setup for every run: retrained controller db5df579, calibrated preparation poses per skill
profile (`nash_skills/prep_pose_tables.py`, measured by `scripts/calibrate_prep_poses.py`),
decisions at racket contacts, target reset, 2500-step cap. Two skill profiles: `aggressive`
and `moderate_aggressive`. Launch scripts: `launch_paper_runs.sh`, `launch_paper_eval.sh`,
`launch_paper_ibr_pga.sh`. All tables: `summary_tables.md`
(`python scripts/summarize_paper_runs.py --root analysis/server_results/paper_runs`).

- Data: per profile 24 shards (12 grid + 12 mixed), ~4.75k rallies / 36k decisions.
- α-potential: Q1/Q2 + Φ (smooth-max loss, mixed θ, Φ picked on a validation split),
  online PGA; 3 seeds plus γ, data-volume, selection and loss ablations.
- Self-play: one actor-critic per player, only the active role's head acts and is credited,
  alternating update rounds (25 iterations), snapshot pools; 1000 iterations x 32 rallies;
  3 seeds at γ 0.99 plus γ 0.9 / 0.7.
- Opponents: 15 static pairs, random, Q-based IBR, simulated IBR (2 rounds, H_IBR = 4
  exchanges, 15 vertices, exact rollouts; 80 decided rallies per method), head-to-head.

Win rates count decided rallies; truncation rates are in the JSON rows. On the moderate
profile self-play leaves 31% (sampled) / 52% (argmax) of static-opponent rallies
unresolved, which inflates its decided-only static numbers.
