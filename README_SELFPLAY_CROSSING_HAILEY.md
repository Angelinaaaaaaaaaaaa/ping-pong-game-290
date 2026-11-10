# Hailey: 81D net-crossing PPO and B1

Base: `angelina-alpha-selfplay` at `498cb89d8cb6063814ccf735cf43beb57629a49b`.
Apply this patch on that source, not on the older feature-selfplay tree. Preserve
Angelina's encoder, MuJoCo environment, calibrated preparation poses and low-level
controller. The new trainer reuses her PPO model/update design; it does not use
`v3_selfplay` (the separate Kexin-native 75D route).

## Design

- Independent player policies; learner alternates every 25 iterations while the
  opposing neural policy is frozen. Each rally retains one sampled opponent.
- 81D = v3 97D minus the 16 explicit previous-choice slots. Ball spin and rally
  status remain. Only the active role's head acts and receives PPO credit.
- Decision after a physical step iff `(previous_x-1.5)*(current_x-1.5)<0`.
  Initialize from reset `obs[36]`; no velocity-based initial event. Exact-plane
  endpoints do not trigger, matching Kexin's strict rule. No post-terminal action.
- Snapshot: 50% round-frozen / 50% historical policy when a pool is available.
  B1: 50% round-frozen / 50% uniformly selected fixed pair among 15 strategies.
  The fixed pair is held for the entire rally, including initialization.
- Four runs: snapshot gamma .99; B1 gamma .99, .7, .9. Seed 0 each,
  1,000 iterations x 100 rallies. Each checkpoint is evaluated sampled and argmax.
- MLP 64/32; lr .001; GAE .95; clip .2; 4 PPO epochs; minibatch 64;
  value weight .5; entropy .05; gradient clipping 1.0; max 2500 physical steps.

## Launch one run

Run from the repository root in an environment with the repository dependencies
and its low-level controller checkpoint. All generated outputs must remain in
an authorized `hailey` directory. Set cache paths there as needed.

```bash
mkdir -p hailey/runs/B1_gamma099
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PINGPONG_SKILL_SEMANTICS=legacy MUJOCO_GL=egl \
python nash_skills/v2/train_b1_v3.py \
  --variant b1 --gamma 0.99 --seed 0 \
  --iterations 1000 --rallies-per-iter 100 --round-iters 25 \
  --encoder attack_defend_joint_v3 --state-mode no_choices \
  --decision-timing crossing --defend-mechanism preparation_cal \
  --skill-profile aggressive --reset-targets \
  --ppo logs/best_model_tracker1/best_model \
  --output hailey/runs/B1_gamma099/model.pth \
  --log hailey/runs/B1_gamma099/train.csv
```

For macOS use `MUJOCO_GL=glfw`. Change variant/gamma and output directory for
other runs. CLI compatibility defaults are NOT the experiment configuration;
keep the explicit encoder, state-mode, timing and budget flags above.

## Evaluation and provenance

The evaluation runner is patched on Angelina's branch, preserving her Kexin-v3
picker and exploit-policy support. The clock initialization is also updated in
her original rollout, data collector and response-gain code, since they share
DecisionClock. This does not regenerate Q/Phi models or historical results.

Crossing PPO metadata must include `decision_clock_version=kexin_net_crossing_v1`.
Never relabel an older checkpoint as newly trained. The optional evaluator flag
`--allow-reference-timing-transfer` permits a contact-trained Q/Phi reference
only when explicitly requested, and records the difference in output metadata.
All other reference checks remain. Such results are timing-transfer experiments,
not fully matched crossing-trained alpha comparisons.

The current isolated rerun uses 50 attempts per side (100 total per matchup),
with truncations counted as nonwins and separately reported. Angelina's table
script normally reports wins / decided rallies. Its displayed summary therefore
must NOT be pasted as the current rerun's all-attempt table. Use saved per-side
wins, decided and truncated counts to compute wins/(decided+truncated), with
attempt cap equal to 50 per side. Always state the denominator. This patch does
not bundle the machine-specific orchestration and Kexin evaluation queue.

Current 81D crossing results are pending; old 90D/74D contact results are not
results of this rerun. Local/server duplicate seed-0 runs are not independent
training seeds. Do not mix their best rows.

The opponent audit's `static_drift` field is a constant legacy diagnostic,
not a measured validation result; inspect pair invariance in rollout tests.

## Validation scope

Run `python -B -m pytest -p no:cacheprovider tests/test_decision_clock.py`.
The delivery notes record checks actually performed on the prepared patch.
A short integration smoke run on the exact merged branch is required before a
new long training job. Existing running experiments were not altered by packaging.
