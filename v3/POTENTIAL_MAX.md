# Potential Max in V3

`potential_max` selects the global maximum of the learned 5x3 potential matrix:

```
(a_star, d_star) = argmax_(a,d) Phi(s,a,d)
```

The attacker executes `a_star`; the defender executes `d_star`. Each physical
player can configure its attack and defense policies independently. Another
opponent policy can execute a different component from the proposed pair.
Ties use the first flat entry, in attack-major/defense-minor order. Existing
Nash-Q/BR-Q payoff signs and all training/PPO code remain unchanged.

## Detailed evaluation

Supply your existing compatible Q, potential and PPO checkpoints. For example:

```bash
python3 v3/eval_q_policy.py \
  --model models/v3/q_preposition_onehot_50k_aggressive.pth \
  --potential-model models/v3/potential_50k_aggressive.pth \
  --ppo logs/best_model_tracker1/best_model \
  --p1-policy potential_max --p2-policy random \
  --skill-profile aggressive --planner legacy \
  --episodes 50 --steps 1000 --seed 0 \
  --output-dir skill_eval/v3/potential_max_vs_random
```

The evaluator loads the potential with the normal V3 inference path. It avoids
an unnecessary Nash solve for Potential Max versus Random, Fixed, or another
potential policy. Nash-Q and BR-Q opponents still receive the Nash solution
needed by their existing selection/fallback logic.

## Fixed-grid evaluation

```bash
python3 v3/eval_policy_fixed_grid.py \
  --model models/v3/q_preposition_onehot_50k_aggressive.pth \
  --potential-model models/v3/potential_50k_aggressive.pth \
  --ppo logs/best_model_tracker1/best_model \
  --policies potential_max --other-opponent potential_max \
  --episodes 50 --seeds 0 --both-sides \
  --output-dir skill_eval/v3/potential_max_fixed_grid
```

Potential Max is also included in the batch script's default policy list.
These examples run simulations only when invoked; the human comparison below
reads saved logs without simulations or checkpoint loading.

## Original-Q unilateral regrets

Both evaluators write `potential_max_regrets.csv` and
`potential_max_regret_summary.json` when Potential Max is used. Regrets are
computed at the proposed potential-maximizing pair, in original Q units:

```
attacker_regret = max_a Q(a,d_star) - Q(a_star,d_star)
defender_regret = Q(a_star,d_star) - min_d Q(a_star,d)
```

The default frequency threshold requires both regrets strictly below 0.05;
configure it with `--potential-max-regret-threshold`. Candidate and executed
pairs are recorded separately. These sidecars do not change selection or
scoring, and a potential maximum need not be a Nash equilibrium of Q.

## Compare saved Potential Max decisions with human landings

Prepare the incoming human comparison inputs using
[human_comparison/README.md](human_comparison/README.md). Then:

```bash
OPENBLAS_NUM_THREADS=1 python3 v3/human_comparison/analyze_random_similarity.py \
  --potential-policy potential_max \
  --potential-decisions skill_eval/v3/potential_max_vs_random/decisions.csv \
  --center-band 0.25 --bootstrap 2000 --seed 42 \
  --output-dir v3/human_comparison/reports/potential_max_vs_human_center_0p25

python3 v3/human_comparison/plot_human_rectangular_skill_bars.py
python3 v3/human_comparison/plot_human_robot_skill_bars.py
```

Use a detailed `decisions.csv` with Potential Max P1 and Random P2. A fixed-grid
batch summary alone cannot recover conditional skill selection. Existing
Nash-Q inputs remain separate; new Potential Max rows retain their actual
policy name and sample count. `--ibr-decisions` remains a backward-compatible
alias for `--potential-decisions`.

Human landings use four base blocks split at depth 0.51 and lateral 0, with
Center Safe depth [0.20,0.70] and lateral [-0.35,0.35] overriding those blocks.
The position-center half-band is independent of the landing regions. Robot
bars are selected commands, not inferred physical landings. Outputs include
JS, TV, Spearman/Pearson correlation, bootstrap intervals and nine-context
side-by-side charts. Exactly uniform correlation is undefined. Rank centering
uses new NumPy arrays so pandas Copy-on-Write/read-only views are supported.
