"""Collect the pure-strategy payoff matrix for the attacker/defender game.

Entry (i, j) is ego's win rate when ego commits to joint action i for the whole
rally and the opponent commits to j. Both players hold their choice fixed, so
the result is an ordinary matrix game -- which a zero-sum LP solves exactly,
giving the equilibrium this stochastic game's one-shot abstraction admits.

Only the upper triangle is played: the game is symmetric and zero-sum, so
A[j][i] = 1 - A[i][j], and the diagonal is 0.5 by construction. That halves the
work and makes the antisymmetry exact rather than noisy.

    python scripts/payoff_matrix_ad.py --rallies 100 --jobs 4 --out logs/AD/payoff.npz
"""
import argparse
import itertools
import os
import time

import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "cgl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nash_skills.env_wrapper import SkillEnv
from nash_skills.skills import N_SKILLS, skill_from_index
from nash_skills.defender import defender_from_index, joint_size, split_joint
from selfplay_5skill_ad import _build_ppo_obs, _swallow_step, N_JOINT
from stable_baselines3 import PPO

TABLE_SHIFT = 1.5


def play(env, ppo, i, j, n_rallies, max_steps, rng):
    """Win rate for ego playing pure i against pure j."""
    e_s, e_d = split_joint(i, N_SKILLS)
    o_s, o_d = split_joint(j, N_SKILLS)
    wins = losses = trunc = 0
    for _ in range(n_rallies):
        env.set_skills(skill_from_index(e_s), skill_from_index(o_s))
        env.set_defenders(defender_from_index(e_d), defender_from_index(o_d))
        obs, info = env.reset()
        for _step in range(max_steps):
            a1, _ = ppo.predict(_build_ppo_obs(obs, info, 1), deterministic=True)
            a2, _ = ppo.predict(_build_ppo_obs(obs, info, 2), deterministic=True)
            act = np.zeros(18, dtype=np.float32)
            act[:9] = a1[:9]
            act[9:] = a2[:9]
            obs, _r, done, _t, info = _swallow_step(env, act)
            if done:
                if info.get("winner") == "ego":
                    wins += 1
                else:
                    losses += 1
                break
        else:
            trunc += 1
    decided = wins + losses
    return (wins / decided if decided else float("nan")), wins, losses, trunc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo-model", default="logs/best_model_tracker1/best_model.zip")
    ap.add_argument("--rallies", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logs/AD/payoff.npz")
    # Both players holding a pure strategy is exactly the regime that produces
    # long unfinishable rallies, so a cell costs far more than a mixed-policy
    # eval of the same size. Shard the pair list across processes.
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    env = SkillEnv(proc_id=1, history=4, skill_profile="aggressive")
    ppo = PPO.load(args.ppo_model, device="cpu")
    rng = np.random.default_rng(args.seed)

    n = N_JOINT
    A = np.full((n, n), np.nan)
    np.fill_diagonal(A, 0.5)          # a strategy against itself is a coin flip
    trunc_total = 0
    pairs = list(itertools.combinations(range(n), 2))
    pairs = pairs[args.shard::args.num_shards]
    t0 = time.time()
    for k, (i, j) in enumerate(pairs, 1):
        wr, w, l, t = play(env, ppo, i, j, args.rallies, args.max_steps, rng)
        A[i, j] = wr
        A[j, i] = 1.0 - wr            # exact antisymmetry, not a second noisy estimate
        trunc_total += t
        if k % 10 == 0 or k == len(pairs):
            el = time.time() - t0
            print(f"  {k}/{len(pairs)} pairs  {el/60:.1f} min elapsed, "
                  f"{el/k*(len(pairs)-k)/60:.1f} min left", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # Each shard writes only the cells it played; merge_payoff_shards combines.
    np.savez(args.out, A=A, rallies=args.rallies, trunc_total=trunc_total,
             shard=args.shard, num_shards=args.num_shards)
    print(f"\nwrote {args.out}   truncated rallies: {trunc_total}")


if __name__ == "__main__":
    main()
