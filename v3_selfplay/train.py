"""
Self-play training in Kexin's V3 game, with the schedule of our Design B self-play
(nash_skills/v2/selfplay_per_player.py): one actor-critic per player; player 1 learns for --round-iters
iterations against player 2's frozen pool (the round's frozen copy, or with probability --snapshot-prob one of
its last --snapshot-buffer-size snapshots, drawn per rally), then player 2 learns against player 1's pool, and so
on. PPO on the head that acted. An iteration's rallies can be spread over --workers processes, each with its own
copy of her environment; every rally gets its own seed (environment reset and action sampling). Run from the root
of a checkout with her v3 package:

    MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONPATH=. python -m v3_selfplay.train --seed 0 --workers 8 \\
        --output runs_selfplay_v3/aggressive_s0.pth --log runs_selfplay_v3/aggressive_s0.csv
"""
import argparse
import csv
import multiprocessing as mp
import os
import random
import time
from copy import deepcopy

import numpy as np
import torch

from v3_selfplay.game import HER_MAX_STEPS, her_game, make_her_env
from v3_selfplay.policy import RolePolicy, save_policies
from v3_selfplay.ppo import build_batch, ppo_update
from v3_selfplay.rollout import play_rally

__all__ = ["learner_for_iteration", "is_round_start", "rally_seeds", "choose_opponents", "run_job", "train",
           "build_parser"]

PPO_MODEL_PATH = "logs/best_model_tracker1/best_model"
LOG_HEADER = ["iter", "learner", "win_rate", "trunc_rate", "mean_exchanges", "mean_steps", "n_decisions",
              "attack_usage", "defend_usage", "policy_loss", "value_loss", "entropy", "pool_size", "wall_time"]
_W = {}   # per worker process: env, ppo, game


def learner_for_iteration(it, round_iters):
    return 1 if ((it - 1) // round_iters) % 2 == 0 else 2


def is_round_start(it, round_iters):
    return (it - 1) % round_iters == 0


def rally_seeds(seed, it, n):
    """One seed per rally of iteration `it`: reproducible for a given run seed, fresh entropy when it is None."""
    entropy = None if seed is None else [int(seed), int(it)]
    return [int(s) for s in np.random.SeedSequence(entropy).generate_state(n)]


def choose_opponents(pool, round_opponent, n, snapshot_prob):
    """Per rally: an older snapshot from the opponent's pool with probability snapshot_prob, else the round's copy."""
    return [random.choice(pool) if pool and random.random() < snapshot_prob else round_opponent for _ in range(n)]


def _frozen(policy):
    snapshot = deepcopy(policy).eval()
    for param in snapshot.parameters():
        param.requires_grad_(False)
    return snapshot


def _policy_from_state(state_dict):
    net = RolePolicy()
    net.load_state_dict(state_dict)
    return net.eval()


def _load_ppo(path):
    from stable_baselines3 import PPO
    return PPO.load(path, device="cpu")


def _init_worker(env_kwargs, ppo_path):
    torch.set_num_threads(1)
    _W.update(env=make_her_env(**env_kwargs), ppo=_load_ppo(ppo_path), game=her_game())


def run_job(job):
    """One rally in a worker: (learner, learner weights, opponent weights, rally seed, step cap)."""
    learner, learner_state, opponent_state, seed, max_steps = job
    torch.manual_seed(seed)
    policies = {learner: _policy_from_state(learner_state), 3 - learner: _policy_from_state(opponent_state)}
    return play_rally(_W["env"], _W["ppo"], policies, learner, _W["game"], max_steps, seed=seed)


class _Sequential:
    def __init__(self, env, ppo_ctrl, game, max_steps):
        self.env, self.ppo, self.game, self.max_steps = env, ppo_ctrl, game, max_steps

    def run(self, learner, learner_policy, opponents, seeds):
        rallies = []
        for opponent, seed in zip(opponents, seeds):
            torch.manual_seed(seed)
            rallies.append(play_rally(self.env, self.ppo, {learner: learner_policy, 3 - learner: opponent}, learner,
                                      self.game, self.max_steps, seed=seed))
        return rallies

    def close(self):
        self.env.close()


class _Parallel:
    def __init__(self, workers, env_kwargs, ppo_path, max_steps):
        self.max_steps = max_steps
        self.pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(env_kwargs, ppo_path))

    def run(self, learner, learner_policy, opponents, seeds):
        learner_state = {k: v.detach().clone() for k, v in learner_policy.state_dict().items()}
        jobs = [(learner, learner_state, opponent.state_dict(), seed, self.max_steps)
                for opponent, seed in zip(opponents, seeds)]
        return self.pool.map(run_job, jobs, chunksize=1)

    def close(self):
        self.pool.close()
        self.pool.join()


def _make_runner(args, env, ppo_ctrl, game):
    if env is not None:
        return _Sequential(env, ppo_ctrl, game, args.steps)
    env_kwargs = {"skill_profile": args.skill_profile, "reset_mode": args.reset_mode,
                  "gantry_speed_scale": args.gantry_speed_scale, "planner": args.planner}
    if args.workers > 1:
        return _Parallel(args.workers, env_kwargs, args.ppo, args.steps)
    return _Sequential(make_her_env(**env_kwargs), _load_ppo(args.ppo), her_game(), args.steps)


def _seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def _log_row(it, learner, rallies, stats, pool_size, wall):
    n = len(rallies)
    attack = sum(r["usage"]["attack"] for r in rallies)
    defend = sum(r["usage"]["defend"] for r in rallies)
    return [it, learner, sum(r["done"] and r["reward"] > 0 for r in rallies) / n,
            sum(not r["done"] for r in rallies) / n, np.mean([r["exchanges"] for r in rallies]),
            np.mean([r["steps"] for r in rallies]), sum(len(r["actions"]) for r in rallies),
            " ".join(map(str, attack)), " ".join(map(str, defend)),
            stats["policy_loss"], stats["value_loss"], stats["entropy"], pool_size, wall]


def _meta(args):
    return {"iterations": args.iterations, "rallies_per_iter": args.rallies_per_iter, "round_iters": args.round_iters,
            "gamma": args.gamma, "gae_lambda": args.gae_lambda, "lr": args.lr, "entropy_coef": args.entropy_coef,
            "max_steps": args.steps, "skill_profile": args.skill_profile, "reset_mode": args.reset_mode,
            "planner": args.planner, "gantry_speed_scale": args.gantry_speed_scale, "ppo": args.ppo,
            "seed": args.seed}


def _print_row(it, args, learner, rallies, stats):
    wins = sum(r["done"] and r["reward"] > 0 for r in rallies) / len(rallies)
    print(f"[iter {it:4d}/{args.iterations}] learner=P{learner} win={wins:.2f} "
          f"exch={np.mean([r['exchanges'] for r in rallies]):.1f} pg={stats['policy_loss']:+.3f} "
          f"vf={stats['value_loss']:.3f} ent={stats['entropy']:.3f}", flush=True)


def train(args, env=None, ppo_ctrl=None, game=None):
    """Without an env: her environment, the low-level controller and her v3 functions, in --workers processes
    (tests inject all three and run sequentially)."""
    _seed(args.seed)
    runner = _make_runner(args, env, ppo_ctrl, game)
    policies = {p: RolePolicy() for p in (1, 2)}
    optims = {p: torch.optim.Adam(policies[p].parameters(), lr=args.lr) for p in (1, 2)}
    pools, round_opponent = {1: [], 2: []}, None
    for path in (args.output, args.log):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(args.log, "w", newline="") as log_f:
        log_w = csv.writer(log_f)
        log_w.writerow(LOG_HEADER)
        try:
            for it in range(1, args.iterations + 1):
                t0 = time.time()
                learner = learner_for_iteration(it, args.round_iters)
                other = 3 - learner
                if is_round_start(it, args.round_iters):
                    round_opponent = _frozen(policies[other])
                opponents = choose_opponents(pools[other], round_opponent, args.rallies_per_iter, args.snapshot_prob)
                rallies = runner.run(learner, policies[learner], opponents,
                                     rally_seeds(args.seed, it, args.rallies_per_iter))
                stats = ppo_update(policies[learner], optims[learner],
                                   build_batch(rallies, args.gamma, args.gae_lambda), args)
                if it % args.snapshot_every == 0:
                    pools[learner] = (pools[learner] + [_frozen(policies[learner])])[-args.snapshot_buffer_size:]
                log_w.writerow(_log_row(it, learner, rallies, stats, len(pools[other]), time.time() - t0))
                log_f.flush()
                if it % args.print_every == 0 or it == 1:
                    _print_row(it, args, learner, rallies, stats)
                if it % args.save_every == 0 or it == args.iterations:
                    save_policies(policies, args.output, _meta(args))
        finally:
            runner.close()
    print(f"Done. Saved {args.output}, log {args.log}")


def build_parser():
    ap = argparse.ArgumentParser(description="Per-player, role-gated self-play in Kexin's V3 game")
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--rallies-per-iter", type=int, default=32)
    ap.add_argument("--round-iters", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--entropy-coef", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-epsilon", type=float, default=0.2)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--minibatch-size", type=int, default=64)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--snapshot-prob", type=float, default=0.5)
    ap.add_argument("--snapshot-every", type=int, default=25)
    ap.add_argument("--snapshot-buffer-size", type=int, default=10)
    ap.add_argument("--print-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--steps", type=int, default=HER_MAX_STEPS, help="rally step cap (hers: 600)")
    ap.add_argument("--workers", type=int, default=1, help="processes playing an iteration's rallies")
    ap.add_argument("--skill-profile", default="aggressive")
    ap.add_argument("--reset-mode", default="ready")
    ap.add_argument("--planner", default="legacy")
    ap.add_argument("--gantry-speed-scale", type=float, default=1.0)
    ap.add_argument("--ppo", default=PPO_MODEL_PATH)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--log", required=True)
    return ap


if __name__ == "__main__":
    train(build_parser().parse_args())
