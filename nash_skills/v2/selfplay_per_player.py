"""
Design B self-play as in the paper (Sec. 4.2): one actor-critic psi_i per
player, decisions at racket contacts with only the active role's head
acting (nash_skills.v2.selfplay_rollout), PPO credit on exactly that head,
and alternating update rounds: player 1 learns for --round-iters iterations
against frozen opponents from player 2's pool, then player 2 learns against
player 1's pool, and so on. Within a round the opponent never changes: the
pool holds the opponent's frozen copy from the round start plus its earlier
snapshots, one sampled per rally and kept for the whole rally. Each player
always plays its own side, so both sides are trained equally; serve
direction (initial role) is random at every reset.

Supersedes nash_skills/v2/selfplay_attack_defend.py (one shared actor,
learner always player 1, both heads resampled and credited at every net
crossing), which is kept for its existing checkpoints.

    MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONPATH=. python nash_skills/v2/selfplay_per_player.py \\
        --iterations 1000 --rallies-per-iter 32 --round-iters 25 \\
        --defend-mechanism preparation_cal --skill-profile moderate_aggressive \\
        --reset-targets --seed 0 --output runs/sp/seed0.pth --log runs/sp/seed0.csv
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import csv
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from nash_skills.decision_clock import DECISION_TIMINGS
from nash_skills.defend_mechanisms import DEFEND_MECHANISMS
from nash_skills.skills import SKILL_PROFILE_NAMES
from nash_skills.v2.selfplay_attack_defend import MetaPolicy, compute_gae, ppo_clipped_loss
from nash_skills.v2.state_encoder_attack_defend import ENCODER_DIMS, STATE_ENCODER_NAME
from nash_skills.v2.selfplay_rollout import (
    DEFAULT_SELFPLAY_TIMING, STATE_MODES, is_round_start, learner_for_iteration, play_rally,
    role_head_log_prob_entropy, save_per_player, state_dim_for,
)

PPO_MODEL_PATH = "logs/best_model_tracker1/best_model"
MAX_STEPS_PER_RALLY = 2500
HISTORY = 4


def make_env(args):
    from nash_skills.env_wrapper import SkillEnv
    return SkillEnv(proc_id=1, history=HISTORY, reset_mode="ready", skill_profile=args.skill_profile,
                    reset_targets=args.reset_targets)


def load_ppo(path):
    from stable_baselines3 import PPO
    return PPO.load(path, device="cpu")


def build_batch(rallies, gamma, gae_lambda):
    batch = {k: [] for k in ("states", "roles", "actions", "old_log_probs", "advantages", "returns")}
    for r in rallies:
        n = len(r["old_values"])
        if n == 0:
            continue
        rewards = [0.0] * (n - 1) + [r["reward"]]
        adv, ret = compute_gae(rewards, r["old_values"], gamma, gae_lambda, bootstrap_value=r["bootstrap_value"])
        for key in ("states", "roles", "actions", "old_log_probs"):
            batch[key] += list(r[key])
        batch["advantages"] += adv
        batch["returns"] += ret
    return batch


def ppo_update(policy, optim, batch, args):
    n = len(batch["advantages"])
    if n == 0:
        return {"policy_loss": float("nan"), "value_loss": float("nan"), "entropy": float("nan")}
    states = torch.as_tensor(np.stack(batch["states"]), dtype=torch.float32)
    actions = torch.tensor(batch["actions"], dtype=torch.long)
    is_attack = torch.tensor([r == "attack" for r in batch["roles"]])
    old_lp = torch.tensor(batch["old_log_probs"], dtype=torch.float32)
    returns = torch.tensor(batch["returns"], dtype=torch.float32)
    adv = torch.tensor(batch["advantages"], dtype=torch.float32)
    if n > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = {"policy_loss": [], "value_loss": [], "entropy": []}
    for _ in range(args.ppo_epochs):
        perm = torch.randperm(n)
        for start in range(0, n, args.minibatch_size):
            idx = perm[start:start + args.minibatch_size]
            attack_logits, defend_logits, values = policy(states[idx])
            lp, ent = role_head_log_prob_entropy(attack_logits, defend_logits, actions[idx], is_attack[idx])
            policy_loss = ppo_clipped_loss(lp, old_lp[idx], adv[idx], args.clip_epsilon)
            value_loss = F.mse_loss(values, returns[idx])
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * ent.mean()
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()
            for k, v in (("policy_loss", policy_loss), ("value_loss", value_loss), ("entropy", ent.mean())):
                stats[k].append(float(v.item()))
    return {k: float(np.mean(v)) for k, v in stats.items()}


def _frozen(policy):
    snap = deepcopy(policy).eval()
    for p in snap.parameters():
        p.requires_grad_(False)
    return snap


def _log_header():
    return ["iter", "learner", "win_rate", "trunc_rate", "mean_exchanges", "mean_steps", "n_decisions",
            "attack_usage", "defend_usage", "policy_loss", "value_loss", "entropy", "pool_size", "wall_time"]


def _seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    return np.random.default_rng(seed)


def train(args):
    rng = _seed(args.seed)
    device = torch.device("cpu")
    env, ppo = make_env(args), load_ppo(args.ppo)
    policies = {p: MetaPolicy(state_dim=state_dim_for(args.state_mode, args.encoder)) for p in (1, 2)}
    optims = {p: torch.optim.Adam(policies[p].parameters(), lr=args.lr) for p in (1, 2)}
    pools, round_opponent = {1: [], 2: []}, None
    meta = {"skill_profile": args.skill_profile, "defend_mechanism": args.defend_mechanism,
            "gamma": args.gamma, "ppo": args.ppo, "iterations": args.iterations,
            "rallies_per_iter": args.rallies_per_iter, "round_iters": args.round_iters,
            "state_mode": args.state_mode, "encoder": args.encoder, "decision_timing": args.decision_timing}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    with open(args.log, "w", newline="") as log_f:
        log_w = csv.writer(log_f)
        log_w.writerow(_log_header())
        try:
            for it in range(1, args.iterations + 1):
                t0 = time.time()
                learner = learner_for_iteration(it, args.round_iters)
                other = 3 - learner
                if is_round_start(it, args.round_iters):
                    round_opponent = _frozen(policies[other])
                rallies = []
                for _ in range(args.rallies_per_iter):
                    use_pool = pools[other] and random.random() < args.snapshot_prob
                    opp = random.choice(pools[other]) if use_pool else round_opponent
                    rallies.append(play_rally(env, ppo, {learner: policies[learner], other: opp}, learner,
                                              device, args.defend_mechanism, MAX_STEPS_PER_RALLY, rng,
                                              state_mode=args.state_mode, encoder=args.encoder,
                                              decision_timing=args.decision_timing))
                stats = ppo_update(policies[learner], optims[learner],
                                   build_batch(rallies, args.gamma, args.gae_lambda), args)
                if it % args.snapshot_every == 0:
                    pools[learner].append(_frozen(policies[learner]))
                    pools[learner] = pools[learner][-args.snapshot_buffer_size:]
                _log_iteration(log_w, it, learner, rallies, stats, len(pools[other]), time.time() - t0)
                log_f.flush()
                if it % args.print_every == 0 or it == 1:
                    _print_iteration(it, args.iterations, learner, rallies, stats)
                if it % args.save_every == 0 or it == args.iterations:
                    save_per_player(policies, args.output, meta)
        finally:
            env.close()
    print(f"Done. Saved {args.output}, log {args.log}")


def _log_iteration(log_w, it, learner, rallies, stats, pool_size, wall):
    n = len(rallies)
    attack = sum(r["usage"]["attack"] for r in rallies)
    defend = sum(r["usage"]["defend"] for r in rallies)
    log_w.writerow([it, learner, sum(r["done"] and r["reward"] > 0 for r in rallies) / n,
                    sum(not r["done"] for r in rallies) / n, np.mean([r["exchanges"] for r in rallies]),
                    np.mean([r["steps"] for r in rallies]), sum(len(r["actions"]) for r in rallies),
                    " ".join(map(str, attack)), " ".join(map(str, defend)),
                    stats["policy_loss"], stats["value_loss"], stats["entropy"], pool_size, wall])


def _print_iteration(it, total, learner, rallies, stats):
    n = len(rallies)
    wins = sum(r["done"] and r["reward"] > 0 for r in rallies) / n
    print(f"[iter {it:4d}/{total}] learner=P{learner} win={wins:.2f} "
          f"exch={np.mean([r['exchanges'] for r in rallies]):.1f} pg={stats['policy_loss']:+.3f} "
          f"vf={stats['value_loss']:.3f} ent={stats['entropy']:.3f}", flush=True)


def build_parser():
    ap = argparse.ArgumentParser(description="Per-player, role-gated Design B self-play (paper Sec 4.2)")
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--rallies-per-iter", type=int, default=32)
    ap.add_argument("--round-iters", type=int, default=25, help="iterations per alternating update round")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--entropy-coef", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-epsilon", type=float, default=0.2)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--minibatch-size", type=int, default=64)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--snapshot-prob", type=float, default=0.5,
                    help="probability a rally's opponent is an older pool snapshot instead of the round's")
    ap.add_argument("--snapshot-every", type=int, default=25)
    ap.add_argument("--snapshot-buffer-size", type=int, default=10)
    ap.add_argument("--print-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--output", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ppo", default=PPO_MODEL_PATH)
    ap.add_argument("--skill-profile", choices=SKILL_PROFILE_NAMES, default="aggressive")
    ap.add_argument("--defend-mechanism", choices=DEFEND_MECHANISMS, default="preparation_cal")
    ap.add_argument("--reset-targets", action="store_true")
    ap.add_argument("--state-mode", choices=STATE_MODES, default="joint",
                    help="actor input: 'joint' (with both players' current choices) or 'no_choices' "
                         "(without them, the paper's x = (ball, robots, role))")
    ap.add_argument("--encoder", choices=sorted(ENCODER_DIMS), default=STATE_ENCODER_NAME,
                    help="state encoder; attack_defend_joint_v3 adds the ball's spin and the rally status")
    ap.add_argument("--decision-timing", choices=DECISION_TIMINGS, default=DEFAULT_SELFPLAY_TIMING,
                    help="when both players decide: each racket contact (paper Sec. 3.2) or each net crossing")
    return ap


if __name__ == "__main__":
    train(build_parser().parse_args())
