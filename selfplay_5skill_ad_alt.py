#!/usr/bin/env python3
"""Alternating (best-response) attacker/defender self-play.

The alternating counterpart to ``selfplay_5skill_ad.py``.  Everything about the
game is shared with that file -- the same ego-centric encoder, the same 15-way
joint head over (skill, defender position), the same four values written per net
crossing, the same reward -- and it imports ``play_one_rally`` from there so the
rollout cannot drift between the two.  Only who holds the weights changes:

  * Two networks, pi_A and pi_B.
  * Phase A trains pi_A against a frozen pi_B; phase B swaps them.  The trainer
    always plays ego, matching ``selfplay_5skill_alt.py``.
  * Phases alternate every ``--phase-length`` iterations.

This is iterated best response rather than equilibrium-seeking: each phase
solves pi_X ~ argmax E[reward(pi_X, pi_other frozen)].  It is deliberately a
different object from the simultaneous trainer, and reporting both is the point
-- IBR exploits a fixed opponent model while simultaneous self-play is trying
not to be exploitable.  The known failure mode is that IBR converges to
specialists that beat the particular frozen opponent rather than to an
unexploitable mixture, which is what the 2026-07-29 five-skill runs showed.

There is no opponent snapshot pool here, unlike the simultaneous trainer.  The
frozen network *is* the held-fixed opponent, so a pool would blur the
best-response objective rather than stabilise it.

Why this file and not the simultaneous trainer (2026-10-27).  Section 4.2 opens
with "for player i, a neural actor with parameters psi_i" -- psi is indexed by
player, i.e. two parameter sets -- and then "alternate learner updates while
holding the opponent fixed within an update round".  This file is the only
implementation in the project that does either: `policy_A`/`policy_B` with
their own optimizers, and `frozen.eval()` for a whole `--phase-length` block.
Both simultaneous trainers (ours and Angelina's PPO one) share a single network
and update it against itself, which contradicts that first sentence.

What is still missing here, and it is a missing feature rather than a conflict:
the snapshot pool of the next sentence.  A pool is orthogonal to the
alternation -- the opponent for a rally can be drawn from {the currently frozen
network} union {its earlier snapshots}, which is what "earlier strategies as
well as recent opponents" asks for.

What has to come from the PPO side.  The algorithm in 4.2 is PPO with a critic,
a discount, and two heads over Delta_5 x Delta_3; truncated rollouts are
bootstrapped from the critic.  This file inherits REINFORCE with an EMA scalar
baseline, no discount, one shared 15-way joint head, an 84-dim ego-centric
state against the planner's 76 ("the same physical information available to the
potential-based planner" is false for us), and -0.5 to both players on
truncation -- the only place in the project that breaks zero-sum, since the
alpha-potential labelling scores a truncated rally (0, 0).

So the target is one trainer that is this file's structure with that algorithm:
two parameter sets and alternating phases, plus PPO, a critic, two heads, the
90-dim joint state, and the pool.  The plan is to re-apply this phase logic on
top of her ``selfplay_attack_defend.py`` once it is pushed -- the same thin
layer this file is over ``selfplay_5skill_ad.py``, which it imports
``play_one_rally`` from rather than reimplementing.  Neither 4.2 nor any
implementation yet does "only the head corresponding to the active role selects
an action": all three select all four values and execute the role-relevant
half, while 3.2 eq (4) defines the action set as 2.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nash_skills.env_wrapper import SkillEnv
from nash_skills.skills import SKILL_NAMES, N_SKILLS
from nash_skills.defender import DEFENDER_NAMES, N_DEFENDER, split_joint
from nash_skills.v2.state_encoder import SKILL_ENCODINGS, state_dim_for
from selfplay_5skill_ad import (
    MetaPolicy, play_one_rally, set_joint, N_JOINT,
    TRUNCATED_PENALTY, PPO_MODEL_PATH, history_checkpoint_path,
)

from stable_baselines3 import PPO


def train(args):
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dim = state_dim_for(args.skill_encoding)
    env = SkillEnv(proc_id=1, history=4, skill_profile=args.skill_profile)
    ppo = PPO.load(args.ppo_model, device=device)
    policy_A = MetaPolicy(state_dim=dim).to(device)
    policy_B = MetaPolicy(state_dim=dim).to(device)
    opt_A = torch.optim.Adam(policy_A.parameters(), lr=args.lr)
    opt_B = torch.optim.Adam(policy_B.parameters(), lr=args.lr)

    fields = ["iter", "phase", "mean_reward", "reward_std", "pg_loss", "entropy",
              "trunc_rate", "mean_crossings", "wall_time"]
    fields += [f"p_att_{s}" for s in SKILL_NAMES]
    fields += [f"p_def_{d}" for d in DEFENDER_NAMES]
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        csv.writer(fh).writerow(fields)

    print(f"alternating AD self-play: {args.skill_encoding} encoding ({dim}-dim), "
          f"{N_JOINT}-way joint head, phase_length={args.phase_length}, "
          f"{args.rallies} rallies/iter", flush=True)

    t0 = time.time()
    for it in range(1, args.iters + 1):
        phase = "A" if ((it - 1) // args.phase_length) % 2 == 0 else "B"
        trainer, frozen = (policy_A, policy_B) if phase == "A" else (policy_B, policy_A)
        opt = opt_A if phase == "A" else opt_B
        frozen.eval()

        rewards, flat_logp, flat_ents, advs = [], [], [], []
        att_counts = np.zeros(N_SKILLS)
        def_counts = np.zeros(N_DEFENDER)
        n_trunc = 0
        crossings = []
        rollouts = []
        for _ in range(args.rallies):
            out = play_one_rally(
                env, ppo, trainer, frozen, device,
                ego_init_idx=random.randint(0, N_JOINT - 1),
                opp_init_idx=random.randint(0, N_JOINT - 1),
                trunc_penalty=args.trunc_penalty, greedy=False,
                skill_encoding=args.skill_encoding)
            rollouts.append(out)
            rewards.append(out["ego_reward"])
            n_trunc += int(not out["done"])
            crossings.append(out["crossings"])
            # ego_skills[0] is the rally's initial pairing, not a decision
            for j in out["ego_skills"][1:]:
                s, d = split_joint(j, N_SKILLS)
                att_counts[s] += 1
                def_counts[d] += 1

        # Gradients flow only through the trainer, and play_one_rally leaves the
        # opponent's log-probs empty whenever the two policies are distinct
        # objects -- which they always are here -- so ego's are all there is.
        baseline = float(np.mean(rewards)) if rewards else 0.0
        for out in rollouts:
            for lp in out["ego_log_probs"]:
                flat_logp.append(lp)
                advs.append(out["ego_reward"] - baseline)
            flat_ents.extend(out["ego_entropies"])
        if not flat_logp:
            continue

        logp_t = torch.stack(flat_logp)
        advs_t = torch.tensor(advs, dtype=torch.float32, device=device)
        ent_t = torch.stack(flat_ents).mean()
        pg_loss = -(logp_t * advs_t).mean()
        loss = pg_loss - args.entropy_coef * ent_t

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.parameters(), args.max_grad_norm)
        opt.step()

        p_att = att_counts / max(att_counts.sum(), 1)
        p_def = def_counts / max(def_counts.sum(), 1)
        row = [it, phase, float(np.mean(rewards)), float(np.std(rewards)),
               float(pg_loss.item()), float(ent_t.item()),
               n_trunc / max(len(rollouts), 1), float(np.mean(crossings)),
               round(time.time() - t0, 1)]
        row += [round(float(v), 4) for v in p_att]
        row += [round(float(v), 4) for v in p_def]
        with open(csv_path, "a", newline="") as fh:
            csv.writer(fh).writerow(row)

        if it % args.log_every == 0 or it == 1:
            print(f"iter {it:4d} [{phase}]  reward {np.mean(rewards):+.3f}  "
                  f"ent {ent_t.item():.3f}  trunc {n_trunc/len(rollouts):.1%}  "
                  f"att {np.round(p_att,2)}  def {np.round(p_def,2)}", flush=True)
        if args.save_history and it % args.save_history == 0:
            for tag, pol in (("A", policy_A), ("B", policy_B)):
                torch.save(pol.state_dict(),
                           history_checkpoint_path(f"{args.output_prefix}_{tag}.pth", it))

    for tag, pol in (("A", policy_A), ("B", policy_B)):
        path = f"{args.output_prefix}_{tag}.pth"
        torch.save(pol.state_dict(), path)
        print(f"wrote {path}  ({args.skill_encoding}, {dim}-dim)")
    env.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--phase-length", type=int, default=50)
    ap.add_argument("--rallies", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy-coef", type=float, default=0.05)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--trunc-penalty", type=float, default=TRUNCATED_PENALTY)
    ap.add_argument("--skill-encoding", choices=SKILL_ENCODINGS, default="onehot")
    ap.add_argument("--skill-profile", default="aggressive")
    ap.add_argument("--ppo-model", default=PPO_MODEL_PATH)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters-log", dest="log_every", type=int, default=25)
    ap.add_argument("--save-history", type=int, default=50)
    ap.add_argument("--output-prefix", default="models/selfplay_ad_alt")
    ap.add_argument("--csv", default="logs/selfplay_ad_alt.csv")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
