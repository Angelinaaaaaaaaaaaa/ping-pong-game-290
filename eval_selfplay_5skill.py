"""
Quick eval for the 5-skill self-play meta-policy.

Loads models/selfplay_5skill.pth and plays N rallies against each fixed-skill
baseline (random + 5 fixed skills).

Reports per opponent:
  - Real win rate  = wins / (wins + losses)
  - Truncation rate
  - Avg crossings per rally
  - Avg rally length in env steps (all rallies + decisive-only)
  - Ego skill usage distribution

Run:
    python eval_selfplay_5skill.py
    python eval_selfplay_5skill.py --checkpoint models/selfplay_5skill.pth --rallies 30
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import io
import random
from contextlib import redirect_stdout

import numpy as np
import torch
from stable_baselines3 import PPO

from nash_skills.env_wrapper import SkillEnv
from nash_skills.skills import SKILL_NAMES, N_SKILLS, skill_from_index
from nash_skills.v2.state_encoder import encode_ego, encode_opp
from nash_skills.v2.scorecard import compute_scorecard, format_scorecard
from nash_skills.winner_inference import infer_terminal_winner
from selfplay_5skill import (
    MetaPolicy, _build_ppo_obs,
    HISTORY, TABLE_SHIFT, MAX_STEPS_PER_RALLY, PPO_MODEL_PATH,
)


def _swallow_step(env, action):
    buf = io.StringIO()
    with redirect_stdout(buf):
        return env.step(action)


def pick_baseline(name, rng):
    """Returns pick_fn(opp_state, obs, info, ego_idx) -> skill_idx.

    Extra args (obs, info, ego_idx) are ignored by simple baselines but used by
    nash-p-hard/ibr-q wrappers built via make_picker.
    """
    if name == "random":
        return lambda s, o, i, e: rng.randint(0, N_SKILLS - 1)
    if name in SKILL_NAMES:
        idx = SKILL_NAMES.index(name)
        return lambda s, o, i, e: idx
    raise ValueError(f"Unknown baseline {name}")


def eval_one(env, ppo, ego_policy, opp_pick_fn, device, n_rallies, rng):
    wins = 0; losses = 0; trunc = 0
    crossings_list = []
    steps_list_all = []      # every rally (done + trunc)
    steps_list_done = []     # only decisive rallies
    ego_skill_count = [0] * N_SKILLS

    # Per-(ego_init, opp_init) pair breakdown to diagnose whether high WR
    # comes from a few short/trunc rallies vs a robust decisive win pattern.
    from collections import defaultdict
    pair_stats = defaultdict(lambda: {"n": 0, "wins": 0, "losses": 0, "trunc": 0,
                                      "steps_all": [], "steps_done": []})

    for _ in range(n_rallies):
        ego_init = rng.randint(0, N_SKILLS - 1)
        opp_init = rng.randint(0, N_SKILLS - 1)
        env.set_skills(skill_from_index(ego_init), skill_from_index(opp_init))
        obs, info = env.reset()
        prev_ball_x = float(obs[36])
        crossings = 0
        steps = 0
        done = False
        pair_key = (ego_init, opp_init)

        while True:
            ppo1 = _build_ppo_obs(obs, info, 1)
            ppo2 = _build_ppo_obs(obs, info, 2)
            a1, _ = ppo.predict(ppo1, deterministic=True)
            a2, _ = ppo.predict(ppo2, deterministic=True)
            action = np.zeros(18, dtype=np.float32)
            action[:9] = a1[:9]; action[9:] = a2[:9]

            obs, _r, done, _t, info = _swallow_step(env, action)
            steps += 1
            curr_ball_x = float(obs[36])

            if (prev_ball_x - TABLE_SHIFT) * (curr_ball_x - TABLE_SHIFT) < 0:
                crossings += 1
                ego_state = encode_ego(obs, info)
                opp_state = encode_opp(obs, info)
                with torch.no_grad():
                    ego_idx, _ = ego_policy.sample(ego_state, device)
                opp_idx = opp_pick_fn(opp_state, obs, info, ego_idx)
                ego_skill_count[ego_idx] += 1
                env.set_skills(skill_from_index(ego_idx), skill_from_index(opp_idx))

            prev_ball_x = curr_ball_x

            if done or steps >= MAX_STEPS_PER_RALLY:
                break

        crossings_list.append(crossings)
        steps_list_all.append(steps)
        pair_stats[pair_key]["n"] += 1
        pair_stats[pair_key]["steps_all"].append(steps)
        if done:
            winner = infer_terminal_winner(obs, info, fallback="position") or "opp"
            if winner == "ego":
                wins += 1
                pair_stats[pair_key]["wins"] += 1
            else:
                losses += 1
                pair_stats[pair_key]["losses"] += 1
            steps_list_done.append(steps)
            pair_stats[pair_key]["steps_done"].append(steps)
        else:
            trunc += 1
            pair_stats[pair_key]["trunc"] += 1

    n_done = wins + losses
    real_wr = wins / n_done if n_done > 0 else float("nan")
    total_skills = max(sum(ego_skill_count), 1)
    skill_usage = dict(zip(SKILL_NAMES, ego_skill_count))
    return {
        "wins": wins, "losses": losses, "trunc": trunc,
        "real_wr": real_wr,
        "trunc_rate": trunc / n_rallies,
        "avg_xs": float(np.mean(crossings_list)),
        "avg_steps_all": float(np.mean(steps_list_all)),
        "avg_steps_done": float(np.mean(steps_list_done)) if steps_list_done else float("nan"),
        "skill_pct": [c / total_skills for c in ego_skill_count],
        "pair_stats": dict(pair_stats),
        # Shared scorecard (nash_skills/v2/scorecard.py, ported from main,
        # meeting note item 19): adds median rally length, skill-usage
        # entropy, and dominant-skill fraction on top of the fields above.
        "scorecard": compute_scorecard(
            wins=wins, losses=losses, truncated=trunc,
            rally_lengths=crossings_list, skill_usage=skill_usage,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/selfplay_5skill.pth")
    parser.add_argument("--rallies", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baselines", nargs="+",
                        default=["random"] + list(SKILL_NAMES))
    parser.add_argument("--vs-checkpoint", default=None,
                        help="Path to another self-play policy checkpoint. "
                             "If provided, ego (--checkpoint) plays against this opp. "
                             "Overrides --baselines.")
    parser.add_argument("--vs-label", default=None,
                        help="Display label for --vs-checkpoint opp (defaults to filename stem)")
    parser.add_argument("--vs-strategy", default=None,
                        choices=["nash-p-hard", "nash-p-br", "nash-p-minimax",
                                 "nash-p-adaptive", "ibr", "ibr-q"],
                        help="Use a Phi/Q-based strategy from eval_matchup as opp.")
    parser.add_argument("--phi-model", default="models/model_p_5skill_v3.pth",
                        help="Path to potential Phi model (for Phi-based strategies)")
    parser.add_argument("--q1-model", default="models/model1_5skill_v3.pth",
                        help="Path to Q1 model (for ibr/ibr-q)")
    parser.add_argument("--q2-model", default="models/model2_5skill_v3.pth",
                        help="Path to Q2 model (for ibr/ibr-q)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    print(f"Loading ego policy from {args.checkpoint} ...")
    policy = MetaPolicy().to(device)
    policy.load_state_dict(torch.load(args.checkpoint, map_location=device))
    policy.eval()

    # Build the list of (label, pick_fn) opponents
    opponents = []
    if args.vs_checkpoint:
        print(f"Loading opp policy from {args.vs_checkpoint} ...")
        opp_policy = MetaPolicy().to(device)
        opp_policy.load_state_dict(torch.load(args.vs_checkpoint, map_location=device))
        opp_policy.eval()
        label = args.vs_label or os.path.splitext(os.path.basename(args.vs_checkpoint))[0]

        def opp_pick(state_np, _obs, _info, _ego_idx, _opp_policy=opp_policy):
            with torch.no_grad():
                idx, _ = _opp_policy.sample(state_np, device)
            return idx
        opponents.append((label, opp_pick))
    elif args.vs_strategy:
        from model_arch import SimpleModel
        from nash_skills.eval_matchup import make_picker

        print(f"Loading Phi from {args.phi_model}")
        phi = SimpleModel(76, [64, 32, 16], 1, last_layer_activation=None)
        phi.load_state_dict(torch.load(args.phi_model, weights_only=True, map_location="cpu"))
        phi.eval()

        m1 = m2 = None
        if args.vs_strategy in ("ibr", "ibr-q"):
            print(f"Loading Q1 from {args.q1_model}")
            m1 = SimpleModel(76, [64, 32, 16], 1)
            m1.load_state_dict(torch.load(args.q1_model, weights_only=True, map_location="cpu"))
            m1.eval()
            print(f"Loading Q2 from {args.q2_model}")
            m2 = SimpleModel(76, [64, 32, 16], 1)
            m2.load_state_dict(torch.load(args.q2_model, weights_only=True, map_location="cpu"))
            m2.eval()

        def _state_enc(obs, info, player):
            return encode_ego(obs, info) if player == 1 else encode_opp(obs, info)

        nash_picker = make_picker(args.vs_strategy, model_p=phi,
                                   state_encoder_fn=_state_enc,
                                   model1=m1, model2=m2)

        def opp_pick(_state_np, obs, info, ego_idx, _picker=nash_picker):
            # Opp is player 2; ego_idx is the "other_skill_idx" from opp's view.
            return int(_picker(2, obs, ego_idx, info))

        opponents.append((args.vs_strategy, opp_pick))
    else:
        for opp_name in args.baselines:
            opponents.append((opp_name, pick_baseline(opp_name, rng)))

    print(f"Loading PPO from {PPO_MODEL_PATH} (CPU) ...")
    ppo = PPO.load(PPO_MODEL_PATH, device="cpu")

    env = SkillEnv(proc_id=1, history=HISTORY, skill_profile="aggressive")

    print(f"\nEvaluating {args.rallies} rallies vs each opponent\n" + "=" * 100)
    # Full names, not truncated: SKILL_NAMES has two 4-char-prefix collisions
    # (left/left_short, right_short/right) that made earlier abbreviated
    # columns genuinely ambiguous -- e.g. "right" could mean either skill.
    skill_col_w = max(len(s) for s in SKILL_NAMES) + 1
    label_w = max(12, max(len(lbl) for lbl, _ in opponents) + 2)
    header = (f"{'opp':<{label_w}} {'real_wr':>8} {'wins':>5} {'loss':>5} {'trunc':>6} "
              f"{'trunc%':>7} {'avg_xs':>7} {'stp_all':>8} {'stp_done':>8}  "
              f"ego skill %: "
              + " ".join(f"{s:>{skill_col_w}}" for s in SKILL_NAMES))
    print(header)
    print("-" * len(header))

    all_results = []
    for label, pick_fn in opponents:
        r = eval_one(env, ppo, policy, pick_fn, device, args.rallies, rng)
        skill_str = " ".join(f"{p:>{skill_col_w}.1%}" for p in r["skill_pct"])
        stp_done = f"{r['avg_steps_done']:>8.1f}" if r['avg_steps_done'] == r['avg_steps_done'] else f"{'nan':>8}"
        print(f"{label:<{label_w}} {r['real_wr']:>8.1%} "
              f"{r['wins']:>5d} {r['losses']:>5d} {r['trunc']:>6d} "
              f"{r['trunc_rate']:>7.1%} {r['avg_xs']:>7.1f} "
              f"{r['avg_steps_all']:>8.1f} {stp_done}  "
              f"           {skill_str}")
        all_results.append((label, r))

    print("-" * len(header))

    print("\nFull scorecards (nash_skills/v2/scorecard.py)")
    print("=" * len(header))
    for label, r in all_results:
        print(format_scorecard(r["scorecard"], label=f"vs {label}"))
        print()

    # ------------------------------------------------------------------ #
    # Per-(init ego, init opp) pair breakdown: sanity-check that high WR
    # is not inflated by a few short/truncated rallies.
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 100)
    print("  PER-INIT-PAIR BREAKDOWN")
    print("  (grouped by initial skill pair sampled at rally start)")
    print("=" * 100)
    for label, r in all_results:
        print(f"\n--- opponent: {label} ---")
        print(f"  {'ego_init':<12} {'opp_init':<12} {'n':>4} "
              f"{'wins':>5} {'loss':>5} {'trunc':>5} {'wr':>7} "
              f"{'stp_all':>8} {'stp_done':>8}")
        for (ei, oi), s in sorted(r["pair_stats"].items()):
            n_done = s["wins"] + s["losses"]
            wr = s["wins"] / n_done if n_done > 0 else float("nan")
            wr_str = f"{wr:>6.1%}" if n_done > 0 else "  ---"
            stp_all = float(np.mean(s["steps_all"])) if s["steps_all"] else float("nan")
            stp_done = float(np.mean(s["steps_done"])) if s["steps_done"] else float("nan")
            stp_done_str = f"{stp_done:>8.1f}" if s["steps_done"] else f"{'---':>8}"
            print(f"  {SKILL_NAMES[ei]:<12} {SKILL_NAMES[oi]:<12} {s['n']:>4d} "
                  f"{s['wins']:>5d} {s['losses']:>5d} {s['trunc']:>5d} {wr_str} "
                  f"{stp_all:>8.1f} {stp_done_str}")
    env.close()


if __name__ == "__main__":
    main()
