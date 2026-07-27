"""
Quick eval for the 5-skill self-play meta-policy.

Loads models/selfplay_5skill.pth and plays N rallies against each fixed-skill
baseline (random + 5 fixed skills).

Reports per opponent:
  - Real win rate  = wins / (wins + losses)
  - Truncation rate
  - Avg crossings per rally
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
    ego_skill_count = [0] * N_SKILLS

    for _ in range(n_rallies):
        ego_init = rng.randint(0, N_SKILLS - 1)
        opp_init = rng.randint(0, N_SKILLS - 1)
        env.set_skills(skill_from_index(ego_init), skill_from_index(opp_init))
        obs, info = env.reset()
        prev_ball_x = float(obs[36])
        crossings = 0
        steps = 0
        done = False

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
        if done:
            winner = infer_terminal_winner(obs, info, fallback="position") or "opp"
            if winner == "ego":
                wins += 1
            else:
                losses += 1
        else:
            trunc += 1

    n_done = wins + losses
    real_wr = wins / n_done if n_done > 0 else float("nan")
    total_skills = max(sum(ego_skill_count), 1)
    skill_usage = dict(zip(SKILL_NAMES, ego_skill_count))
    return {
        "wins": wins, "losses": losses, "trunc": trunc,
        "real_wr": real_wr,
        "trunc_rate": trunc / n_rallies,
        "avg_xs": float(np.mean(crossings_list)),
        "skill_pct": [c / total_skills for c in ego_skill_count],
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

    env = SkillEnv(proc_id=1, history=HISTORY)

    print(f"\nEvaluating {args.rallies} rallies vs each opponent\n" + "=" * 100)
    skill_short = [s[:5] for s in SKILL_NAMES]
    label_w = max(12, max(len(lbl) for lbl, _ in opponents) + 2)
    header = (f"{'opp':<{label_w}} {'real_wr':>8} {'wins':>5} {'loss':>5} {'trunc':>6} "
              f"{'trunc%':>7} {'avg_xs':>7}  ego skill %: "
              + " ".join(f"{s:>6}" for s in skill_short))
    print(header)
    print("-" * len(header))

    all_results = []
    for label, pick_fn in opponents:
        r = eval_one(env, ppo, policy, pick_fn, device, args.rallies, rng)
        all_results.append((label, r))
        skill_str = " ".join(f"{p:>5.1%}" for p in r["skill_pct"])
        print(f"{label:<{label_w}} {r['real_wr']:>8.1%} "
              f"{r['wins']:>5d} {r['losses']:>5d} {r['trunc']:>6d} "
              f"{r['trunc_rate']:>7.1%} {r['avg_xs']:>7.1f}  "
              f"             {skill_str}")

    print("-" * len(header))

    print("\nFull scorecards (nash_skills/v2/scorecard.py)")
    print("=" * len(header))
    for label, r in all_results:
        print(format_scorecard(r["scorecard"], label=f"vs {label}"))
        print()

    env.close()


if __name__ == "__main__":
    main()
