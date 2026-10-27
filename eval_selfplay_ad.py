"""Evaluate an attacker/defender policy (5 skills x 3 defender positions).

Baselines are deliberately one-sided so each half of the joint action can be
read separately:

  random            uniform over all 15 joint actions
  <skill>           that skill, paired with the centre defender position
  <def_*>           that defender position, paired with a uniformly random skill

A head-to-head between two checkpoints is available through --vs-checkpoint,
which is the sharpest test of whether mixing is worth anything here: a
deterministic policy that is exploitable should lose to one that mixes, even
when the two score the same against fixed opponents.
"""


import sys
import os
from pathlib import Path
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
from nash_skills.defender import (
    DEFENDER_NAMES, N_DEFENDER, defender_from_index, join_index, joint_size,
    split_joint,
)
from nash_skills.v2.state_encoder import (
    encode_ego, encode_opp, state_dim_for, skill_encoding_from_state_dict,
)
from nash_skills.v2.scorecard import compute_scorecard, format_scorecard
from nash_skills.winner_inference import infer_terminal_winner
from selfplay_5skill_ad import (
    MetaPolicy, _build_ppo_obs,
    HISTORY, TABLE_SHIFT, MAX_STEPS_PER_RALLY, PPO_MODEL_PATH,
)


def _swallow_step(env, action):
    buf = io.StringIO()
    with redirect_stdout(buf):
        return env.step(action)



# --- where ego's shots actually came down ------------------------------------
# The scorecard only knows who won; it cannot say whether a loss was a ball off
# the side, a ball past the baseline, or one that never cleared the net. Those
# fail for different reasons and want different fixes, so track them here.
_TABLE_Z = 0.56
_BALL_R = 0.02
_LAND_Z = _TABLE_Z + _BALL_R
# Table geometry, taken from the env rather than restated, so the in/out test
# here and the one the rally logic uses cannot drift apart.
from mujoco_env_comp import TABLE_HALF_WIDTH as _HALF_W
_TABLE_LEN = 1.37          # net -> back edge
_NET_X = TABLE_SHIFT
_BACK_X = TABLE_SHIFT + _TABLE_LEN


def _landings(traj):
    """(x, y, vx, hit_table) for each descending ball, table there or not.

    bounce_history only records contacts, so it is blind to exactly the shots we
    care about. Descend below z=0.70 and either bottom out (a bounce -> take the
    minimum) or keep falling (a miss -> interpolate at table height).
    """
    out, n, i = [], len(traj), 1
    while i < n:
        if not (traj[i - 1][2] >= 0.70 > traj[i][2]):
            i += 1
            continue
        j = i
        while j + 1 < n and 0.45 < traj[j + 1][2] < 0.70:
            j += 1
        window = traj[i:j + 1]
        if not window:
            i += 1
            continue
        if j + 1 < n and traj[j + 1][2] <= 0.45:
            a, b = traj[j], traj[j + 1]
            f = min(max((a[2] - _LAND_Z) / max(a[2] - b[2], 1e-9), 0.0), 1.0)
            out.append((a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[3], False))
        else:
            k = min(range(len(window)), key=lambda t: window[t][2])
            out.append((window[k][0], window[k][1], window[k][3], True))
        i = j + 2
    return out


def classify_landings(traj):
    """Count ego's descending shots by where they came down.

    vx > 0 means the ball was travelling towards the opponent, i.e. it was ego's
    shot. `net` means it came down on our own side, so it never crossed.
    """
    c = {"shots": 0, "in": 0, "wide": 0, "long": 0, "net": 0}
    for x, y, vx, _hit in _landings(traj):
        if vx <= 0:
            continue
        c["shots"] += 1
        if x < _NET_X:
            c["net"] += 1
        elif abs(y) > _HALF_W:
            c["wide"] += 1
        elif x > _BACK_X:
            c["long"] += 1
        else:
            c["in"] += 1
    return c

N_JOINT = joint_size(N_SKILLS)
JOINT_NAMES = [f"{skill_from_index(k)}+{defender_from_index(d)[4:]}"
               for d in range(N_DEFENDER) for k in range(N_SKILLS)]
_CENTER = DEFENDER_NAMES.index("def_center")


def set_joint(env, ego_joint, opp_joint):
    e_s, e_d = split_joint(ego_joint, N_SKILLS)
    o_s, o_d = split_joint(opp_joint, N_SKILLS)
    env.set_skills(skill_from_index(e_s), skill_from_index(o_s))
    env.set_defenders(defender_from_index(e_d), defender_from_index(o_d))


def pick_baseline(name, rng):
    """Returns pick_fn(opp_state, obs, info, ego_idx) -> skill_idx.

    Extra args (obs, info, ego_idx) are ignored by simple baselines but used by
    nash-p-hard/ibr-q wrappers built via make_picker.
    """
    if name == "random":
        return lambda s, o, i, e: rng.randint(0, N_JOINT - 1)
    if name in SKILL_NAMES:
        # a fixed skill still has to stand somewhere; pin it to centre so the
        # baseline varies only in the half it is named for
        idx = join_index(SKILL_NAMES.index(name), _CENTER, N_SKILLS)
        return lambda s, o, i, e: idx
    if name in DEFENDER_NAMES:
        # the mirror image: fixed position, skill drawn uniformly
        d = DEFENDER_NAMES.index(name)
        return lambda s, o, i, e: join_index(rng.randint(0, N_SKILLS - 1), d, N_SKILLS)
    raise ValueError(f"Unknown baseline {name}")


def eval_one(env, ppo, ego_policy, opp_pick_fn, device, n_rallies, rng,
             ego_encoding="onehot", opp_encoding="onehot",
             decided_target=None, max_rallies=None, pool_sides=False):
    """Evaluate `ego_policy` against `opp_pick_fn`.

    Two options exist to match the team's Table 1 protocol ("200 decided rallies
    per opponent; both sides pooled"):

    `decided_target` plays until that many rallies have produced a point rather
    than a fixed total, so truncations shrink the compute spent and not the
    sample.  `pool_sides` alternates which physical player ego is, so any
    left/right asymmetry in the environment cancels instead of loading onto one
    seat -- `set_joint` otherwise wires ego to player 1 permanently.  Both
    default off so existing numbers stay reproducible.
    """
    wins = 0; losses = 0; trunc = 0
    crossings_list = []
    steps_list_all = []      # every rally (done + trunc)
    steps_list_done = []     # only decisive rallies
    ego_skill_count = [0] * N_JOINT
    land = {"shots": 0, "in": 0, "wide": 0, "long": 0, "net": 0}
    # Why each rally ended, keyed by the skill ego played last. Dvij's question:
    # is short winning because the ball is unreturnable, or because the opponent
    # returns it and then errs? Those call for different fixes.
    from collections import Counter
    reason_by_skill = {i: Counter() for i in range(N_JOINT)}

    # Per-(ego_init, opp_init) pair breakdown to diagnose whether high WR
    # comes from a few short/trunc rallies vs a robust decisive win pattern.
    from collections import defaultdict
    pair_stats = defaultdict(lambda: {"n": 0, "wins": 0, "losses": 0, "trunc": 0,
                                      "steps_all": [], "steps_done": []})

    cap = max_rallies if max_rallies else (
        n_rallies if decided_target is None else 20 * decided_target)
    attempted = 0
    while True:
        if decided_target is None:
            if attempted >= n_rallies:
                break
        elif (wins + losses) >= decided_target or attempted >= cap:
            break
        attempted += 1
        # Which physical player ego is this rally.  set_joint takes (ego, opp)
        # and maps the first to player 1, so swapping the arguments is what
        # moves ego to the other seat; the encoders and the winner test below
        # have to follow.
        ego_player = 2 if (pool_sides and attempted % 2 == 0) else 1
        ego_init = rng.randint(0, N_JOINT - 1)
        opp_init = rng.randint(0, N_JOINT - 1)
        if ego_player == 1:
            set_joint(env, ego_init, opp_init)
        else:
            set_joint(env, opp_init, ego_init)
        obs, info = env.reset()
        prev_ball_x = float(obs[36])
        traj = [(float(obs[36]), float(obs[37]), float(obs[38]), float(obs[39]))]
        ego_skills_this_rally = []
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
            traj.append((float(obs[36]), float(obs[37]), float(obs[38]), float(obs[39])))
            curr_ball_x = float(obs[36])

            if (prev_ball_x - TABLE_SHIFT) * (curr_ball_x - TABLE_SHIFT) < 0:
                crossings += 1
                p1_state = encode_ego(obs, info, ego_encoding)
                p2_state = encode_opp(obs, info, opp_encoding)
                ego_state, opp_state = ((p1_state, p2_state) if ego_player == 1
                                        else (p2_state, p1_state))
                with torch.no_grad():
                    ego_idx, _ = ego_policy.sample(ego_state, device)
                opp_idx = opp_pick_fn(opp_state, obs, info, ego_idx)
                ego_skill_count[ego_idx] += 1
                ego_skills_this_rally.append(ego_idx)
                if ego_player == 1:
                    set_joint(env, ego_idx, opp_idx)
                else:
                    set_joint(env, opp_idx, ego_idx)

            prev_ball_x = curr_ball_x

            if done or steps >= MAX_STEPS_PER_RALLY:
                break

        crossings_list.append(crossings)
        steps_list_all.append(steps)
        for _k, _v in classify_landings(traj).items():
            land[_k] += _v
        if ego_skills_this_rally:
            reason_by_skill[ego_skills_this_rally[-1]][
                f"{info.get('termination_reason', 'unknown')}|{info.get('winner', '?')}"] += 1
        pair_stats[pair_key]["n"] += 1
        pair_stats[pair_key]["steps_all"].append(steps)
        if done:
            winner = infer_terminal_winner(obs, info, fallback="position") or "opp"
            ego_won = (winner == "ego") if ego_player == 1 else (winner != "ego")
            if ego_won:
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
    skill_usage = dict(zip(JOINT_NAMES, ego_skill_count))
    return {
        "wins": wins, "losses": losses, "trunc": trunc,
        "attempted": attempted,
        "real_wr": real_wr,
        "trunc_rate": trunc / max(attempted, 1),
        "avg_xs": float(np.mean(crossings_list)),
        "avg_steps_all": float(np.mean(steps_list_all)),
        "avg_steps_done": float(np.mean(steps_list_done)) if steps_list_done else float("nan"),
        "skill_pct": [c / total_skills for c in ego_skill_count],
        "landings": land,
        "reason_by_skill": {k: dict(v) for k, v in reason_by_skill.items()},
        # rates over ego's descending shots, so "out" and "net" share a denominator.
        # Reporting only out-rate hides net faults: those never reach the far side,
        # so they fall out of any count restricted to shots that crossed.
        "out_rate": (land["wide"] + land["long"]) / max(land["shots"], 1),
        "wide_rate": land["wide"] / max(land["shots"], 1),
        "long_rate": land["long"] / max(land["shots"], 1),
        "net_rate": land["net"] / max(land["shots"], 1),
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
    parser.add_argument("--ppo-model", default=PPO_MODEL_PATH,

                        help="racket tracker to evaluate under; the "

                             "calibration in aim_geometry is fitted per "

                             "tracker, so these must match")

    parser.add_argument("--rallies", type=int, default=30,
                        help="fixed rally count; ignored when --decided is given")
    parser.add_argument("--decided", type=int, default=None,
                        help="play until this many rallies are decided (team Table 1 uses 200)")
    parser.add_argument("--max-rallies", type=int, default=None,
                        help="hard cap when --decided is used (default 20x)")
    parser.add_argument("--pool-sides", action="store_true",
                        help="alternate which physical player ego is, as Table 1 does")
    parser.add_argument("--json", default=None,
                        help="Also write the summary table to this path as JSON, so "
                             "drivers do not have to parse the printed output.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baselines", nargs="+",
                        default=["random"] + list(SKILL_NAMES) + list(DEFENDER_NAMES))
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
    parser.add_argument("--tau", type=float, default=0.2,
                        help="Softmax temperature for the low-confidence fallback in the Phi "
                             "pickers. tau<=0 disables the fallback entirely (pure argmax).")
    parser.add_argument("--confidence-margin", type=float, default=0.05,
                        dest="confidence_margin",
                        help="Phi pickers fall back to softmax sampling when the top-two "
                             "action scores differ by less than this (default: 0.05). Measured "
                             "Phi top1-top2 gaps have a median of ~0.007, so the default fires "
                             "on ~97%% of decisions and makes the picker near-uniform; set 0.0 "
                             "for pure argmax.")
    parser.add_argument("--q2-model", default="models/model2_5skill_v3.pth",
                        help="Path to Q2 model (for ibr/ibr-q)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Seed every source, not just ours.  env.reset() randomises the robots'
    # initial pose through the *global* numpy RNG (mujoco_env_comp lines 277-282)
    # and MetaPolicy.sample draws with torch.multinomial, so seeding only this
    # Random() left the whole evaluation irreproducible -- the same command with
    # the same --seed gave 25.0% and 62.5% on consecutive runs.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    print(f"Loading ego policy from {args.checkpoint} ...")
    saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
    ego_encoding = skill_encoding_from_state_dict(saved)
    policy = MetaPolicy(state_dim=state_dim_for(ego_encoding)).to(device)
    policy.load_state_dict(saved)
    policy.eval()
    opp_encoding = ego_encoding

    # Build the list of (label, pick_fn) opponents
    opponents = []
    if args.vs_checkpoint:
        print(f"Loading opp policy from {args.vs_checkpoint} ...")
        opp_saved = torch.load(args.vs_checkpoint, map_location=device, weights_only=True)
        opp_encoding = skill_encoding_from_state_dict(opp_saved)
        opp_policy = MetaPolicy(state_dim=state_dim_for(opp_encoding)).to(device)
        opp_policy.load_state_dict(opp_saved)
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
            # Phi/Q are SimpleModel(76, ...) -- trained on the linear encoding,
            # so they get it explicitly rather than relying on the default.
            # The self-play policy's own encoding is read from its checkpoint
            # and is independent of this.
            return (encode_ego(obs, info, "linear") if player == 1
                    else encode_opp(obs, info, "linear"))

        nash_picker = make_picker(args.vs_strategy, model_p=phi,
                                   state_encoder_fn=_state_enc,
                                   model1=m1, model2=m2,
                                   tau=args.tau,
                                   confidence_margin=args.confidence_margin)

        def opp_pick(_state_np, obs, info, ego_idx, _picker=nash_picker):
            # Opp is player 2; ego_idx is the "other_skill_idx" from opp's view.
            return int(_picker(2, obs, ego_idx, info))

        opponents.append((args.vs_strategy, opp_pick))
    else:
        for opp_name in args.baselines:
            opponents.append((opp_name, pick_baseline(opp_name, rng)))

    print(f"Loading PPO from {args.ppo_model} (CPU) ...")
    ppo = PPO.load(args.ppo_model, device="cpu")

    env = SkillEnv(proc_id=1, history=HISTORY, skill_profile="aggressive")

    print(f"\nEvaluating {args.rallies} rallies vs each opponent\n" + "=" * 100)
    # Full names, not truncated: SKILL_NAMES has two 4-char-prefix collisions
    # (left/left_short, right_short/right) that made earlier abbreviated
    # columns genuinely ambiguous -- e.g. "right" could mean either skill.
    skill_col_w = max(len(s) for s in JOINT_NAMES) + 1
    label_w = max(12, max(len(lbl) for lbl, _ in opponents) + 2)
    header = (f"{'opp':<{label_w}} {'real_wr':>8} {'wins':>5} {'loss':>5} {'trunc':>6} "
              f"{'trunc%':>7} {'avg_xs':>7} {'stp_all':>8} {'stp_done':>8}  "
              f"ego skill %: "
              + " ".join(f"{s:>{skill_col_w}}" for s in JOINT_NAMES))
    print(header)
    print("-" * len(header))

    all_results = []
    for label, pick_fn in opponents:
        r = eval_one(env, ppo, policy, pick_fn, device, args.rallies, rng,
                     ego_encoding=ego_encoding, opp_encoding=opp_encoding,
                     decided_target=args.decided, max_rallies=args.max_rallies,
                     pool_sides=args.pool_sides)
        skill_str = " ".join(f"{p:>{skill_col_w}.1%}" for p in r["skill_pct"])
        stp_done = f"{r['avg_steps_done']:>8.1f}" if r['avg_steps_done'] == r['avg_steps_done'] else f"{'nan':>8}"
        print(f"{label:<{label_w}} {r['real_wr']:>8.1%} "
              f"{r['wins']:>5d} {r['losses']:>5d} {r['trunc']:>6d} "
              f"{r['trunc_rate']:>7.1%} {r['avg_xs']:>7.1f} "
              f"{r['avg_steps_all']:>8.1f} {stp_done}  "
              f"           {skill_str}")
        all_results.append((label, r))

    print("-" * len(header))

    if args.json:
        import json as _json
        payload = {
            "checkpoint": args.checkpoint,
            "skill_encoding": ego_encoding,
            "ppo_model": args.ppo_model,
            "rallies": args.rallies,
            "seed": args.seed,
            "opponents": [
                {
                    "label": label,
                    "real_wr": r["real_wr"],
                    "wins": r["wins"],
                    "losses": r["losses"],
                    "trunc": r["trunc"],
                    "trunc_rate": r["trunc_rate"],
                    "avg_xs": r["avg_xs"],
                    "avg_steps_all": r["avg_steps_all"],
                    "avg_steps_done": r["avg_steps_done"],
                    "skill_pct": list(r["skill_pct"]),
                    "landings": r["landings"],
                    "reason_by_skill": r["reason_by_skill"],
                    "out_rate": r["out_rate"],
                    "wide_rate": r["wide_rate"],
                    "long_rate": r["long_rate"],
                    "net_rate": r["net_rate"],
                }
                for label, r in all_results
            ],
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(_json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")

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
        print(f"  {'ego_init':<22} {'opp_init':<22} {'n':>4} "
              f"{'wins':>5} {'loss':>5} {'trunc':>5} {'wr':>7} "
              f"{'stp_all':>8} {'stp_done':>8}")
        for (ei, oi), s in sorted(r["pair_stats"].items()):
            n_done = s["wins"] + s["losses"]
            wr = s["wins"] / n_done if n_done > 0 else float("nan")
            wr_str = f"{wr:>6.1%}" if n_done > 0 else "  ---"
            stp_all = float(np.mean(s["steps_all"])) if s["steps_all"] else float("nan")
            stp_done = float(np.mean(s["steps_done"])) if s["steps_done"] else float("nan")
            stp_done_str = f"{stp_done:>8.1f}" if s["steps_done"] else f"{'---':>8}"
            print(f"  {JOINT_NAMES[ei]:<22} {JOINT_NAMES[oi]:<22} {s['n']:>4d} "
                  f"{s['wins']:>5d} {s['losses']:>5d} {s['trunc']:>5d} {wr_str} "
                  f"{stp_all:>8.1f} {stp_done_str}")
    env.close()


if __name__ == "__main__":
    main()
