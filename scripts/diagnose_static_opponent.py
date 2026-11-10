"""
Rally-level diagnosis of a method against one static (attack, preparation) opponent.

Logs every exchange (decision, attacker, the live half of each player's choice, racket
contacts) and how each point ended, then reports:
  - the method's win rate per side,
  - how points end (who missed a return, whose shot went out),
  - the method's reception outcome by the preparation it chose before the opponent's shot
    (returned / missed / opponent's shot out), and its attack choices.

    MUJOCO_GL=egl PYTHONPATH=. python scripts/diagnose_static_opponent.py --profile moderate_aggressive \\
        --model-dir runs_paper/ablations_qval/moderate_aggressive/phi/gamma0.99_s0 \\
        --method nash-p-pga --opponent right_short:right --rallies 40
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import collections
import contextlib
import io
import json

import numpy as np

MISS = "missed_{p}_return_after_valid_bounce"
OUT = "{p}_shot_out_before_valid_{q}_bounce"


def reception_outcomes(events, me):
    """For each opponent shot `me` had to receive: (my preparation before it, outcome).

    events: ordered dicts {"kind": "decision", "attacker", "prep": {1: name, 2: name}} /
            {"kind": "contact", "hitter": 1|2} / {"kind": "end", "reason": str}.
    Outcome: "returned" (my next contact), "missed" (point ended by my miss),
             "their_out" (their shot went out), "open" (rally truncated).
    """
    me_tag, them_tag = ("ego", "opp") if me == 1 else ("opp", "ego")
    out, prep_before, waiting = [], None, False
    for ev in events:
        if ev["kind"] == "decision" and ev["attacker"] != me:
            prep_before = ev["prep"][me]          # I am the defender: this preparation is live
        elif ev["kind"] == "contact":
            if ev["hitter"] != me:
                waiting = True                     # their shot is coming to me
            elif waiting:
                out.append((prep_before, "returned"))
                waiting = False
        elif ev["kind"] == "end" and waiting:
            reason = ev["reason"] or ""
            if reason == MISS.format(p=me_tag):
                out.append((prep_before, "missed"))
            elif reason == OUT.format(p=them_tag, q=me_tag):
                out.append((prep_before, "their_out"))
            else:
                out.append((prep_before, "other:" + reason))
            waiting = False
    if waiting:
        out.append((prep_before, "open"))
    return out


def summarize(rallies, me):
    wins = collections.Counter(r["winner"] for r in rallies)
    reasons = collections.Counter(r["reason"] for r in rallies)
    rec = collections.defaultdict(collections.Counter)
    for r in rallies:
        for prep, outcome in reception_outcomes(r["events"], me):
            rec[prep][outcome] += 1
    attacks = collections.Counter(a for r in rallies for a in r["my_attacks"])
    preps = collections.Counter(p for r in rallies for p in r["my_preps"])
    return {"wins": dict(wins), "reasons": dict(reasons), "reception": {k: dict(v) for k, v in rec.items()},
            "attacks": dict(attacks), "preps": dict(preps)}


def play(env, ppo, method_pick, opp_pick, me, mechanism, max_steps):
    from nash_skills.decision_clock import DecisionClock
    from nash_skills.ibr_sim import apply_joint_pair
    from nash_skills.v2.joint_action import joint_action_from_index
    from nash_skills.defend_positions import defend_position_from_index
    from nash_skills.skills import skill_from_index
    from nash_skills.v2.selfplay_rollout import _ppo_step
    apply_joint_pair(env, mechanism, 0, 0)
    obs, info = env.reset()
    clock = DecisionClock("contact")
    clock.reset()
    joints, events, my_attacks, my_preps = {1: 0, 2: 0}, [], [], []
    prev_hitter, n_dec, done = env._env.rally_state.get("last_hitter"), 0, False
    picks = {me: method_pick, 3 - me: opp_pick}
    for _ in range(max_steps):
        obs, _r, done, _t, info = _ppo_step(env, ppo, obs, info)
        hitter = env._env.rally_state.get("last_hitter")
        if hitter is not None and hitter != prev_hitter:
            events.append({"kind": "contact", "hitter": 1 if hitter == "ego" else 2})
        prev_hitter = hitter
        attacker = clock.update(env)
        if attacker is not None:
            pi = {**info, "current_joints": (joints[1], joints[2]), "attacker": attacker, "env": env,
                  "decision_index": n_dec}
            new = {p: picks[p](p, obs, joints[3 - p], pi) for p in (1, 2)}
            joints = new
            apply_joint_pair(env, mechanism, joints[1], joints[2])
            halves = {p: joint_action_from_index(joints[p]) for p in (1, 2)}
            events.append({"kind": "decision", "attacker": attacker,
                           "prep": {p: defend_position_from_index(halves[p][1]) for p in (1, 2)}})
            if attacker == me:
                my_attacks.append(skill_from_index(halves[me][0]))
            else:
                my_preps.append(defend_position_from_index(halves[me][1]))
            n_dec += 1
        if done:
            break
    winner = info.get("winner") if done else None
    reason = info.get("termination_reason") if done else "truncated"
    events.append({"kind": "end", "reason": reason})
    me_tag = "ego" if me == 1 else "opp"
    return {"winner": "me" if winner == me_tag else ("them" if winner else "none"), "reason": reason,
            "events": events, "my_attacks": my_attacks, "my_preps": my_preps}


def main():
    from stable_baselines3 import PPO
    from nash_skills.env_wrapper import SkillEnv
    from nash_skills.eval_matchup_attack_defend import make_picker_joint
    sys.path.insert(0, os.path.dirname(__file__))
    import eval_table1_attack_defend as t1
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--method", default="nash-p-pga")
    ap.add_argument("--opponent", required=True, help="attack:preparation, e.g. right_short:right")
    ap.add_argument("--rallies", type=int, default=40, help="per side")
    ap.add_argument("--mechanism", default="preparation_cal")
    ap.add_argument("--ppo", default="logs/best_model_tracker1/best_model")
    ap.add_argument("--max-steps", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    ppo = PPO.load(args.ppo, device="cpu")
    models = t1._load_models(args.model_dir)
    env = SkillEnv(proc_id=1, history=4, reset_mode="ready", skill_profile=args.profile, reset_targets=True)
    opp = tuple(args.opponent.split(":"))
    report = {}
    for me in (1, 2):
        np.random.seed(args.seed + me)
        method_pick = make_picker_joint(args.method, models["model_p"], state_encoder_fn=t1.make_phi_encoder(models["encoder"]),
                                        model1=models["model1"], model2=models["model2"], pga_seed=args.seed + me)
        opp_pick = make_picker_joint(opp, None)
        with contextlib.redirect_stdout(io.StringIO()):
            rallies = [play(env, ppo, method_pick, opp_pick, me, args.mechanism, args.max_steps)
                       for _ in range(args.rallies)]
        report[f"as_P{me}"] = summarize(rallies, me)
    env.close()
    print(json.dumps(report, indent=1))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
