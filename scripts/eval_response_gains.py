"""
Table 2 (paper Eq. 15): unilateral response gains of a method playing both seats, estimated from
exact simulated continuations at decision points sampled from its own rallies
(nash_skills.response_gains). Candidates: the method itself plus the 15 fixed strategies.

    MUJOCO_GL=egl PYTHONPATH=. python scripts/eval_response_gains.py --method nash-p-argmax \\
        --model-dir runs_paper/paper_alpha/aggressive/phi/main_s0 --skill-profile aggressive \\
        --points 24 --workers 6 --repeats 1 --output out.json
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import math
import multiprocessing as mp
import time

import numpy as np

_W = {}


def summarize(points):
    """Per player: the policy-level gain (best fixed response by its mean value over all points, minus
    the method's own mean, with a bootstrap 95% interval), plus the per-state max for reference."""
    from nash_skills.response_gains import gain_from_values, policy_level_gain
    out = {}
    for player in (1, 2):
        vals = [{("orig" if k == "orig" else int(k)): v for k, v in p[f"p{player}"].items()} for p in points]
        pol = policy_level_gain(vals)
        per_state = [gain_from_values(v["orig"], {k: x for k, x in v.items() if k != "orig"})["gain"] for v in vals]
        out[f"player{player}"] = {**pol, "n_points": len(vals), "per_state_mean_gain": float(np.mean(per_state))}
    out["profile_regret"] = max(out["player1"]["gain"], out["player2"]["gain"])
    return out


def _init(ppo_path, model_dir, spec, profile, mechanism, gamma, max_steps, repeats, timing):
    from stable_baselines3 import PPO
    from nash_skills.env_wrapper import SkillEnv
    from nash_skills.eval_matchup_attack_defend import make_picker_joint
    import eval_table1_attack_defend as t1
    ppo = PPO.load(ppo_path, device="cpu")
    models = t1._load_models(model_dir)
    counter = {"n": 0}

    def make_method_picker():
        counter["n"] += 1
        return make_picker_joint(spec, models["model_p"], state_encoder_fn=t1.make_phi_encoder(models["encoder"]),
                                 model1=models["model1"], model2=models["model2"], ppo=ppo,
                                 skill_profile=profile, defend_mechanism=mechanism, pga_seed=counter["n"])

    def make_env():
        return SkillEnv(proc_id=1, history=4, reset_mode="ready", skill_profile=profile, reset_targets=True)

    _W.update(ppo=ppo, make_method_picker=make_method_picker, make_env=make_env, mechanism=mechanism,
              gamma=gamma, max_steps=max_steps, repeats=repeats, timing=timing)


def run_chunk(task):
    from nash_skills.response_gains import ResponseGainEvaluator, collect_decision_points
    seed, n_points = task
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    points = collect_decision_points(_W["make_env"](), _W["ppo"], _W["mechanism"], _W["make_method_picker"],
                                     n_points, rng, _W["max_steps"], decision_timing=_W["timing"])
    ev = ResponseGainEvaluator(_W["make_env"](), _W["ppo"], _W["mechanism"], _W["gamma"], _W["max_steps"],
                               _W["make_method_picker"], repeats=_W["repeats"], decision_timing=_W["timing"])
    out = []
    for p in points:
        row = {"attacker": p.attacker, "decision_index": p.decision_index}
        for player in (1, 2):
            row[f"p{player}"] = {str(k): v for k, v in ev.candidate_values(p, player).items()}
        out.append(row)
    return out


def main():
    from eval_table1_attack_defend import check_model_encoder, check_model_settings
    from nash_skills.decision_clock import DECISION_TIMINGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--skill-profile", required=True)
    ap.add_argument("--defend-mechanism", default="preparation_cal")
    ap.add_argument("--decision-timing", choices=DECISION_TIMINGS, default="contact",
                    help="when decisions happen: each racket contact or each net crossing (must match the models)")
    ap.add_argument("--points", type=int, default=24, help="decision points in total")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=1, help="rollouts per candidate (stochastic methods > 1)")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--max-steps", type=int, default=2500)
    ap.add_argument("--ppo", default="logs/best_model_tracker1/best_model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    check_model_encoder(args.model_dir)
    check_model_settings(args.model_dir, {"defend_mechanism": args.defend_mechanism, "decision_timing": args.decision_timing,
                                          "skill_profile": args.skill_profile})
    per = [args.points // args.workers + (1 if k < args.points % args.workers else 0) for k in range(args.workers)]
    tasks = [(args.seed * 1000 + k, n) for k, n in enumerate(per) if n > 0]
    start, points = time.time(), []
    with mp.get_context("spawn").Pool(len(tasks), initializer=_init,
                                      initargs=(args.ppo, args.model_dir, args.method, args.skill_profile,
                                                args.defend_mechanism, args.gamma, args.max_steps, args.repeats,
                                                args.decision_timing)) as pool:
        for chunk in pool.imap_unordered(run_chunk, tasks):
            points += chunk
            print(f"  {len(points)}/{args.points} decision points ({time.time() - start:.0f}s)", flush=True)
    summary = summarize(points)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"args": vars(args), "summary": summary, "points": points}, f)
    for player in (1, 2):
        s = summary[f"player{player}"]
        print(f"player {player}: response gain {s['gain']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}] "
              f"(best response {s['best_response']}, method's own value {s['original_mean']:+.3f}, "
              f"{s['n_points']} points; per-state max {s['per_state_mean_gain']:.3f})")
    print(f"profile regret (max over players): {summary['profile_regret']:.3f}  ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
