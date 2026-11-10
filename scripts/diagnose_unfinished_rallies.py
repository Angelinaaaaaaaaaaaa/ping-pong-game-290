"""
Is counting a rally that is unfinished at the evaluation's step limit as half a point justified?
Plays single rallies of a method against given opponents (both sides) with a much longer step limit,
then reads each rally's outcome at the evaluation's limit and at longer cutoffs:

  decided_win_rate   wins / (wins + losses) among rallies finished by the cutoff (Table 1's statistic)
  half_score         (wins + unfinished / 2) / rallies
  later_fate         of the rallies unfinished at the first cutoff: how many end before the long
                     limit, how many of those the method wins, and how many never end

If the unfinished rallies mostly end later and the method wins about half of them, half a point is a
fair estimate; if they never end, they are stalemates (draws) whose share must be reported.

    MUJOCO_GL=egl PYTHONPATH=. python scripts/diagnose_unfinished_rallies.py --method nash-p-argmax \\
        --model-dir runs_v3/paper/aggressive/phi/gamma0.99_s0 --opponents center_safe/center \\
        --rallies-per-side 30 --max-steps 20000 --workers 8 --output out.json
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import multiprocessing as mp
import time

import numpy as np

from nash_skills.decision_clock import DECISION_TIMINGS

EVAL_STEP_LIMIT = 2500
_W = {}


def parse_opponent(label):
    return tuple(label.split("/")) if "/" in label else label


def _finished_by(record, cutoff):
    return record["steps"] is not None and record["steps"] <= cutoff


def outcomes_at(records, cutoff):
    wins = sum(_finished_by(r, cutoff) and r["winner"] == "method" for r in records)
    losses = sum(_finished_by(r, cutoff) and r["winner"] == "opponent" for r in records)
    return {"wins": wins, "losses": losses, "unfinished": len(records) - wins - losses}


def scores_at(records, cutoff):
    o = outcomes_at(records, cutoff)
    n, decided = len(records), o["wins"] + o["losses"]
    return {**o, "decided_win_rate": o["wins"] / decided if decided else None,
            "half_score": (o["wins"] + 0.5 * o["unfinished"]) / n if n else None,
            "unfinished_share": o["unfinished"] / n if n else None}


def later_fate(records, cutoff):
    late = [r for r in records if not _finished_by(r, cutoff)]
    ended = [r for r in late if r["steps"] is not None]
    return {"unfinished_at_cutoff": len(late), "ended_later": len(ended),
            "method_won_later": sum(r["winner"] == "method" for r in ended), "never_ended": len(late) - len(ended)}


def _init(ppo_path, model_dir, settings, max_steps):
    import eval_table1_attack_defend as t1
    t1._init_worker(ppo_path, model_dir, settings, max_steps, None)
    _W.update(t1._W)


def play(task):
    """One rally of the method against an opponent; steps None when unfinished at max_steps."""
    import torch
    from nash_skills.eval_matchup_attack_defend import run_matchup_joint
    np.random.seed(task["seed"] % (2 ** 32))
    torch.manual_seed(task["seed"])
    method, opponent = task["method"], parse_opponent(task["opponent"])
    s1, s2 = (method, opponent) if task["method_player"] == 1 else (opponent, method)
    r = run_matchup_joint(s1, s2, _W["ppo"], _W["model_p"], n_episodes=1, max_steps_per_episode=_W["max_steps"],
                          max_attempts=1, state_encoder_fn=_W["encode"], model1=_W["model1"], model2=_W["model2"],
                          defend_mechanism=_W["mechanism"], reset_targets=True, skill_profile=_W["profile"],
                          decision_timing=_W["timing"], picker_seed=task["seed"])
    if r.episodes == 0:
        return {**task, "steps": None, "winner": None, "exchanges": None}
    ego_won = r.ego_wins == 1
    method_won = ego_won if task["method_player"] == 1 else not ego_won
    return {**task, "steps": int(r.rally_lengths[0]), "winner": "method" if method_won else "opponent",
            "exchanges": int(r.exchanges[0])}


def summarize(records, cutoffs):
    out = {}
    for opp in sorted({r["opponent"] for r in records}):
        rs = [r for r in records if r["opponent"] == opp]
        out[opp] = {"rallies": len(rs), "by_cutoff": {str(c): scores_at(rs, c) for c in cutoffs},
                    "later_fate": later_fate(rs, cutoffs[0]),
                    "exchanges_of_late_finishers": sorted(r["exchanges"] for r in rs
                                                          if r["steps"] is not None and r["steps"] > cutoffs[0])}
    return out


def main():
    from eval_table1_attack_defend import PPO_MODEL_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--opponents", nargs="+", required=True, help="e.g. center_safe/center")
    ap.add_argument("--rallies-per-side", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--cutoffs", type=int, nargs="+", default=[EVAL_STEP_LIMIT, 5000, 10000, 20000])
    ap.add_argument("--skill-profile", default="aggressive")
    ap.add_argument("--defend-mechanism", default="preparation_cal")
    ap.add_argument("--decision-timing", choices=DECISION_TIMINGS, default="contact")
    ap.add_argument("--ppo", default=PPO_MODEL_PATH)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if max(args.cutoffs) > args.max_steps:
        raise ValueError(f"cutoffs {args.cutoffs} exceed --max-steps {args.max_steps}")
    settings = {"defend_mechanism": args.defend_mechanism, "decision_timing": args.decision_timing,
                "skill_profile": args.skill_profile}
    tasks = [{"method": args.method, "opponent": opp, "method_player": side, "seed": args.seed * 100000 + k}
             for opp in args.opponents for side in (1, 2) for k in range(args.rallies_per_side)]
    start, records = time.time(), []
    with mp.get_context("spawn").Pool(args.workers, initializer=_init,
                                      initargs=(args.ppo, args.model_dir, settings, args.max_steps)) as pool:
        for rec in pool.imap_unordered(play, tasks):
            records.append(rec)
            if len(records) % 10 == 0:
                print(f"  {len(records)}/{len(tasks)} rallies ({time.time() - start:.0f}s)", flush=True)
    summary = summarize(records, sorted(args.cutoffs))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"args": vars(args), "summary": summary, "records": records}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
