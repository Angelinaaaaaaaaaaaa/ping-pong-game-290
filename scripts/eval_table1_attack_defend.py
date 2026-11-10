"""
Table 1 (paper Sec. 5.1) for Design B: one method's point win rate against the
15 static (attack target, preparation position) pairings, the random baseline,
IBR, and any extra opponents (e.g. the other proposed method).

Every opponent is played in both side orientations (method as P1 and as P2).
Win rates count decided rallies only; truncations are reported separately.
Static mean is the unweighted mean of the 15 static cells, static minimum
their minimum.

Methods / opponents (strategy specs of eval_matchup_attack_defend):
  nash-p-hard, nash-p-pga, ... (need --model-dir with model_p/model1/model2)
  ibr, ibr-q                   (Q-based stand-ins for the paper's simulated IBR)
  random, static               ("static" expands to the 15 pairings)
  selfplay:<ckpt>, selfplay-argmax:<ckpt>

Run:
    MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONPATH=. python scripts/eval_table1_attack_defend.py \\
        --method nash-p-hard --model-dir runs/phi_prep_main --episodes 100 --workers 40 \\
        --output runs/table1/nash_p_hard.json
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import math
import multiprocessing as mp
import time

import numpy as np

from nash_skills.skills import SKILL_NAMES
from nash_skills.defend_positions import DEFEND_POSITION_NAMES

PPO_MODEL_PATH = "logs/best_model_tracker1/best_model"
STATIC_OPPONENTS = [(a, d) for a in SKILL_NAMES for d in DEFEND_POSITION_NAMES]
MODEL_NAMES = {"model_p": "model_p_attack_defend.pth", "model1": "model1_attack_defend.pth",
               "model2": "model2_attack_defend.pth"}

_W = {}


def expand_opponents(specs):
    out = []
    for spec in specs:
        out.extend(STATIC_OPPONENTS if spec == "static" else [spec])
    return out


def spec_label(spec):
    return f"{spec[0]}/{spec[1]}" if isinstance(spec, tuple) else spec


def check_model_encoder(model_dir):
    """The state encoder the Phi/Q models were trained with; refuses unknown ones (another layout)."""
    from nash_skills.v2.state_encoder_attack_defend import ENCODER_DIMS
    path = os.path.join(model_dir, "heldout_attack_defend.json")
    encoder = None
    if os.path.exists(path):
        with open(path) as f:
            encoder = json.load(f).get("encoder")
    if encoder not in ENCODER_DIMS:
        raise ValueError(f"{model_dir} was trained with state encoder {encoder!r}, not one of {sorted(ENCODER_DIMS)}; "
                         "retrain it with a current encoder.")
    return encoder


_SETTING_DEFAULTS = {"defend_mechanism": "reactive", "decision_timing": "crossing", "skill_profile": "aggressive"}


def _compare_settings(source, found, settings):
    for key, want in settings.items():
        got = found.get(key, _SETTING_DEFAULTS[key])
        if got != want:
            raise ValueError(f"{source} was made with {key}={got!r}, but this evaluation uses {key}={want!r}")


def check_model_settings(model_dir, settings):
    """Phi/Q must come from data collected under the evaluation's mechanism, timing and profile."""
    with open(os.path.join(model_dir, "heldout_attack_defend.json")) as f:
        _compare_settings(model_dir, json.load(f), settings)


def check_selfplay_settings(spec, settings):
    if not isinstance(spec, str) or spec.split(":", 1)[0] not in ("selfplay", "selfplay-argmax"):
        return
    path = spec.split(":", 1)[1]
    with open(path + ".json") as f:
        meta = json.load(f)
    # Legacy shared-actor checkpoints never recorded their mechanism.
    checked = {k: v for k, v in settings.items() if k in meta or k != "defend_mechanism"}
    _compare_settings(path, meta, checked)


def build_tasks(method, opponents, episodes, split, seed, max_attempt_factor=None):
    """One task per (opponent, side, chunk); `split` maps an opponent label to its chunk count.
    max_attempt_factor caps the rally attempts of a task at factor x its decided rallies (rallies that
    reach the step limit are re-drawn); None keeps run_matchup_joint's default of 10x."""
    tasks = []
    for opp in opponents:
        n_chunks = max(1, split.get(spec_label(opp), 1))
        sizes = [episodes // n_chunks + (1 if k < episodes % n_chunks else 0) for k in range(n_chunks)]
        for side in (1, 2):
            for size in sizes:
                if size > 0:
                    task = {"method": method, "opponent": opp, "method_player": side, "episodes": size,
                            "seed": seed * 100000 + len(tasks)}
                    if max_attempt_factor is not None:
                        task["max_attempts"] = max(size, int(math.ceil(max_attempt_factor * size)))
                    tasks.append(task)
    return tasks


def timing_summary(rows):
    """Mean strategic computation time per decision, for the method and per opponent."""
    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")
    by_opp = {}
    for r in rows:
        d = by_opp.setdefault(r["opponent"], {"method": [], "opponent": []})
        d["method"] += r["method_seconds"]
        d["opponent"] += r["opponent_seconds"]
    return {"method_mean_s": mean([x for r in rows for x in r["method_seconds"]]),
            "by_opponent": {k: {"method_mean_s": mean(v["method"]), "opponent_mean_s": mean(v["opponent"])}
                            for k, v in by_opp.items()}}


def _load_models(model_dir):
    """Phi, Q1, Q2 sized for the state encoder they were trained with (also returned, as "encoder")."""
    import torch
    from model_arch import SimpleModel
    from nash_skills.v2.state_encoder_attack_defend import STATE_ENCODER_NAME, state_dim
    if model_dir is None:
        return {"model_p": None, "model1": None, "model2": None, "encoder": STATE_ENCODER_NAME}
    encoder = check_model_encoder(model_dir)
    d = state_dim(encoder)
    models = {"model_p": SimpleModel(d, [64, 32, 16], 1, last_layer_activation=None),
              "model1": SimpleModel(d, [64, 32, 16], 1), "model2": SimpleModel(d, [64, 32, 16], 1)}
    for key, model in models.items():
        model.load_state_dict(torch.load(os.path.join(model_dir, MODEL_NAMES[key]), map_location="cpu"))
        model.eval()
    return {**models, "encoder": encoder}


def _init_worker(ppo_path, model_dir, settings, max_steps, ibr_config):
    from stable_baselines3 import PPO
    _W["ppo"] = PPO.load(ppo_path, device="cpu")
    _W.update(_load_models(model_dir))
    _W["encode"] = make_phi_encoder(_W["encoder"])
    _W.update(mechanism=settings["defend_mechanism"], timing=settings["decision_timing"],
              profile=settings["skill_profile"], max_steps=max_steps, ibr_config=ibr_config)


def make_phi_encoder(encoder):
    """State function for Phi/Q: the ego-perspective state they were trained on, with their encoder;
    every rule overwrites all four action blocks, so the defend placeholders never matter."""
    from nash_skills.v2.state_encoder_attack_defend import encode_ego

    def encode(obs, info, _player):
        return encode_ego(obs, info, "center", "center", encoder=encoder)
    return encode


def _encode_for_phi(obs, info, player):
    """make_phi_encoder for the v2 encoder (models trained before the v3 state)."""
    from nash_skills.v2.state_encoder_attack_defend import STATE_ENCODER_NAME
    return make_phi_encoder(STATE_ENCODER_NAME)(obs, info, player)


def run_task(task):
    import torch
    from nash_skills.eval_matchup_attack_defend import run_matchup_joint
    np.random.seed(task["seed"] % (2 ** 32))
    torch.manual_seed(task["seed"])
    method, opponent, side = task["method"], task["opponent"], task["method_player"]
    s1, s2 = (method, opponent) if side == 1 else (opponent, method)
    result = run_matchup_joint(
        s1, s2, _W["ppo"], _W["model_p"], n_episodes=task["episodes"], max_steps_per_episode=_W["max_steps"],
        max_attempts=task.get("max_attempts"), state_encoder_fn=_W["encode"], model1=_W["model1"], model2=_W["model2"],
        defend_mechanism=_W["mechanism"], reset_targets=True, skill_profile=_W["profile"],
        decision_timing=_W["timing"], ibr_sim_config=_W["ibr_config"], picker_seed=task["seed"],
    )
    wins = result.ego_wins if side == 1 else result.opp_wins
    opp_side = 3 - side
    return {"opponent": spec_label(opponent), "method_player": side,
            "decided": result.episodes, "wins": wins, "truncated": result.truncated_episodes,
            "rally_lengths": result.rally_lengths, "exchanges": result.exchanges,
            "method_seconds": result.decision_seconds[side], "opponent_seconds": result.decision_seconds[opp_side],
            "opponent_picker_stats": result.picker_stats.get(opp_side)}


def wilson(wins, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def summarize(rows):
    """Pool both orientations per opponent; static mean/min over the 15 pairings."""
    cells = {}
    for r in rows:
        c = cells.setdefault(r["opponent"], {"wins": 0, "decided": 0, "truncated": 0, "by_side": {}})
        c["wins"] += r["wins"]
        c["decided"] += r["decided"]
        c["truncated"] += r["truncated"]
        side = c["by_side"].setdefault(str(r["method_player"]), {"wins": 0, "decided": 0})
        side["wins"] += r["wins"]
        side["decided"] += r["decided"]
    for c in cells.values():
        c["win"] = c["wins"] / c["decided"] if c["decided"] else float("nan")
        c["ci"] = wilson(c["wins"], c["decided"])
        attempts = c["decided"] + c["truncated"]
        c["trunc_rate"] = c["truncated"] / attempts if attempts else float("nan")
    static = [cells[spec_label(s)] for s in STATIC_OPPONENTS if spec_label(s) in cells]
    summary = {"cells": cells}
    if static:
        wins = np.array([c["win"] for c in static])
        se = math.sqrt(sum(c["win"] * (1 - c["win"]) / c["decided"] for c in static if c["decided"])) / len(static)
        labels = [spec_label(s) for s in STATIC_OPPONENTS if spec_label(s) in cells]
        summary["static_mean"] = float(wins.mean())
        summary["static_mean_ci"] = (float(wins.mean() - 1.96 * se), float(wins.mean() + 1.96 * se))
        summary["static_min"] = float(wins.min())
        summary["static_min_opponent"] = labels[int(wins.argmin())]
    return summary


def print_table(method, summary):
    cells = summary["cells"]
    print(f"\nMethod: {method}  (decided-only win rate, both sides pooled)")
    print(f"{'attack target':<14}" + "".join(f"{d:>10}" for d in DEFEND_POSITION_NAMES))
    for a in SKILL_NAMES:
        row = [cells.get(f"{a}/{d}", {}).get("win", float("nan")) for d in DEFEND_POSITION_NAMES]
        print(f"{a:<14}" + "".join(f"{w:>10.3f}" for w in row))
    if "static_mean" in summary:
        lo, hi = summary["static_mean_ci"]
        print(f"\nstatic mean {summary['static_mean']:.3f} [{lo:.3f}, {hi:.3f}]   "
              f"static min {summary['static_min']:.3f} (vs {summary['static_min_opponent']})")
    for label, c in cells.items():
        if "/" in label:
            continue
        sides = "  ".join(f"as P{k}: {v['wins'] / v['decided']:.3f}" for k, v in sorted(c["by_side"].items()) if v["decided"])
        print(f"vs {label:<34} {c['win']:.3f} [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}]  n={c['decided']}  "
              f"trunc={c['trunc_rate']:.3f}  ({sides})")
    total_trunc = sum(c["truncated"] for c in cells.values())
    total = sum(c["truncated"] + c["decided"] for c in cells.values())
    print(f"truncated attempts overall: {total_trunc}/{total}")


def main():
    from nash_skills.decision_clock import DECISION_TIMINGS
    from nash_skills.ibr_sim import IBR_DEFAULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--opponents", nargs="+", default=["static", "random", "ibr"])
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--episodes", type=int, default=100, help="decided rallies per opponent per side")
    ap.add_argument("--max-steps", type=int, default=2500)
    ap.add_argument("--max-attempt-factor", type=float, default=None,
                    help="cap rally attempts per task at this multiple of its decided rallies "
                         "(default: 10x); rallies reaching --max-steps are re-drawn up to the cap")
    ap.add_argument("--defend-mechanism", default="preparation")
    ap.add_argument("--decision-timing", choices=DECISION_TIMINGS, default="crossing")
    ap.add_argument("--skill-profile", default="aggressive")
    ap.add_argument("--split", nargs="*", default=[], metavar="OPPONENT=N",
                    help="spread an opponent's rallies over N tasks, e.g. ibr-sim=20")
    ap.add_argument("--ibr-rounds", type=int, default=IBR_DEFAULTS["rounds"])
    ap.add_argument("--ibr-horizon", type=int, default=IBR_DEFAULTS["horizon_exchanges"],
                    help="H_IBR in exchanges")
    ap.add_argument("--ibr-gamma", type=float, default=IBR_DEFAULTS["gamma"])
    ap.add_argument("--ppo", default=PPO_MODEL_PATH)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    settings = {"defend_mechanism": args.defend_mechanism, "decision_timing": args.decision_timing,
                "skill_profile": args.skill_profile}
    if args.model_dir is not None:
        check_model_encoder(args.model_dir)
        check_model_settings(args.model_dir, settings)
    opponents = expand_opponents(args.opponents)
    for spec in [args.method] + opponents:
        check_selfplay_settings(spec, settings)
    split = {k: int(v) for k, v in (item.split("=", 1) for item in args.split)}
    ibr_config = {"rounds": args.ibr_rounds, "horizon_exchanges": args.ibr_horizon, "gamma": args.ibr_gamma}
    tasks = build_tasks(args.method, opponents, args.episodes, split, args.seed, args.max_attempt_factor)
    print(f"{args.method}: {len(opponents)} opponents x 2 sides x {args.episodes} decided rallies "
          f"({len(tasks)} tasks) on {args.workers} workers, {settings}", flush=True)
    start, rows = time.time(), []
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker,
                                      initargs=(args.ppo, args.model_dir, settings, args.max_steps,
                                                ibr_config)) as pool:
        for i, row in enumerate(pool.imap_unordered(run_task, tasks), 1):
            rows.append(row)
            print(f"  {i}/{len(tasks)} tasks done ({time.time() - start:.0f}s)", flush=True)
    summary = summarize(rows)
    summary["timing"] = timing_summary(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"args": vars(args), "settings": settings, "ibr_config": ibr_config,
                   "summary": summary, "rows": rows}, f)
    print_table(args.method, summary)
    t = summary["timing"]
    print(f"strategic time per decision: method {t['method_mean_s'] * 1000:.1f} ms; "
          + ", ".join(f"{k} {v['opponent_mean_s'] * 1000:.1f} ms" for k, v in t["by_opponent"].items()
                      if "/" not in k))
    print(f"\nSaved {args.output}  ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
