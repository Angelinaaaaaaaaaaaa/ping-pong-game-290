"""
Simulator-based experiment to validate whether TABLE_NEAR = 1.65 is reachable.

For each of the 5 skills, runs the PPO policy in the real MuJoCo environment
(comp.py-style: continuously across resets with warm-up discard) and records:
  - contact count
  - successful return count
  - landing x / y distribution

Then compares short-skill landings against the deep baseline to determine
whether TABLE_NEAR = 1.65 should stay or be raised.

Run from the project root:
    MUJOCO_GL=cgl venv/bin/python skill_eval/validate_table_near.py
    MUJOCO_GL=cgl venv/bin/python skill_eval/validate_table_near.py --trials 30 --steps 600
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import csv
import io
import json
import numpy as np

PPO_MODEL_PATH  = "logs/best_model_tracker1/best_model"
HISTORY         = 4
TABLE_SHIFT     = 1.5          # net x position
TABLE_X_MAX     = TABLE_SHIFT + 1.37   # far end of opponent half (2.87)
TABLE_Y_MAX     = 0.75         # table half-width

# The value under test
TABLE_NEAR_CURRENT = 1.65
# Candidate alternative if 1.65 is too aggressive
TABLE_NEAR_CANDIDATE = 1.75

SKILLS_TO_TEST  = ["left", "right", "left_short", "right_short", "center_safe"]


# ─────────────────────────────────────────────────────────────────────────── #
# PPO obs slicing                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

def build_obs1(obs, info):
    o = np.zeros(9 + 9 + 7 + 7 + 9 * HISTORY, dtype=np.float32)
    o[:9]    = obs[:9]
    o[9:18]  = obs[18:27]
    o[18:21] = info['diff_pos']
    o[21:25] = info['diff_quat']
    o[25:32] = info['target']
    o[32:]   = obs[42: 42 + HISTORY * 9]
    return o


def build_obs2(obs, info):
    o = np.zeros(9 + 9 + 7 + 7 + 9 * HISTORY, dtype=np.float32)
    o[:9]    = obs[9:18]
    o[9:18]  = obs[27:36]
    o[18:21] = info['diff_pos_opp']
    o[21:25] = info['diff_quat_opp']
    o[25:32] = info['target_opp']
    o[32:]   = obs[42 + HISTORY * 9: 42 + 2 * HISTORY * 9]
    return o


# ─────────────────────────────────────────────────────────────────────────── #
# Per-skill evaluation                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def evaluate_skill(skill_name, ppo, n_trials, n_steps):
    """
    Run simulator comp.py-style (continuous across resets).
    Discard first 500 steps as arm warm-up.
    Capture stdout to detect landing events printed by env.step().
    """
    from nash_skills.env_wrapper import SkillEnv

    total_steps  = n_trials * n_steps
    warmup_steps = min(500, total_steps // 4)

    env = SkillEnv(proc_id=0, history=HISTORY)
    env.set_skills(skill_name, "left")  # opponent always uses "left"
    obs, info = env.reset()

    # Per-contact records: (x_land, y_land, success)
    ego_records = []
    opp_records = []
    episodes = 0

    for step in range(total_steps):
        obs1 = build_obs1(obs, info)
        obs2 = build_obs2(obs, info)

        action1, _ = ppo.predict(obs1, deterministic=True)
        action2, _ = ppo.predict(obs2, deterministic=True)

        action = np.zeros(18)
        action[:9] = action1[:9]
        action[9:] = action2[:9]

        buf = io.StringIO()
        sys.stdout = buf
        obs, _, done, _, info = env.step(action)
        sys.stdout = sys.__stdout__

        if step >= warmup_steps:
            for line in buf.getvalue().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    x_land = float(parts[-2])
                    y_land = float(parts[-1])
                except ValueError:
                    continue

                if "by ego" in line:
                    in_opp = TABLE_SHIFT < x_land < TABLE_X_MAX and abs(y_land) < TABLE_Y_MAX
                    ego_records.append((x_land, y_land, in_opp))
                elif "by opp" in line:
                    in_ego = 0 < x_land < TABLE_SHIFT and abs(y_land) < TABLE_Y_MAX
                    opp_records.append((x_land, y_land, in_ego))

        if done:
            episodes += 1
            obs, info = env.reset()
            env.set_skills(skill_name, "left")

    env.close()

    def aggregate(records):
        if not records:
            return {
                'n': 0, 'success': 0, 'success_rate': None,
                'x_mean': None, 'x_std': None,
                'y_mean': None, 'y_std': None,
                'x_min': None, 'x_max': None,
                'pct_near_net': None,   # fraction landing x < 1.80
            }
        xs = [r[0] for r in records]
        ys = [r[1] for r in records]
        ok = [r[2] for r in records]
        near_net = sum(1 for x in xs if x < 1.80) / len(xs)
        return {
            'n':            len(records),
            'success':      sum(ok),
            'success_rate': float(np.mean(ok)),
            'x_mean':       float(np.mean(xs)),
            'x_std':        float(np.std(xs)),
            'y_mean':       float(np.mean(ys)),
            'y_std':        float(np.std(ys)),
            'x_min':        float(np.min(xs)),
            'x_max':        float(np.max(xs)),
            'pct_near_net': float(near_net),
        }

    return {
        'skill':        skill_name,
        'total_steps':  total_steps,
        'warmup_steps': warmup_steps,
        'episodes':     episodes,
        'ego':          aggregate(ego_records),
        'opp':          aggregate(opp_records),
    }


# ─────────────────────────────────────────────────────────────────────────── #
# Analysis: is TABLE_NEAR = 1.65 reachable?                                  #
# ─────────────────────────────────────────────────────────────────────────── #

def analyse_table_near(skill_results):
    """
    Compare short-skill landings vs deep-skill landings and judge whether
    TABLE_NEAR = 1.65 is a realistic target.

    Criteria:
      - short skills must land closer to net than deep skills  (x_mean ↓)
      - at least some landings should be < 1.80  (pct_near_net > 0)
      - if x_mean for short skills is >= deep skill x_mean, TABLE_NEAR is not reached

    Returns a recommendation dict.
    """
    analysis = {}

    for short_skill, deep_skill in [("left_short", "left"), ("right_short", "right")]:
        rs = skill_results[short_skill]['ego']
        rd = skill_results[deep_skill]['ego']

        entry = {
            'short_skill':        short_skill,
            'deep_skill':         deep_skill,
            'short_n':            rs['n'],
            'deep_n':             rd['n'],
        }

        if rs['n'] == 0 or rd['n'] == 0:
            entry['verdict'] = "INSUFFICIENT_DATA"
            entry['recommendation'] = "Run more trials (--trials 40 --steps 800)"
            analysis[short_skill] = entry
            continue

        dx = rs['x_mean'] - rd['x_mean']   # negative = shorter (good)
        entry['x_mean_short'] = rs['x_mean']
        entry['x_mean_deep']  = rd['x_mean']
        entry['delta_x']      = dx
        entry['pct_near_net'] = rs['pct_near_net']

        # The intended target for short = TABLE_NEAR_CURRENT = 1.65
        # The intended target for deep  ≈ 2.185
        # Expected delta ≈ -0.535; acceptable range: delta < -0.15 (clearly shorter)

        if dx < -0.15 and rs['pct_near_net'] > 0.05:
            entry['verdict'] = "CONFIRMED"
            entry['recommendation'] = f"TABLE_NEAR = {TABLE_NEAR_CURRENT} is reachable."
        elif dx < 0:
            entry['verdict'] = "PARTIAL"
            entry['recommendation'] = (
                f"Short skills do land closer to net (Δx={dx:+.3f}) but "
                f"only {rs['pct_near_net']:.0%} of landings are within 1.80m. "
                f"Consider raising TABLE_NEAR to {TABLE_NEAR_CANDIDATE}."
            )
        else:
            entry['verdict'] = "NOT_CONFIRMED"
            entry['recommendation'] = (
                f"Short skills are NOT landing closer to net (Δx={dx:+.3f} ≥ 0). "
                f"TABLE_NEAR = {TABLE_NEAR_CURRENT} is too aggressive for the current PPO. "
                f"Recommend raising to {TABLE_NEAR_CANDIDATE} or higher."
            )

        analysis[short_skill] = entry

    # center_safe: should land centrally (|y_mean| << |left y_mean|)
    rc = skill_results['center_safe']['ego']
    rl = skill_results['left']['ego']
    center_entry = {}
    if rc['n'] > 0 and rl['n'] > 0:
        abs_y_center = abs(rc['y_mean'])
        abs_y_left   = abs(rl['y_mean'])
        center_entry = {
            'center_y_mean': rc['y_mean'],
            'left_y_mean':   rl['y_mean'],
            'more_central':  abs_y_center < abs_y_left,
            'verdict':       "CONFIRMED" if abs_y_center < abs_y_left * 0.5 else "PARTIAL",
        }
    else:
        center_entry = {'verdict': "INSUFFICIENT_DATA"}
    analysis['center_safe'] = center_entry

    return analysis


# ─────────────────────────────────────────────────────────────────────────── #
# CSV export                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

def save_csv(skill_results, path):
    rows = []
    for skill_name, r in skill_results.items():
        for player, stats in [("ego", r['ego']), ("opp", r['opp'])]:
            row = {
                'skill':        skill_name,
                'player':       player,
                'total_steps':  r['total_steps'],
                'warmup_steps': r['warmup_steps'],
                'episodes':     r['episodes'],
            }
            row.update(stats)
            rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────── #
# Main                                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(
        description="Validate TABLE_NEAR=1.65 reachability in MuJoCo.")
    parser.add_argument("--trials", type=int, default=20,
                        help="Episodes per skill (default 20)")
    parser.add_argument("--steps",  type=int, default=500,
                        help="Max steps per episode (default 500)")
    parser.add_argument("--output-json", default="skill_eval/table_near_results.json")
    parser.add_argument("--output-csv",  default="skill_eval/table_near_results.csv")
    args = parser.parse_args()

    from stable_baselines3 import PPO
    print("Loading PPO policy ...")
    ppo = PPO.load(PPO_MODEL_PATH)
    print(f"  Loaded: {PPO_MODEL_PATH}")

    total_steps_per_skill = args.trials * args.steps
    print(f"\nRunning {total_steps_per_skill} steps per skill "
          f"({args.trials} trials × {args.steps} steps), "
          f"first 500 discarded as warm-up.\n")

    skill_results = {}
    for skill_name in SKILLS_TO_TEST:
        print(f"--- Evaluating: {skill_name} ---")
        r = evaluate_skill(skill_name, ppo, n_trials=args.trials, n_steps=args.steps)
        skill_results[skill_name] = r

        e = r['ego']
        if e['n'] == 0:
            print(f"  ego: 0 contacts")
        else:
            print(f"  ego: n={e['n']}  success={e['success_rate']:.1%}  "
                  f"x={e['x_mean']:.3f}±{e['x_std']:.3f}  "
                  f"y={e['y_mean']:.3f}±{e['y_std']:.3f}  "
                  f"x_range=[{e['x_min']:.3f}, {e['x_max']:.3f}]  "
                  f"pct<1.80={e['pct_near_net']:.0%}")

        o = r['opp']
        if o['n'] == 0:
            print(f"  opp: 0 contacts")
        else:
            print(f"  opp: n={o['n']}  success={o['success_rate']:.1%}  "
                  f"x={o['x_mean']:.3f}±{o['x_std']:.3f}  "
                  f"y={o['y_mean']:.3f}±{o['y_std']:.3f}")
        print()

    # ── Analysis ─────────────────────────────────────────────────────────── #
    print("=" * 65)
    print("TABLE_NEAR = 1.65 VALIDATION")
    print("=" * 65)

    analysis = analyse_table_near(skill_results)

    for skill_name in ["left_short", "right_short", "center_safe"]:
        a = analysis[skill_name]
        print(f"\n  [{skill_name}]  verdict: {a.get('verdict', 'N/A')}")
        if 'delta_x' in a:
            print(f"    Δx vs deep baseline = {a['delta_x']:+.3f}  "
                  f"(pct landings < 1.80 = {a['pct_near_net']:.0%})")
        if 'recommendation' in a:
            print(f"    → {a['recommendation']}")
        if 'center_y_mean' in a:
            print(f"    center y_mean={a['center_y_mean']:.3f}  "
                  f"left y_mean={a['left_y_mean']:.3f}")

    # ── Overall recommendation ────────────────────────────────────────────── #
    print(f"\n{'=' * 65}")
    print("OVERALL RECOMMENDATION")
    print("=" * 65)
    verdicts = [analysis.get(s, {}).get('verdict') for s in ["left_short", "right_short"]]
    confirmed     = verdicts.count("CONFIRMED")
    not_confirmed = verdicts.count("NOT_CONFIRMED")
    insufficient  = verdicts.count("INSUFFICIENT_DATA")
    partial       = verdicts.count("PARTIAL")

    if confirmed == 2:
        print(f"\n  TABLE_NEAR = {TABLE_NEAR_CURRENT} — KEEP IT.")
        print("  Both short skills land measurably closer to the net than deep skills.")
        overall = "keep"
    elif not_confirmed >= 1 and partial == 0 and confirmed == 0:
        print(f"\n  TABLE_NEAR = {TABLE_NEAR_CURRENT} — TOO AGGRESSIVE.")
        print(f"  Recommend: raise to TABLE_NEAR = {TABLE_NEAR_CANDIDATE}.")
        print("  The PPO (trained on deep targets only) cannot reliably")
        print("  aim at 1.65m with the current policy. A shallower target")
        f"  ({TABLE_NEAR_CANDIDATE}m) is closer to the deep target's distribution"
        print("  and should be more achievable without retraining.")
        overall = "raise"
    elif partial >= 1 or (confirmed == 1 and not_confirmed == 1):
        print(f"\n  TABLE_NEAR = {TABLE_NEAR_CURRENT} — MARGINAL.")
        print("  Short skills do land slightly closer to net but not reliably.")
        print(f"  Consider raising to {TABLE_NEAR_CANDIDATE} as a safer default.")
        overall = "marginal"
    else:
        print(f"\n  INCONCLUSIVE — not enough contact data.")
        print("  Run with larger --trials (40) and --steps (800).")
        overall = "inconclusive"

    # ── Save results ──────────────────────────────────────────────────────── #
    all_results = {
        'config': {
            'trials':             args.trials,
            'steps':              args.steps,
            'TABLE_NEAR_current': TABLE_NEAR_CURRENT,
            'TABLE_NEAR_candidate': TABLE_NEAR_CANDIDATE,
            'ppo_model':          PPO_MODEL_PATH,
        },
        'skill_results': skill_results,
        'analysis':      analysis,
        'overall':       overall,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  JSON saved to: {args.output_json}")

    save_csv(skill_results, args.output_csv)
    print(f"  CSV  saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
