"""
Collect runs_paper/ (scripts/launch_paper_runs.sh + launch_paper_eval.sh) into markdown
tables: Table 1, ablations, alpha statistics, simulated IBR with computation time,
head-to-heads and the preparation checks. Missing files are skipped.

    python scripts/summarize_paper_runs.py --root runs_paper > summary.md
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import glob
import json

import numpy as np

PROFILES = ("aggressive", "moderate_aggressive")


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def table1_row(result):
    """(static mean, static min, random, Q-IBR) from an eval_table1 JSON."""
    s = result["summary"]
    cells = s["cells"]
    get = lambda k: cells[k]["win"] if k in cells else float("nan")
    return s.get("static_mean", float("nan")), s.get("static_min", float("nan")), get("random"), get("ibr")


def mean_sd(values):
    v = np.array([x for x in values if x == x], dtype=float)
    if v.size == 0:
        return "—"
    return f"{v.mean():.3f}" if v.size == 1 else f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"


def _seed_rows(root, pattern):
    return [table1_row(r) for r in (_load(p) for p in sorted(glob.glob(os.path.join(root, pattern)))) if r]


def _single_win(root, pattern, opponent_prefix):
    """Mean win rate of the method vs opponents whose label starts with opponent_prefix, per file."""
    out = []
    for p in sorted(glob.glob(os.path.join(root, pattern))):
        r = _load(p)
        if r:
            wins = [c for k, c in r["summary"]["cells"].items() if k.startswith(opponent_prefix)]
            if wins:
                out.append(sum(c["wins"] for c in wins) / max(1, sum(c["decided"] for c in wins)))
    return out


def print_table1(root, profile):
    d = os.path.join(root, profile)
    print(f"\n### Table 1 — {profile} (mean ± sd over seeds)\n")
    print("| Method | Static mean | Static min | Random | IBR (Q) | IBR (sim) | Other proposed |")
    print("|---|---|---|---|---|---|---|")
    rows = [
        ("α-potential, PGA γ0.99", "table1/pga_g099_s?.json", "ibr_sim/pga_g099_s?.json", "h2h/pga_vs_selfplay_s?.json", "selfplay:"),
        ("Self-play, sampled", "table1/selfplay_g099_s?.json", "ibr_sim/selfplay_g099_s?.json", None, None),
        ("Self-play, argmax", "table1/selfplay-argmax_g099_s?.json", "ibr_sim/selfplay-argmax_g099_s?.json", None, None),
    ]
    h2h_pga = _single_win(d, "h2h/pga_vs_selfplay_s?.json", "selfplay:")
    h2h_pga_arg = _single_win(d, "h2h/pga_vs_selfplay_s?.json", "selfplay-argmax:")
    for name, t1, ibr, _h, _p in rows:
        seeds = _seed_rows(d, t1)
        cols = [mean_sd([s[i] for s in seeds]) for i in range(4)]
        sim = mean_sd(_single_win(d, ibr, "ibr-sim"))
        if name.startswith("α"):
            other = f"{mean_sd(h2h_pga)} vs sampled, {mean_sd(h2h_pga_arg)} vs argmax"
        elif "sampled" in name:
            other = mean_sd([1 - x for x in h2h_pga])
        else:
            other = mean_sd([1 - x for x in h2h_pga_arg])
        print(f"| {name} | {' | '.join(cols)} | {sim} | {other} |")


def print_ablations(root, profile):
    d = os.path.join(root, profile)
    print(f"\n### α-potential ablations — {profile} (seed 0 unless noted)\n")
    print("| Phi run | train rallies | Q1 MSE (held-out) | α max | α p95 | α median | best epoch | static mean | static min | random | IBR (Q) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for phi_dir in sorted(glob.glob(os.path.join(d, "phi", "*/"))):
        name = os.path.basename(phi_dir.rstrip("/"))
        h = _load(os.path.join(phi_dir, "heldout_attack_defend.json"))
        t = _load(os.path.join(d, "table1", f"pga_{name}.json"))
        if not h:
            continue
        res = h.get("heldout_residuals", {})
        row = table1_row(t) if t else (float("nan"),) * 4
        print(f"| {name} | {h['n_train_rallies']} | {h.get('heldout_q1_mse', float('nan')):.3f} | "
              f"{res.get('max_abs', float('nan')):.2f} | {res.get('p95_abs', float('nan')):.2f} | "
              f"{res.get('median_abs', float('nan')):.2f} | {h.get('phi_best_epoch')} | "
              + " | ".join(f"{x:.3f}" for x in row) + " |")
    print(f"\n### Self-play discount ablation — {profile} (sampled, seed 0)\n")
    print("| γ | static mean | static min | random |")
    print("|---|---|---|---|")
    for tag, f in (("0.99", "selfplay_g099_s0"), ("0.9", "selfplay_g09_s0"), ("0.7", "selfplay_g07_s0")):
        t = _load(os.path.join(d, "table1", f + ".json"))
        if t:
            row = table1_row(t)
            print(f"| {tag} | {row[0]:.3f} | {row[1]:.3f} | {row[2]:.3f} |")


def print_ibr_cost(root, profile):
    d = os.path.join(root, profile)
    times = {"method": [], "ibr": []}
    for p in glob.glob(os.path.join(d, "ibr_sim", "pga_*.json")):
        r = _load(p)
        if r:
            t = r["summary"]["timing"]
            times["method"].append(t["method_mean_s"])
            times["ibr"].append(t["by_opponent"].get("ibr-sim", {}).get("opponent_mean_s", float("nan")))
    if times["method"]:
        print(f"\nStrategic computation per decision ({profile}): α-potential PGA "
              f"{np.mean(times['method']) * 1000:.0f} ms, simulated IBR {np.nanmean(times['ibr']):.1f} s")


def print_checks(root):
    print("\n### Does preparation matter? (one-sided, matched − mismatched preparation)\n")
    print("| profile | mechanism | gate (mech − off) | matched − mismatched |")
    print("|---|---|---|---|")
    for profile in PROFILES:
        r = _load(os.path.join(root, "checks", f"one_sided_{profile}.json"))
        if not r:
            continue
        for mech, s in r["summary"]["mechanisms"].items():
            g, c = s["gate"], s["matched_minus_mismatched"]
            print(f"| {profile} | {mech} | {g[0]:+.3f} [{g[1]:+.3f}, {g[2]:+.3f}] | "
                  f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}] |")
    print("\n### Rally length (both players random choices)\n")
    print("| profile | condition | truncated | mean steps | returns / rally |")
    print("|---|---|---|---|---|")
    for profile in PROFILES:
        r = _load(os.path.join(root, "checks", f"both_sided_{profile}.json"))
        if r:
            for key, row in r["summary"].items():
                print(f"| {profile} | {key} | {row['trunc']:.3f} | {row['steps_mean']:.0f} | "
                      f"{row['returns_per_rally']:.2f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_paper")
    args = ap.parse_args()
    print_checks(args.root)
    for profile in PROFILES:
        print_table1(args.root, profile)
        print_ibr_cost(args.root, profile)
        print_ablations(args.root, profile)


if __name__ == "__main__":
    main()
