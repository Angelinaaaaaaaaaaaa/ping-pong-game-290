"""
Tables in Kexin's V3 format (one row per decision rule) from scripts/launch_kexin_format.sh:
Static mean | Static minimum | Random | IBR (Q-based) | Other proposed (vs our self-play, sampled).
Means over seeds; static minimum = per-seed minimum over the 15 static opponents, averaged.

    python scripts/summarize_kexin_format.py --root analysis/server_results/paper_runs
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import os

import numpy as np


def _static_labels():
    from nash_skills.defend_positions import DEFEND_POSITION_NAMES
    from nash_skills.skills import SKILL_NAMES
    return {f"{a}/{d}" for a in SKILL_NAMES for d in DEFEND_POSITION_NAMES}


STATIC_LABELS = _static_labels()

RULES = (("nash-q", "Nash-Q"), ("br-q", "BR-Q"), ("nash-p-ibr", "Potential IBR"), ("nash-p-br", "Potential BR"),
         ("nash-p-adaptive", "Potential Adaptive Match"), ("nash-p-minimax", "Potential Minimax"),
         ("nash-p-pga", "α-potential PGA (paper)"))
PROFILES = ("aggressive", "moderate_aggressive")
DATA = (("base", "4.7k rallies collected (3.4k train)"), ("x10", "47.5k rallies collected (34k train)"))


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def row_stats(result):
    cells = result["summary"]["cells"]
    static = [v["win"] for k, v in cells.items() if k in STATIC_LABELS]
    selfplay = [v for k, v in cells.items() if k.startswith("selfplay:")]
    return {"static_mean": float(np.mean(static)) if static else float("nan"),
            "static_min": float(np.min(static)) if static else float("nan"),
            "random": cells.get("random", {}).get("win", float("nan")),
            "ibr": cells.get("ibr", {}).get("win", float("nan")),
            "other": selfplay[0]["win"] if selfplay else float("nan")}


def _pga_base(root, profile, seed):
    """Base-data PGA row from runs_paper/alpha_fixed (table1 + head-to-head files)."""
    t = _load(os.path.join(root, "alpha_fixed", profile, f"table1_s{seed}.json"))
    h = _load(os.path.join(root, "alpha_fixed", profile, f"h2h_s{seed}.json"))
    if not t:
        return None
    stats = row_stats(t)
    if h:
        stats["other"] = row_stats(h)["other"]
    return stats


def collect(root, profile, data, rule):
    out = []
    for s in range(3):
        if rule == "nash-p-pga" and data == "base":
            stats = _pga_base(root, profile, s)
        else:
            r = _load(os.path.join(root, "kexin_format", profile, data, f"{rule}_s{s}.json"))
            stats = row_stats(r) if r else None
        if stats:
            out.append(stats)
    return out


def table(root, profile, data):
    lines = ["Method                    | Static mean | Static minimum | Random | IBR   | Other proposed",
             "--------------------------+-------------+----------------+--------+-------+---------------"]
    for rule, label in RULES:
        seeds = collect(root, profile, data, rule)
        if not seeds:
            continue
        m = {k: 100 * np.nanmean([s[k] for s in seeds]) for k in seeds[0]}
        lines.append(f"{label:<25} | {m['static_mean']:>10.1f}% | {m['static_min']:>13.1f}% | "
                     f"{m['random']:>5.1f}% | {m['ibr']:>4.1f}% | {m['other']:>5.1f}%" + ("" if len(seeds) == 3 else f"  ({len(seeds)} seeds)"))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="analysis/server_results/paper_runs")
    args = ap.parse_args()
    for profile in PROFILES:
        for data, label in DATA:
            print(f"\n{profile}, {label}, γ 0.99, 3 seeds, 100 rallies per side per matchup")
            print(table(args.root, profile, data))


if __name__ == "__main__":
    main()
