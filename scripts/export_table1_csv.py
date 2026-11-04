"""
Export Table 1 JSONs (from scripts/eval_table1_attack_defend.py) to CSV:

  table1_summary.csv        one row per run (method x seed) plus a mean-over-seeds row
                            per method: static mean/min, random, IBR, head-to-heads,
                            truncations, rally length
  table1_static_matrix.csv  the 5x3 static-opponent matrix, long format, per run and mean

    python scripts/export_table1_csv.py --out-dir <dir> runs/table1/*.json

Rally length in legal returns is estimated from steps with the fit on the
preparation training data (steps ~= 96 + 69 x returns, correlation 0.97).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import csv
import json
import re
from collections import defaultdict

import numpy as np

from nash_skills.skills import SKILL_NAMES
from nash_skills.defend_positions import DEFEND_POSITION_NAMES

STEPS_INTERCEPT, STEPS_PER_RETURN = 96.0, 69.0
TWO_RETURNS_STEPS = STEPS_INTERCEPT + 2.5 * STEPS_PER_RETURN
HEADLINE = ("static_mean", "static_min", "random", "ibr")
SUMMARY_FIELDS = ["method", "seed", "static_mean", "static_mean_ci_lo", "static_mean_ci_hi", "static_mean_range",
                  "static_min", "static_min_range", "static_min_opponent", "random", "random_ci_lo", "random_ci_hi",
                  "random_range", "ibr", "ibr_ci_lo", "ibr_ci_hi", "ibr_range", "vs_nash_p_pga", "vs_nash_p_hard",
                  "decided", "truncated", "median_rally_steps", "share_le_2_returns_est", "source"]


def method_label(method):
    iteration = re.search(r"_iter(\d+)", method)
    if method.startswith("selfplay-argmax:"):
        return "self-play (argmax" + (f", iter {iteration.group(1)})" if iteration else ")")
    if method.startswith("selfplay:"):
        return "self-play (sampled" + (f", iter {iteration.group(1)})" if iteration else ")")
    return {"nash-p-pga": "alpha-potential (PGA, Eq. 18)", "nash-p-hard": "alpha-potential (argmax)"}.get(method, method)


def run_seed(method, model_dir):
    for text, pattern in ((method, r"seed(\d+)"), (model_dir or "", r"_s(\d+)$")):
        found = re.search(pattern, text)
        if found:
            return found.group(1)
    return ""


def _cell(cells, key, field="win"):
    return cells[key][field] if key in cells else None


def summary_row(path):
    with open(path) as f:
        run = json.load(f)
    s, cells = run["summary"], run["summary"]["cells"]
    lengths = np.array([x for r in run["rows"] for x in r["rally_lengths"]], dtype=float)
    return {
        "method": method_label(run["args"]["method"]),
        "seed": run_seed(run["args"]["method"], run["args"].get("model_dir")),
        "static_mean": s.get("static_mean"), "static_mean_ci_lo": (s.get("static_mean_ci") or [None])[0],
        "static_mean_ci_hi": (s.get("static_mean_ci") or [None, None])[1],
        "static_min": s.get("static_min"), "static_min_opponent": s.get("static_min_opponent"),
        "random": _cell(cells, "random"), "random_ci_lo": (cells.get("random", {}).get("ci") or [None])[0],
        "random_ci_hi": (cells.get("random", {}).get("ci") or [None, None])[1],
        "ibr": _cell(cells, "ibr"), "ibr_ci_lo": (cells.get("ibr", {}).get("ci") or [None])[0],
        "ibr_ci_hi": (cells.get("ibr", {}).get("ci") or [None, None])[1],
        "vs_nash_p_pga": _cell(cells, "nash-p-pga"), "vs_nash_p_hard": _cell(cells, "nash-p-hard"),
        "decided": sum(c["decided"] for c in cells.values()),
        "truncated": sum(c["truncated"] for c in cells.values()),
        "median_rally_steps": float(np.median(lengths)) if lengths.size else None,
        "share_le_2_returns_est": float(np.mean(lengths <= TWO_RETURNS_STEPS)) if lengths.size else None,
        "source": os.path.basename(path),
    }, cells


def mean_row(method, rows):
    out = {"method": method, "seed": f"mean ({len(rows)} seeds)"}
    for key in ("static_mean", "static_min", "random", "ibr", "vs_nash_p_pga", "vs_nash_p_hard",
                "median_rally_steps", "share_le_2_returns_est"):
        values = [r[key] for r in rows if r[key] is not None]
        if values:
            out[key] = float(np.mean(values))
            if key in HEADLINE:
                out[f"{key}_range"] = f"{min(values):.3f}-{max(values):.3f}"
    out["decided"] = sum(r["decided"] for r in rows)
    out["truncated"] = sum(r["truncated"] for r in rows)
    return out


def export(paths, out_dir):
    by_method, matrix = defaultdict(list), []
    for path in paths:
        row, cells = summary_row(path)
        by_method[row["method"]].append(row)
        for a in SKILL_NAMES:
            for d in DEFEND_POSITION_NAMES:
                c = cells.get(f"{a}/{d}")
                if c:
                    matrix.append({"method": row["method"], "seed": row["seed"], "opp_attack": a, "opp_defend": d,
                                   "win": c["win"], "decided": c["decided"], "truncated": c["truncated"]})
    summary = []
    for method, rows in by_method.items():
        summary.extend(sorted(rows, key=lambda r: r["seed"]))
        if len(rows) > 1:
            summary.append(mean_row(method, rows))
    for method in by_method:
        groups = defaultdict(list)
        for m in matrix:
            if m["method"] == method:
                groups[(m["opp_attack"], m["opp_defend"])].append(m)
        if len({m["seed"] for ms in groups.values() for m in ms}) > 1:
            for (a, d), ms in groups.items():
                matrix.append({"method": method, "seed": "mean", "opp_attack": a, "opp_defend": d,
                               "win": float(np.mean([m["win"] for m in ms])),
                               "decided": sum(m["decided"] for m in ms), "truncated": sum(m["truncated"] for m in ms)})
    os.makedirs(out_dir, exist_ok=True)
    _write(os.path.join(out_dir, "table1_summary.csv"), SUMMARY_FIELDS, summary)
    _write(os.path.join(out_dir, "table1_static_matrix.csv"),
           ["method", "seed", "opp_attack", "opp_defend", "win", "decided", "truncated"], matrix)
    return summary


def _write(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items() if k in fields})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    for row in export(args.runs, args.out_dir):
        print(f"{row['method']:<36} {row['seed']:<14} static mean {row.get('static_mean') or float('nan'):.3f}  "
              f"min {row.get('static_min') or float('nan'):.3f}  random {row.get('random') or float('nan'):.3f}  "
              f"ibr {row.get('ibr') or float('nan'):.3f}")


if __name__ == "__main__":
    main()
