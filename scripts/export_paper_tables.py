"""
Paper-format tables (L4DC draft Sec. 5) from the Design B server results:

  paper_table1.csv                Table 1: method x {static mean, static minimum, random,
                                  IBR, other proposed}, mean (min-max) over seeds
  paper_table1_static_matrix.csv  the complete 5x3 static-opponent matrix per method
  paper_table_ablations.csv       Sec. 5.3: data volume, Phi loss, theta mode, deployment

Inputs are table1_summary.csv / table1_static_matrix.csv (scripts/export_table1_csv.py)
and runs/phi/*/heldout_attack_defend.json.

    python scripts/export_paper_tables.py --results analysis/server_results/designb_preparation
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

ROWS = [  # (Table 1 label, deployment, summary-CSV method label, head-to-head column, invert)
    ("alpha-potential", "PGA (Eq. 18)", "alpha-potential (PGA, Eq. 18)", ("self-play (sampled)", "vs_nash_p_pga"), True),
    ("Self-play RL", "sampled", "self-play (sampled)", ("self-play (sampled)", "vs_nash_p_pga"), False),
    ("alpha-potential (ablation)", "argmax", "alpha-potential (argmax)", ("self-play (sampled)", "vs_nash_p_hard"), True),
    ("Self-play RL (ablation)", "argmax", "self-play (argmax)", ("self-play (argmax)", "vs_nash_p_pga"), False),
]
DEFENDS = ("left", "right", "center")
ATTACKS = ("left", "left_short", "center_safe", "right_short", "right")
PHI_RUNS = [("data volume", "25% of train rallies", "vol25"), ("data volume", "50% of train rallies", "vol50"),
            ("data volume", "100% of train rallies, seed 0", "main_s0"),
            ("discount", "gamma 0.7, seed 0", "main_s0"), ("discount", "gamma 0.99, seed 0", "gamma099_s0"),
            ("Phi loss", "smooth max (LSE)", "main_s0"), ("Phi loss", "MSE", "mse_mixed"),
            ("training data", "pure + mixed theta", "main_s0"), ("training data", "pure only", "grid_lse_pure")]


def mean_range(values):
    values = [v for v in values if v is not None]
    if not values:
        return ""
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{np.mean(values):.3f} ({min(values):.3f}-{max(values):.3f})"


def complement(values):
    return [1.0 - v for v in values]


def per_seed_runs(rows):
    groups = defaultdict(list)
    for r in rows:
        if not r["seed"].startswith("mean"):
            groups[r["method"]].append(r)
    return groups


def _floats(runs, key):
    return [float(r[key]) for r in runs if r.get(key)]


def hardest_opponent(matrix_rows, method):
    cells = [(float(r["win"]), f"{r['opp_attack']}/{r['opp_defend']}")
             for r in matrix_rows if r["method"] == method and r["seed"] == "mean"]
    return min(cells)[1] if cells else ""


def table1(groups, matrix_rows):
    out = []
    for label, deployment, method, (h2h_method, h2h_key), invert in ROWS:
        runs = groups.get(method, [])
        h2h = _floats(groups.get(h2h_method, []), h2h_key)
        out.append({
            "Method": label, "Deployment": deployment, "Seeds": len(runs),
            "Static mean": mean_range(_floats(runs, "static_mean")),
            "Static minimum": mean_range(_floats(runs, "static_min")),
            "Hardest static opponent (seed mean)": hardest_opponent(matrix_rows, method),
            "Random": mean_range(_floats(runs, "random")),
            "IBR (Q-based)": mean_range(_floats(runs, "ibr")),
            "Other proposed": mean_range(complement(h2h) if invert else h2h),
        })
    return out


def static_matrix(matrix_rows):
    out = []
    for _label, deployment, method, _h2h, _inv in ROWS:
        cells = {(r["opp_attack"], r["opp_defend"]): float(r["win"])
                 for r in matrix_rows if r["method"] == method and r["seed"] == "mean"}
        for a in ATTACKS:
            out.append({"Method": method, "Opponent attack target": a,
                        **{f"prepares {d}": f"{cells[(a, d)]:.3f}" if (a, d) in cells else "" for d in DEFENDS}})
    return out


def ablations(results_dir, groups):
    out = []
    for study, setting, run in PHI_RUNS:
        path = os.path.join(results_dir, "runs", "phi", run, "heldout_attack_defend.json")
        if not os.path.exists(path):
            continue
        r = json.load(open(path))
        h = r["heldout_residuals"]
        out.append({"Study": study, "Setting": setting, "Train rallies": r["n_train_rallies"],
                    "Held-out Q MSE (P1/P2)": f"{r['heldout_q1_mse']:.3f}/{r['heldout_q2_mse']:.3f}",
                    "alpha = max|e|": f"{h['max_abs']:.3f}", "p95 |e|": f"{h['p95_abs']:.3f}",
                    "mean |e|": f"{h['mean_abs']:.3f}", "Static mean": "", "Random": "", "IBR (Q-based)": ""})
    for method, setting in (("alpha-potential (PGA, Eq. 18)", "alpha-potential, mixed (PGA)"),
                            ("alpha-potential (argmax)", "alpha-potential, argmax"),
                            ("self-play (sampled)", "self-play, sampled"),
                            ("self-play (argmax)", "self-play, argmax")):
        runs = groups.get(method, [])
        out.append({"Study": "deployment (sampled vs argmax)", "Setting": setting,
                    "Static mean": mean_range(_floats(runs, "static_mean")),
                    "Random": mean_range(_floats(runs, "random")),
                    "IBR (Q-based)": mean_range(_floats(runs, "ibr"))})
    return out


def _write(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="folder with table1_summary.csv, table1_static_matrix.csv, runs/")
    args = ap.parse_args()
    summary = list(csv.DictReader(open(os.path.join(args.results, "table1_summary.csv"))))
    matrix = list(csv.DictReader(open(os.path.join(args.results, "table1_static_matrix.csv"))))
    groups = per_seed_runs(summary)
    for name, rows in (("paper_table1.csv", table1(groups, matrix)),
                       ("paper_table1_static_matrix.csv", static_matrix(matrix)),
                       ("paper_table_ablations.csv", ablations(args.results, groups))):
        _write(os.path.join(args.results, name), rows)
        print(f"wrote {name} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
