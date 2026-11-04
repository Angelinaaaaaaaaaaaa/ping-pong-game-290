"""
Updated Table 1: alpha-potential with early-stopped Q (runs_paper/alpha_fixed, 3 seeds) next to
the self-play rows (runs_paper/<profile>/table1, ibr_sim, h2h). Win rates count decided rallies;
the "(½)" columns count an unresolved rally as half a point.

    python scripts/summarize_table1_fixed.py --root analysis/server_results/paper_runs
"""
import argparse
import glob
import json
import os

import numpy as np

PROFILES = ("aggressive", "moderate_aggressive")


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def static_stats(result):
    cells = result["summary"]["cells"]
    st = [v for k, v in cells.items() if "/" in k]
    dec = [v["win"] for v in st]
    half = [(v["wins"] + 0.5 * v["truncated"]) / max(1, v["decided"] + v["truncated"]) for v in st]
    unresolved = sum(v["truncated"] for v in st) / max(1, sum(v["decided"] + v["truncated"] for v in st))
    get = lambda k: cells[k]["win"] if k in cells else float("nan")
    return {"static_mean": np.mean(dec), "static_min": np.min(dec), "static_mean_half": np.mean(half),
            "static_min_half": np.min(half), "random": get("random"), "ibr_q": get("ibr"), "unresolved": unresolved}


def win_vs(result, prefix):
    cells = [v for k, v in result["summary"]["cells"].items() if k.startswith(prefix)]
    n = sum(v["decided"] for v in cells)
    return sum(v["wins"] for v in cells) / n if n else float("nan")


def ms(values):
    v = np.array([x for x in values if x == x], dtype=float)
    if v.size == 0:
        return "—"
    return f"{v.mean():.3f}" if v.size == 1 else f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"


def rows_for(root, profile):
    d, f = os.path.join(root, profile), os.path.join(root, "alpha_fixed", profile)
    out = []
    alpha = [static_stats(r) for r in (_load(os.path.join(f, f"table1_s{s}.json")) for s in range(3)) if r]
    ibr = [win_vs(r, "ibr-sim") for r in (_load(os.path.join(f, f"ibr_sim_s{s}.json")) for s in range(3)) if r]
    h2h = [_load(os.path.join(f, f"h2h_s{s}.json")) for s in range(3)]
    h2h = [r for r in h2h if r]
    vs_sampled = [win_vs(r, "selfplay:") for r in h2h]
    vs_argmax = [win_vs(r, "selfplay-argmax:") for r in h2h]
    out.append(("α-potential (fixed Q)", alpha, ibr, f"{ms(vs_sampled)} / {ms(vs_argmax)}"))
    for kind, label, others in (("selfplay", "Self-play sampled", vs_sampled), ("selfplay-argmax", "Self-play argmax", vs_argmax)):
        rows = [static_stats(r) for r in (_load(p) for p in sorted(glob.glob(os.path.join(d, "table1", f"{kind}_g099_s?.json")))) if r]
        sim = [win_vs(r, "ibr-sim") for r in (_load(p) for p in sorted(glob.glob(os.path.join(d, "ibr_sim", f"{kind}_g099_s?.json")))) if r]
        out.append((label, rows, sim, ms([1 - x for x in others])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="analysis/server_results/paper_runs")
    args = ap.parse_args()
    for profile in PROFILES:
        print(f"\n### {profile}\n")
        print("| Method | Static mean | Static min | Static mean / min (½) | Random | IBR (Q-based) | IBR (simulated) | vs other method | Unresolved vs static |")
        print("|---|---|---|---|---|---|---|---|---|")
        for label, stats, sim, other in rows_for(args.root, profile):
            col = lambda k: ms([s[k] for s in stats])
            half = f"{np.mean([s['static_mean_half'] for s in stats]):.3f} / {np.mean([s['static_min_half'] for s in stats]):.3f}" if stats else "—"
            print(f"| {label} | {col('static_mean')} | {col('static_min')} | {half} | {col('random')} | {col('ibr_q')} | "
                  f"{ms(sim)} | {other} | {np.mean([s['unresolved'] for s in stats]):.3f} |")


if __name__ == "__main__":
    main()
