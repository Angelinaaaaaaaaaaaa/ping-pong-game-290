"""
Tables, CSVs and plots for the alpha-potential ablations (scripts/launch_alpha_ablations.sh):
win rate vs discount factor and vs training rallies, mean +- sd over seeds, with the
model-quality diagnostics (held-out Q error, alpha percentiles, Phi vs constant potential).

    python scripts/summarize_alpha_ablations.py --root runs_paper/ablations_qval --out-dir OUT
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


def _static_labels():
    from nash_skills.defend_positions import DEFEND_POSITION_NAMES
    from nash_skills.skills import SKILL_NAMES
    return {f"{a}/{d}" for a in SKILL_NAMES for d in DEFEND_POSITION_NAMES}


STATIC_LABELS = _static_labels()

PROFILES = ("aggressive", "moderate_aggressive")
METRICS = ("static_mean", "static_min", "static_mean_half", "static_min_half", "random", "heldout_q1_mse",
           "alpha_p95", "alpha_median",
           "alpha_mean", "alpha_mean_constant", "trunc", "n_train_rallies")


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


TABLE_DIR = "table1"  # folder of the evaluation JSONs, next to phi/ (paper_alpha uses "figevals")


def run_metrics(root, profile, name):
    rep = _load(os.path.join(root, profile, "phi", name, "heldout_attack_defend.json"))
    t1 = _load(os.path.join(root, profile, TABLE_DIR, name + ".json"))
    if rep is None:
        return None
    h, c = rep.get("heldout_residuals", {}), rep.get("heldout_residuals_constant_phi", {})
    out = {"heldout_q1_mse": rep.get("heldout_q1_mse"), "alpha_p95": h.get("p95_abs"),
           "alpha_median": h.get("median_abs"), "alpha_mean": h.get("mean_abs"),
           "alpha_mean_constant": c.get("mean_abs"), "n_train_rallies": rep.get("n_train_rallies")}
    if t1:
        s = t1["summary"]
        cells = s["cells"]
        half = [(v["wins"] + 0.5 * v["truncated"]) / max(1, v["decided"] + v["truncated"])
                for k, v in cells.items() if k in STATIC_LABELS]
        out.update(static_mean=s.get("static_mean"), static_min=s.get("static_min"),
                   static_mean_half=float(np.mean(half)) if half else None,
                   static_min_half=float(np.min(half)) if half else None,
                   random=cells.get("random", {}).get("win"),
                   trunc=sum(v["truncated"] for v in cells.values())
                   / max(1, sum(v["truncated"] + v["decided"] for v in cells.values())))
    return out


def group(root, profile, prefix):
    """{x value: [metrics per seed]} for runs named <prefix><x>_s<seed>."""
    out = defaultdict(list)
    for d in glob.glob(os.path.join(root, profile, "phi", f"{prefix}*_s*")):
        name = os.path.basename(d)
        m = re.fullmatch(re.escape(prefix) + r"([0-9.]+)_s(\d+)", name)
        if m and os.path.isdir(d):
            metrics = run_metrics(root, profile, name)
            if metrics:
                out[float(m.group(1))].append(metrics)
    return dict(sorted(out.items()))


def mean_sd(values):
    v = np.array([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, int(v.size)


def table(groups, xlabel):
    lines = [f"| {xlabel} | seeds | static mean | static min | static mean / min (unresolved = ½) | random | "
             "held-out Q MSE | α p95 | α mean (constant Φ) | truncated |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for x, runs in groups.items():
        stats = {k: mean_sd([r.get(k) for r in runs]) for k in METRICS}
        rows.append((x, stats))
        f = lambda k: f"{stats[k][0]:.3f} ± {stats[k][1]:.3f}" if stats[k][2] else "—"
        n_train = stats["n_train_rallies"][0]
        label = f"{int(n_train)}" if xlabel == "train rallies" else f"{x:g}"
        lines.append(f"| {label} | {stats['static_mean'][2]} | {f('static_mean')} | {f('static_min')} | "
                     f"{stats['static_mean_half'][0]:.3f} / {stats['static_min_half'][0]:.3f} | {f('random')} | "
                     f"{f('heldout_q1_mse')} | {f('alpha_p95')} | {stats['alpha_mean'][0]:.3f} ({stats['alpha_mean_constant'][0]:.3f}) | "
                     f"{stats['trunc'][0]:.3f} |")
    return "\n".join(lines), rows


def write_csv(path, rows, xname):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([xname, "seeds"] + [f"{k}_{s}" for k in METRICS for s in ("mean", "sd")])
        for x, stats in rows:
            w.writerow([x, stats["static_mean"][2]] + [v for k in METRICS for v in stats[k][:2]])


def plot(path, series, xlabel, logx=False):
    if not any(series.values()):
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    plotted = False
    for ax, (profile, rows) in zip(axes, series.items()):
        ax.set_title(profile)
        ax.set_xlabel(xlabel)
        for key, label in (("static_mean", "static mean"), ("static_min", "static min"), ("random", "vs random")):
            pts = [(r[1]["n_train_rallies"][0] if xlabel == "train rallies" else r[0], r[1][key][0], r[1][key][1])
                   for r in rows if np.isfinite(r[1][key][0])]
            if not pts:
                continue
            xs, m, s = map(np.array, zip(*pts))
            ax.errorbar(xs, m, yerr=s, marker="o", capsize=3, label=label)
            plotted = True
            if logx:
                ax.set_xscale("log")
                ax.set_xlim(xs.min() / 1.6, xs.max() * 1.6)
        ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    if not plotted:
        plt.close(fig)
        return
    axes[0].set_ylabel("win rate")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--table-dir", default="table1", help="folder of evaluation JSONs next to phi/")
    args = ap.parse_args()
    global TABLE_DIR
    TABLE_DIR = args.table_dir
    os.makedirs(args.out_dir, exist_ok=True)
    md, gamma_series, vol_series = [], {}, {}
    for profile in PROFILES:
        g = group(args.root, profile, "gamma")
        v = group(args.root, profile, "vol")
        t, gamma_series[profile] = table(g, "γ")
        md += [f"\n### Discount factor — {profile}\n", t]
        write_csv(os.path.join(args.out_dir, f"gamma_{profile}.csv"), gamma_series[profile], "gamma")
        t, vol_series[profile] = table(v, "train rallies")
        md += [f"\n### Training data volume (γ 0.99) — {profile}\n", t]
        write_csv(os.path.join(args.out_dir, f"volume_{profile}.csv"), vol_series[profile], "train_rally_limit")
    plot(os.path.join(args.out_dir, "gamma_ablation.png"), gamma_series, "discount factor γ")
    plot(os.path.join(args.out_dir, "volume_ablation.png"), vol_series, "train rallies", logx=True)
    text = "\n".join(md)
    with open(os.path.join(args.out_dir, "ablation_tables.md"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
