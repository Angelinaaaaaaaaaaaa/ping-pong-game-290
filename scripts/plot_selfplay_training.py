"""
Self-play training curves from nash_skills/v2/selfplay_per_player.py logs (one CSV per seed):
rally length in exchanges, policy entropy, and the share of each target / preparation chosen,
mean over seeds with a trailing rolling window.

    python scripts/plot_selfplay_training.py --logs runs_paper/selfplay_nochoice/aggressive_g099_s*.csv --out fig
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import csv

import numpy as np

DEFEND_ORDER = ("left", "right", "center")  # column order of the logged defend_usage counts


def rolling_mean(x, window):
    """Trailing mean over the last `window` values (shorter at the start)."""
    c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
    idx = np.arange(1, len(x) + 1)
    start = np.maximum(idx - window, 0)
    return (c[idx] - c[start]) / (idx - start)


def read_log(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))

    def share(key):
        counts = np.array([[float(v) for v in r[key].split()] for r in rows])
        return counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    prep = share("defend_usage")
    return {"iter": np.array([int(r["iter"]) for r in rows]),
            "exchanges": np.array([float(r["mean_exchanges"]) for r in rows]),
            "entropy": np.array([float(r["entropy"]) for r in rows]),
            "target_share": share("attack_usage"), "prep_share": prep,
            "prep_center": prep[:, DEFEND_ORDER.index("center")]}


def plot(paths, out_stem, window=25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nash_skills.skills import SKILL_NAMES
    logs = [read_log(p) for p in paths]
    n = min(len(c["iter"]) for c in logs)
    it = logs[0]["iter"][:n]
    mean = lambda key: np.mean([c[key][:n] for c in logs], axis=0)
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.6))
    for ax, key, label in ((axes[0], "exchanges", "exchanges per rally"), (axes[1], "entropy", "policy entropy (nats)")):
        ax.plot(it, rolling_mean(mean(key), window), lw=1.2)
        ax.set_ylabel(label, fontsize=8)
    for ax, key, names, title in ((axes[2], "target_share", SKILL_NAMES, "attack targets chosen"),
                                  (axes[3], "prep_share", DEFEND_ORDER, "preparations chosen")):
        share = mean(key)
        for k, name in enumerate(names):
            ax.plot(it, rolling_mean(share[:, k], window), lw=1.1, label=name)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=6, frameon=False, ncol=2)
    for ax in axes:
        ax.set_xlabel("iteration (32 rallies each)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle(f"Self-play training, mean of {len(logs)} seeds ({window}-iteration rolling mean)", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=200)
    plt.close(fig)
    return f"{out_stem}.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--window", type=int, default=25)
    args = ap.parse_args()
    print(plot(args.logs, args.out, args.window))


if __name__ == "__main__":
    main()
