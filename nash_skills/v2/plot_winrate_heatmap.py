"""
5×5 skill-pair win-rate heatmap for the Nash potential pipeline.

Usage (standalone):
    venv/bin/python nash_skills/v2/plot_winrate_heatmap.py
    venv/bin/python nash_skills/v2/plot_winrate_heatmap.py --input data/rallies_5skill_v2.pkl
    venv/bin/python nash_skills/v2/plot_winrate_heatmap.py --output logs/my_heatmap.png

Rows  = ego skill   (player 1, the row-chooser)
Cols  = opponent skill (player 2)
Color = ego win rate among DECIDED rallies (NaN cells shown in grey = no data)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = ["build_counts_matrix", "build_winrate_matrix", "plot_heatmap"]

_DEFAULT_INPUT  = "data/rallies_5skill_v2.pkl"
_DEFAULT_OUTPUT = "logs/winrate_heatmap.png"


def build_counts_matrix(
    rallies: list,
    skills: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Count ego wins and total decided rallies for every skill pair.

    Returns
    -------
    ego_wins : (N, N) int array  — number of rallies where player 1 won
    total    : (N, N) int array  — total decided rallies (ego + opp wins)

    Row = ego skill index, col = opponent skill index.
    Truncated rallies (winner == 0 or None) are excluded from both counts.
    """
    n = len(skills)
    skill_idx = {s: i for i, s in enumerate(skills)}
    ego_wins = np.zeros((n, n), dtype=np.int32)
    total    = np.zeros((n, n), dtype=np.int32)

    for rally in rallies:
        winner = rally.get("winner")
        if winner not in (1, 2):
            continue
        r = skill_idx.get(rally.get("skill1"))
        c = skill_idx.get(rally.get("skill2"))
        if r is None or c is None:
            continue
        total[r, c] += 1
        if winner == 1:
            ego_wins[r, c] += 1

    return ego_wins, total


def build_winrate_matrix(
    rallies: list,
    skills: list[str],
) -> np.ndarray:
    """
    Build an N×N ego win-rate matrix.

    Cells with no decided rallies are NaN.
    Row = ego skill index, col = opponent skill index.
    """
    ego_wins, total = build_counts_matrix(rallies, skills)
    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = np.where(total > 0, ego_wins / total, np.nan).astype(np.float64)
    return matrix


def plot_heatmap(
    matrix: np.ndarray,
    counts: np.ndarray,
    skills: list[str],
    output_path: str = _DEFAULT_OUTPUT,
    title: str = "Ego Win Rate by Skill Matchup",
) -> None:
    """
    Render and save the 5×5 win-rate heatmap.

    Parameters
    ----------
    matrix      : (N, N) float array from build_winrate_matrix — values in [0,1] or NaN
    counts      : (N, N) int array of total decided rallies (for cell annotation)
    skills      : ordered list of skill names (row/col labels)
    output_path : file path to save the PNG
    title       : figure title
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    n = len(skills)
    fig, ax = plt.subplots(figsize=(8, 7))

    # Grey mask for NaN cells
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#cccccc")

    im = ax.imshow(masked, vmin=0.0, vmax=1.0, cmap=cmap, aspect="auto")

    # Colour bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Ego Win Rate", fontsize=11)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

    # Axis labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(skills, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(skills, fontsize=10)
    ax.set_xlabel("Opponent Skill", fontsize=12)
    ax.set_ylabel("Ego Skill", fontsize=12)
    ax.set_title(title, fontsize=13, pad=12)

    # Cell annotations: "XX%\n(n=K)"
    for r in range(n):
        for c in range(n):
            val = matrix[r, c]
            n_decided = int(counts[r, c])
            if np.isnan(val):
                text = "N/A"
                color = "#666666"
            else:
                text = f"{val:.0%}\n(n={n_decided})"
                color = "white" if (val < 0.25 or val > 0.75) else "black"
            ax.text(c, r, text, ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap → {output_path}")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Plot 5×5 skill win-rate heatmap")
    parser.add_argument("--input",  default=_DEFAULT_INPUT,
                        help="Path to rallies pickle file")
    parser.add_argument("--output", default=_DEFAULT_OUTPUT,
                        help="Output PNG path")
    parser.add_argument("--title",  default="Ego Win Rate by Skill Matchup")
    args = parser.parse_args()

    from nash_skills.skills import SKILL_NAMES

    with open(args.input, "rb") as f:
        rallies = pickle.load(f)
    print(f"Loaded {len(rallies)} rallies from {args.input}")

    matrix = build_winrate_matrix(rallies, SKILL_NAMES)
    _, total = build_counts_matrix(rallies, SKILL_NAMES)

    decided = int(total.sum())
    coverage = np.sum(~np.isnan(matrix))
    print(f"Decided rallies: {decided}  |  pairs with data: {coverage}/{len(SKILL_NAMES)**2}")

    plot_heatmap(matrix, total, SKILL_NAMES,
                 output_path=args.output, title=args.title)


if __name__ == "__main__":
    _main()
