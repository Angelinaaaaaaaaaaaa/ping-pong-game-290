"""
Inspect truncation behavior in a collected rally pickle -- pure Python,
no MuJoCo, no model loading.

Reports overall and per-skill-pair truncation rates, average rally length,
and decided-only win rate, to help diagnose whether dataset imbalance or
long-stable-rally behavior comes from particular skill pairs, one player's
policy, or an issue in the collector (meeting note item 3).

Usage
-----
    venv/bin/python nash_skills/v2/inspect_truncated_rallies.py
    venv/bin/python nash_skills/v2/inspect_truncated_rallies.py \\
        --input data/rallies_5skill_v2.pkl --top-k 10
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import pickle
from pathlib import Path
from typing import Optional

__all__ = ["summarize_overall", "build_pair_stats"]

_DEFAULT_INPUT = "data/rallies_5skill_v2.pkl"


def _is_truncated(winner) -> bool:
    return winner not in (1, 2)


def summarize_overall(rallies: list) -> dict:
    """
    Overall decided/truncated counts and truncation rate.

    truncation_rate = truncated / total (None if rallies is empty).
    """
    total = len(rallies)
    truncated = sum(1 for r in rallies if _is_truncated(r.get("winner")))
    decided = total - truncated
    return {
        "total": total,
        "decided": decided,
        "truncated": truncated,
        "truncation_rate": (truncated / total) if total > 0 else None,
    }


def build_pair_stats(rallies: list) -> list:
    """
    Group rallies by (skill1, skill2) and compute per-pair stats.

    Returns a list of dicts, sorted by truncation_rate descending (pairs
    with no rallies are never included -- only pairs actually present in
    the data). Each dict has:
        skill1, skill2, total, decided, truncated, truncation_rate,
        avg_rally_length, win_rate (ego win rate among decided rallies,
        None if no decided rallies for this pair)
    """
    groups: dict = {}
    for r in rallies:
        key = (r.get("skill1"), r.get("skill2"))
        groups.setdefault(key, []).append(r)

    stats = []
    for (s1, s2), group in groups.items():
        total = len(group)
        truncated = sum(1 for r in group if _is_truncated(r.get("winner")))
        decided = total - truncated
        ego_wins = sum(1 for r in group if r.get("winner") == 1)
        avg_len = sum(len(r.get("states", [])) for r in group) / total if total > 0 else 0.0

        stats.append({
            "skill1": s1,
            "skill2": s2,
            "total": total,
            "decided": decided,
            "truncated": truncated,
            "truncation_rate": truncated / total if total > 0 else None,
            "avg_rally_length": avg_len,
            "win_rate": (ego_wins / decided) if decided > 0 else None,
        })

    stats.sort(key=lambda r: r["truncation_rate"], reverse=True)
    return stats


def _fmt_pct(v: Optional[float]) -> str:
    return f"{v:.1%}" if v is not None else "  ---"


def _print_table(stats: list, top_k: int) -> None:
    hdr = (f"{'skill1':<14} {'skill2':<14} {'total':>6} {'trunc':>6} "
           f"{'tr%':>7} {'rally':>7} {'wr':>7}")
    print(hdr)
    print("-" * len(hdr))
    for row in stats[:top_k]:
        print(
            f"{str(row['skill1']):<14} {str(row['skill2']):<14} "
            f"{row['total']:>6} {row['truncated']:>6} "
            f"{_fmt_pct(row['truncation_rate']):>7} "
            f"{row['avg_rally_length']:>7.1f} "
            f"{_fmt_pct(row['win_rate']):>7}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inspect truncation behavior in a rally pickle (no MuJoCo)."
    )
    ap.add_argument("--input", default=_DEFAULT_INPUT,
                    help=f"Rally pickle path (default: {_DEFAULT_INPUT})")
    ap.add_argument("--top-k", type=int, default=10, dest="top_k",
                    help="Number of skill pairs to show, sorted by truncation rate (default: 10)")
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Rally file not found: {args.input}")

    with open(path, "rb") as f:
        rallies = pickle.load(f)

    print(f"Loaded {len(rallies)} rallies from {args.input}\n")

    overall = summarize_overall(rallies)
    print("Overall")
    print("-" * 40)
    print(f"  total rallies     : {overall['total']}")
    print(f"  decided           : {overall['decided']}")
    print(f"  truncated         : {overall['truncated']}")
    print(f"  truncation rate   : {_fmt_pct(overall['truncation_rate'])}")

    stats = build_pair_stats(rallies)
    print(f"\nTop {args.top_k} skill pairs by truncation rate")
    print("-" * 40)
    _print_table(stats, args.top_k)

    print(f"\nAll {len(stats)} skill pairs present in data (sorted by truncation rate)")
    print("-" * 40)
    _print_table(stats, len(stats))


if __name__ == "__main__":
    main()
