"""
Inspect how many training states each rally contributes -- pure Python, no
MuJoCo, no model loading.

Long rallies contribute more (correlated) states to the training set than
short ones, since every net-crossing appends one encoded state
(nash_skills/v2/collect_data.py: `net_crossings = len(curr_states)`). If a
small number of long rallies dominate the dataset, the effective sample size
is much smaller than the raw state count suggests (meeting note item 7).

Reports:
  - Gini coefficient of the per-rally state-count distribution (0 = every
    rally contributes equally, 1 = one rally contributes everything)
  - Share of total states held by the top 1% / 5% / 10% longest rallies
  - A table of the longest individual rallies (skill pair, winner, length)

Usage
-----
    venv/bin/python nash_skills/v2/inspect_rally_redundancy.py
    venv/bin/python nash_skills/v2/inspect_rally_redundancy.py \\
        --input data_new/rallies_v2_1000.pkl --top-k 15
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import pickle
from pathlib import Path
from typing import Optional

__all__ = ["gini_coefficient", "concentration_stats", "top_k_rallies_table"]

_DEFAULT_INPUT = "data/rallies_5skill_v2.pkl"
_DEFAULT_TOP_FRACS = (0.01, 0.05, 0.10)


def _validate_rally_schema(rallies: list, source: str) -> None:
    """
    Raise ValueError if `rallies` isn't a list of rally dicts.

    Some legacy pickles (e.g. data_new/rallies.pkl) are a flat list of raw
    state arrays from the old pre-v2 pipeline, not the
    {'skill1','skill2','states','winner'} dict schema this script expects.
    Checking only the first entry is enough to catch "wrong file entirely"
    -- this codebase's own collectors never produce a schema-mixed pickle.
    Fails fast with a clear message instead of crashing deep inside
    concentration_stats/top_k_rallies_table with an opaque AttributeError.
    """
    if rallies and not isinstance(rallies[0], dict):
        raise ValueError(
            f"{source!r} does not use the expected rally-dict schema "
            f"(first entry is {type(rallies[0]).__name__}, not dict). "
            "This looks like a legacy/incompatible pickle format (e.g. a "
            "flat list of raw state arrays) rather than a list of "
            "{'skill1','skill2','states','winner'} rally dicts."
        )


def gini_coefficient(values: list) -> float:
    """
    Gini coefficient of a non-negative distribution: 0 = perfectly equal
    contribution, approaching 1 as one value dominates the rest.
    """
    total = sum(values)
    if not values or total == 0:
        return 0.0

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    weighted_sum = sum((i + 1) * v for i, v in enumerate(sorted_vals))
    return (2 * weighted_sum) / (n * total) - (n + 1) / n


def concentration_stats(rallies: list, top_fracs=_DEFAULT_TOP_FRACS) -> dict:
    """
    Quantify how concentrated the per-rally state counts are.

    For each fraction f in top_fracs, reports how many states the longest
    f-fraction of rallies (by state count) contribute, and what share of
    the dataset's total states that is.
    """
    counts = [len(r.get("states", [])) for r in rallies]
    total_states = sum(counts)
    total_rallies = len(counts)
    sorted_counts = sorted(counts, reverse=True)

    result = {
        "total_rallies": total_rallies,
        "total_states": total_states,
        "gini": gini_coefficient(counts),
        "max_states_single_rally": sorted_counts[0] if sorted_counts else 0,
        "top_fracs": {},
    }

    for frac in top_fracs:
        k = max(1, round(total_rallies * frac)) if total_rallies > 0 else 0
        top_states = sum(sorted_counts[:k])
        share = (top_states / total_states) if total_states > 0 else None
        result["top_fracs"][frac] = {
            "n_rallies": k,
            "states": top_states,
            "share_of_total": share,
        }

    return result


def top_k_rallies_table(rallies: list, k: int = 10) -> list:
    """
    Return the k rallies with the most states, sorted descending, as plain
    dicts: skill1, skill2, winner, n_states.
    """
    rows = [
        {
            "skill1": r.get("skill1"),
            "skill2": r.get("skill2"),
            "winner": r.get("winner"),
            "n_states": len(r.get("states", [])),
        }
        for r in rallies
    ]
    rows.sort(key=lambda row: row["n_states"], reverse=True)
    return rows[:k]


def _fmt_pct(v: Optional[float]) -> str:
    return f"{v:.1%}" if v is not None else "  ---"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inspect per-rally sample redundancy in a rally pickle (no MuJoCo)."
    )
    ap.add_argument("--input", default=_DEFAULT_INPUT,
                    help=f"Rally pickle path (default: {_DEFAULT_INPUT})")
    ap.add_argument("--top-k", type=int, default=10, dest="top_k",
                    help="Number of longest individual rallies to list (default: 10)")
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Rally file not found: {args.input}")

    with open(path, "rb") as f:
        rallies = pickle.load(f)
    print(f"Loaded {len(rallies)} rallies from {args.input}")
    _validate_rally_schema(rallies, args.input)

    stats = concentration_stats(rallies)

    print("\nSample redundancy / concentration")
    print("-" * 50)
    print(f"  total rallies           : {stats['total_rallies']}")
    print(f"  total states            : {stats['total_states']}")
    print(f"  gini coefficient        : {stats['gini']:.3f}  (0=equal, 1=one rally dominates)")
    print(f"  longest single rally    : {stats['max_states_single_rally']} states")

    print("\n  Share of total states held by the longest N% of rallies:")
    for frac, info in sorted(stats["top_fracs"].items()):
        print(f"    top {frac:>5.0%} ({info['n_rallies']:>4} rallies): "
              f"{info['states']:>7} states = {_fmt_pct(info['share_of_total'])}")

    table = top_k_rallies_table(rallies, args.top_k)
    print(f"\nTop {len(table)} longest individual rallies")
    print("-" * 60)
    hdr = f"{'skill1':<14} {'skill2':<14} {'winner':>7} {'n_states':>9}"
    print(hdr)
    print("-" * len(hdr))
    for row in table:
        print(f"{str(row['skill1']):<14} {str(row['skill2']):<14} "
              f"{str(row['winner']):>7} {row['n_states']:>9}")


if __name__ == "__main__":
    main()
