"""
Offline audit: how much of self-play's total reward comes from per-crossing
shaping vs. the terminal win/loss signal -- pure Python, no MuJoCo, no model
loading (meeting note items 6 and 13).

Background
----------
nash_skills/v2/collect_data.py sets `net_crossings = len(curr_states)`, i.e.
the number of encoded states stored per rally IS the net-crossing count for
that rally. selfplay_5skill.py (origin/feature-selfplay, not yet merged to
main) computes, per rally:

    shaped        = SHAPING_COEF * crossings          # SHAPING_COEF = 0.05
    ego_terminal  = +1.0 / -1.0 / 0.0  (win / loss / truncated)
    ego_reward    = shaped + ego_terminal
    opp_reward    = shaped - ego_terminal

Both players receive the SAME positive `shaped` term regardless of who
"caused" the long rally -- it is not zero-sum. For a truncated rally (no
winner), `shaped` is the ENTIRE reward both players receive. For a decided
rally, `shaped` can rival or exceed the terminal +/-1 once crossings >= 20
(0.05 * 20 = 1.0), meaning rally length alone can outweigh actually winning.

This script quantifies that risk using rally-length data already collected
locally -- no new MuJoCo runs needed.

Usage
-----
    venv/bin/python nash_skills/v2/audit_reward_shaping.py
    venv/bin/python nash_skills/v2/audit_reward_shaping.py \\
        --input data_new/rallies_v2_1000.pkl --shaping-coef 0.05
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = [
    "shaped_reward",
    "compute_rally_shaping_stats",
    "SHAPING_COEF",
    "TERMINAL_MAGNITUDE",
]

_DEFAULT_INPUT = "data/rallies_5skill_v2.pkl"

# Mirrors selfplay_5skill.py (origin/feature-selfplay) exactly. Duplicated
# here as a local constant -- that file lives on a branch not merged to
# main, so it cannot be imported directly.
SHAPING_COEF = 0.05
TERMINAL_MAGNITUDE = 1.0


def _is_truncated(winner) -> bool:
    return winner not in (1, 2)


def _validate_rally_schema(rallies: list, source: str) -> None:
    """
    Raise ValueError if `rallies` isn't a list of rally dicts.

    Some legacy pickles (e.g. data_new/rallies.pkl) are a flat list of raw
    state arrays from the old pre-v2 pipeline, not the
    {'skill1','skill2','states','winner'} dict schema this script expects.
    Checking only the first entry is enough to catch "wrong file entirely"
    -- this codebase's own collectors never produce a schema-mixed pickle.
    Fails fast with a clear message instead of crashing deep inside
    compute_rally_shaping_stats with an opaque AttributeError.
    """
    if rallies and not isinstance(rallies[0], dict):
        raise ValueError(
            f"{source!r} does not use the expected rally-dict schema "
            f"(first entry is {type(rallies[0]).__name__}, not dict). "
            "This looks like a legacy/incompatible pickle format (e.g. a "
            "flat list of raw state arrays) rather than a list of "
            "{'skill1','skill2','states','winner'} rally dicts."
        )


def shaped_reward(n_crossings: int, shaping_coef: float = SHAPING_COEF) -> float:
    """The per-crossing shaping bonus BOTH players receive for a rally."""
    return shaping_coef * n_crossings


def compute_rally_shaping_stats(
    rallies: list,
    shaping_coef: float = SHAPING_COEF,
    terminal_magnitude: float = TERMINAL_MAGNITUDE,
) -> dict:
    """
    Split rallies into truncated / decided and quantify shaping's share of
    the total reward magnitude for each group.

    Truncated rallies: `shaped` IS the entire reward (ego_terminal == 0).
    Decided rallies: ratio = shaped / terminal_magnitude. ratio >= 1.0 means
    shaping alone is worth at least as much as the actual win/loss signal
    ("shaping dominates" that rally's reward).
    """
    truncated_shaped = []
    decided_ratios = []

    for r in rallies:
        n = len(r.get("states", []))
        shaped = shaped_reward(n, shaping_coef)
        if _is_truncated(r.get("winner")):
            truncated_shaped.append(shaped)
        else:
            decided_ratios.append(shaped / terminal_magnitude)

    def _pct(values, q):
        return float(np.percentile(values, q)) if values else None

    frac_dominates = (
        sum(1 for r in decided_ratios if r >= 1.0) / len(decided_ratios)
        if decided_ratios else None
    )

    return {
        "total": len(rallies),
        "truncated_count": len(truncated_shaped),
        "decided_count": len(decided_ratios),
        "truncated_shaped_rewards": truncated_shaped,
        "decided_shaping_ratios": decided_ratios,
        "truncated_shaped_mean": float(np.mean(truncated_shaped)) if truncated_shaped else None,
        "truncated_shaped_median": _pct(truncated_shaped, 50),
        "decided_ratio_mean": float(np.mean(decided_ratios)) if decided_ratios else None,
        "decided_ratio_median": _pct(decided_ratios, 50),
        "decided_ratio_p90": _pct(decided_ratios, 90),
        "decided_ratio_p99": _pct(decided_ratios, 99),
        "frac_decided_shaping_dominates": frac_dominates,
    }


def _fmt(v: Optional[float], suffix: str = "") -> str:
    return f"{v:.3f}{suffix}" if v is not None else "  ---"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit self-play reward-shaping budget vs terminal reward (no MuJoCo)."
    )
    ap.add_argument("--input", default=_DEFAULT_INPUT,
                    help=f"Rally pickle path (default: {_DEFAULT_INPUT})")
    ap.add_argument("--shaping-coef", type=float, default=SHAPING_COEF, dest="shaping_coef",
                    help=f"Per-crossing shaping coefficient (default: {SHAPING_COEF}, "
                         "matches selfplay_5skill.py)")
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Rally file not found: {args.input}")

    with open(path, "rb") as f:
        rallies = pickle.load(f)
    print(f"Loaded {len(rallies)} rallies from {args.input}")
    _validate_rally_schema(rallies, args.input)

    stats = compute_rally_shaping_stats(rallies, args.shaping_coef)

    print(f"\nReward-shaping budget audit "
          f"(SHAPING_COEF={args.shaping_coef}, terminal=+/-{TERMINAL_MAGNITUDE})")
    print("-" * 60)
    print(f"  total rallies       : {stats['total']}")
    print(f"  decided             : {stats['decided_count']}")
    print(f"  truncated           : {stats['truncated_count']}")

    print("\nTruncated rallies (shaped reward IS the entire reward, terminal=0):")
    print(f"  mean shaped reward  : {_fmt(stats['truncated_shaped_mean'])}")
    print(f"  median shaped reward: {_fmt(stats['truncated_shaped_median'])}")

    print("\nDecided rallies (ratio = shaped / terminal_magnitude):")
    print(f"  mean ratio          : {_fmt(stats['decided_ratio_mean'])}")
    print(f"  median ratio        : {_fmt(stats['decided_ratio_median'])}")
    print(f"  p90 ratio           : {_fmt(stats['decided_ratio_p90'])}")
    print(f"  p99 ratio           : {_fmt(stats['decided_ratio_p99'])}")
    frac = stats["frac_decided_shaping_dominates"]
    frac_str = _fmt(frac * 100 if frac is not None else None, "%")
    print(f"  frac ratio >= 1.0   : {frac_str}   (shaping alone >= a full win)")


if __name__ == "__main__":
    main()
