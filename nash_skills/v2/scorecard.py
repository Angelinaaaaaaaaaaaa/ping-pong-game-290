"""
Shared evaluation scorecard -- a single metric set that any eval script can
compute from a handful of primitives (win/loss/truncated counts, rally
lengths, skill usage counts), so eval_matchup.py (main) and
eval_selfplay_5skill.py (origin/feature-selfplay) report the same complete
picture instead of each rolling their own ad hoc subset (meeting note item
19).

Metrics covered: win rate (decided-only), truncation rate, average and
median rally length, skill-usage entropy (normalised 0-1), dominant-skill
fraction, and per-skill usage percentages. This module is pure Python/numpy
with no MuJoCo, torch, or env dependency, so it is trivially unit-testable
and safe to import from either branch.
"""

import math
from typing import Optional

import numpy as np

__all__ = [
    "compute_scorecard",
    "skill_entropy",
    "skill_entropy_normalized",
    "dominant_skill_info",
    "format_scorecard",
]


def skill_entropy(skill_usage: dict) -> Optional[float]:
    """Shannon entropy (bits) of the skill-usage distribution, or None if empty/all-zero."""
    total = sum(skill_usage.values())
    if total == 0:
        return None
    probs = [v / total for v in skill_usage.values() if v > 0]
    return float(-sum(p * math.log2(p) for p in probs))


def skill_entropy_normalized(skill_usage: dict) -> Optional[float]:
    """
    Entropy normalised to [0, 1] by the maximum possible entropy for the
    number of distinct skills present (log2(n)). 1.0 = perfectly uniform
    usage, 0.0 = a single skill used exclusively. None if undefined (empty,
    all-zero, or only one skill key to begin with, since log2(1) == 0).
    """
    n = len(skill_usage)
    if n <= 1:
        return None
    ent = skill_entropy(skill_usage)
    if ent is None:
        return None
    max_ent = math.log2(n)
    return ent / max_ent if max_ent > 0 else None


def dominant_skill_info(skill_usage: dict):
    """Return (dominant_skill_name, fraction) or (None, None) if empty/all-zero."""
    total = sum(skill_usage.values())
    if total == 0:
        return None, None
    dominant = max(skill_usage, key=skill_usage.get)
    return dominant, skill_usage[dominant] / total


def compute_scorecard(
    wins: int,
    losses: int,
    truncated: int,
    rally_lengths: list,
    skill_usage: dict,
) -> dict:
    """
    Build the full scorecard from primitive counts.

    Parameters
    ----------
    wins, losses   : decided-outcome counts (ego perspective)
    truncated      : number of rallies that hit a step/length cap with no winner
    rally_lengths  : length (crossings or states, caller's choice of unit) for
                     every rally -- decided and truncated combined
    skill_usage    : {skill_name: pick_count} for the ego policy

    Returns
    -------
    dict with: wins, losses, truncated, decided, total, win_rate,
    truncation_rate, avg_rally_length, median_rally_length,
    skill_usage_frac, skill_entropy_bits, skill_entropy_normalized,
    dominant_skill, dominant_skill_fraction
    """
    decided = wins + losses
    total = decided + truncated

    win_rate = wins / decided if decided > 0 else None
    truncation_rate = truncated / total if total > 0 else None

    lengths = list(rally_lengths)
    avg_len = float(np.mean(lengths)) if lengths else None
    median_len = float(np.median(lengths)) if lengths else None

    usage_total = sum(skill_usage.values())
    skill_fracs = {
        k: (v / usage_total if usage_total > 0 else 0.0)
        for k, v in skill_usage.items()
    }
    dominant_skill, dominant_frac = dominant_skill_info(skill_usage)

    return {
        "wins": wins,
        "losses": losses,
        "truncated": truncated,
        "decided": decided,
        "total": total,
        "win_rate": win_rate,
        "truncation_rate": truncation_rate,
        "avg_rally_length": avg_len,
        "median_rally_length": median_len,
        "skill_usage_frac": skill_fracs,
        "skill_entropy_bits": skill_entropy(skill_usage),
        "skill_entropy_normalized": skill_entropy_normalized(skill_usage),
        "dominant_skill": dominant_skill,
        "dominant_skill_fraction": dominant_frac,
    }


def _pct(v: Optional[float]) -> str:
    return f"{v:.1%}" if v is not None else "---"


def _num(v: Optional[float], fmt: str = "{:.1f}") -> str:
    return fmt.format(v) if v is not None else "---"


def format_scorecard(sc: dict, label: str = "") -> str:
    """Render a scorecard dict (from compute_scorecard) as a readable block."""
    title = f"Scorecard{f' — {label}' if label else ''}"
    lines = [title, "-" * len(title)]

    lines.append(
        f"  win_rate (decided-only) : {_pct(sc['win_rate'])}  "
        f"({sc['wins']}W/{sc['losses']}L, n={sc['decided']})"
    )
    lines.append(
        f"  truncation_rate         : {_pct(sc['truncation_rate'])}  "
        f"({sc['truncated']}/{sc['total']})"
    )
    lines.append(
        f"  rally length            : avg={_num(sc['avg_rally_length'])}  "
        f"median={_num(sc['median_rally_length'])}"
    )
    lines.append(
        f"  skill entropy (0-1)     : {_num(sc['skill_entropy_normalized'], '{:.2f}')}"
    )
    dom = sc["dominant_skill"] or "---"
    lines.append(
        f"  dominant skill          : {dom} ({_pct(sc['dominant_skill_fraction'])})"
    )

    return "\n".join(lines)
