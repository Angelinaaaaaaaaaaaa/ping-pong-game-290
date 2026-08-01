"""
Analyze existing evaluation results without running MuJoCo.

Reads JSON/CSV result files produced by eval_matchup.py, eval_matchup_2skill.py,
or eval_multiseed.py and prints summary tables covering:
  - win rate (denominator: done episodes only)
  - clean win rate when available (decided only)
  - truncated episodes and truncation rate
  - average rally length
  - most-used skill and dominant fraction

Supported formats
-----------------
Format A — "results" list (5-skill matchup, eval_matchup.py):
    {"results": [{strategy1, strategy2, episodes, truncated_episodes,
                  ego_wins, opp_wins, win_rate, avg_rally_length, ...}]}

Format B — "per_seed_results" + "aggregate_results" (2-skill, eval_matchup_2skill.py):
    {"config": {...}, "per_seed_results": [...], "aggregate_results": [...]}

CSV columns are auto-detected from the header row.

Usage
-----
    python nash_skills/v2/analyze_existing_results.py
    python nash_skills/v2/analyze_existing_results.py --file path/to/results.json
    python nash_skills/v2/analyze_existing_results.py --file results.csv --format csv
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Optional

# ── default search paths ────────────────────────────────────────────────────
DEFAULT_FILES = [
    "skill_eval/matchup_results_5skill.json",
    "skill_eval/results_5skill_eval.json",
    "skill_eval/matchup_results_5skill.csv",
    "results_2skill_small.json",
    "results_2skill_small_per_seed.csv",
    "results_2skill_small.csv",
    "skill_eval/matchup_results_v2.json",
]

_W = 16   # column width for strategy names


# ── helpers ─────────────────────────────────────────────────────────────────

def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return f if f == f else None  # NaN guard
    except (TypeError, ValueError):
        return None


def _trunc_rate(episodes, truncated) -> Optional[float]:
    """truncated / (episodes + truncated)  — avoids double-counting."""
    eps = _safe_float(episodes)
    trunc = _safe_float(truncated)
    if eps is None or trunc is None:
        return None
    denom = eps + trunc
    return trunc / denom if denom > 0 else 0.0


def _done_win_rate(ego_wins, opp_wins) -> Optional[float]:
    """ego_wins / (ego_wins + opp_wins)  — decided-only denominator."""
    e = _safe_float(ego_wins)
    o = _safe_float(opp_wins)
    if e is None or o is None or (e + o) == 0:
        return None
    return e / (e + o)


def _fmt(v, fmt=".3f", missing="  ---") -> str:
    if v is None or v == "":
        return missing
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return str(v)


def _row_banner(label: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(f"{'=' * 72}")


# ── format A: results list ───────────────────────────────────────────────────

def _parse_format_a(data: dict) -> list[dict]:
    return data.get("results", [])


def _print_format_a(rows: list[dict], source: str) -> None:
    _row_banner(f"Format A: {source}")
    hdr = (f"{'s1':<{_W}} {'s2':<{_W}} {'ep':>5} {'trunc':>6} "
           f"{'tr%':>5} {'wr':>6} {'wr_dec':>7} {'rally':>6} {'top_skill'}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ep    = _safe_float(r.get("episodes"))
        trunc = _safe_float(r.get("truncated_episodes", 0))
        tr_pc = _trunc_rate(ep, trunc)
        wr    = _safe_float(r.get("win_rate"))
        wr_c  = _done_win_rate(r.get("ego_wins"), r.get("opp_wins"))
        rally = _safe_float(r.get("avg_rally_length"))
        skill = r.get("most_used_skill") or r.get("dominant_skill") or ""
        dfrac = _safe_float(r.get("dominant_fraction"))
        skill_str = f"{skill}({_fmt(dfrac, '.0%', '?')})" if skill else "---"
        print(
            f"{str(r.get('strategy1','')):<{_W}} "
            f"{str(r.get('strategy2','')):<{_W}} "
            f"{_fmt(ep, '.0f'):>5} "
            f"{_fmt(trunc, '.0f'):>6} "
            f"{_fmt(tr_pc, '.1%', '  ---'):>5} "
            f"{_fmt(wr, '.3f'):>6} "
            f"{_fmt(wr_c, '.3f'):>7} "
            f"{_fmt(rally, '.2f'):>6} "
            f"{skill_str}"
        )
    print(f"\n  Rows: {len(rows)}")
    print("  Columns: ep=total episodes requested, trunc=truncated before done,")
    print("  tr%=truncated/(ep+trunc), wr=reported win rate, wr_dec=decided-only win rate")


# ── format B: per_seed + aggregate ──────────────────────────────────────────

def _parse_format_b_per_seed(data: dict) -> list[dict]:
    return data.get("per_seed_results", [])


def _parse_format_b_agg(data: dict) -> list[dict]:
    return data.get("aggregate_results", [])


def _print_format_b_agg(rows: list[dict], source: str) -> None:
    _row_banner(f"Format B aggregate: {source}")
    hdr = (f"{'s1':<{_W}} {'s2':<{_W}} "
           f"{'wr_mean':>8} {'wr_std':>7} {'trunc_m':>8} "
           f"{'rally_m':>8} {'dominant_skill'}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{str(r.get('strategy1','')):<{_W}} "
            f"{str(r.get('strategy2','')):<{_W}} "
            f"{_fmt(r.get('win_rate_mean'), '.3f'):>8} "
            f"{_fmt(r.get('win_rate_std'),  '.3f'):>7} "
            f"{_fmt(r.get('trunc_mean'),    '.1f'):>8} "
            f"{_fmt(r.get('avg_rally_mean'),'.2f'):>8} "
            f"{r.get('dominant_skill','---')}"
        )
    print(f"\n  Rows: {len(rows)}")


def _print_format_b_per_seed(rows: list[dict], source: str) -> None:
    _row_banner(f"Format B per-seed: {source}")
    hdr = (f"{'s':<3} {'s1':<{_W}} {'s2':<{_W}} "
           f"{'ep':>5} {'trunc':>6} {'tr%':>5} "
           f"{'wr':>6} {'wr_c':>6} {'rally':>6} {'top_skill'}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ep    = _safe_float(r.get("episodes"))
        trunc = _safe_float(r.get("truncated_episodes", 0))
        tr_pc = _trunc_rate(ep, trunc)
        wr    = _safe_float(r.get("win_rate"))
        wr_c  = _safe_float(r.get("win_rate_clean")) or _done_win_rate(
                    r.get("ego_wins"), r.get("opp_wins"))
        rally = _safe_float(r.get("avg_rally_length"))
        skill = r.get("most_used_skill", "")
        dfrac = _safe_float(r.get("dominant_fraction"))
        skill_str = f"{skill}({_fmt(dfrac, '.0%', '?')})" if skill else "---"
        seed = r.get("seed", r.get("matchup_seed", "?"))
        print(
            f"{str(seed):<3} "
            f"{str(r.get('strategy1','')):<{_W}} "
            f"{str(r.get('strategy2','')):<{_W}} "
            f"{_fmt(ep, '.0f'):>5} "
            f"{_fmt(trunc, '.0f'):>6} "
            f"{_fmt(tr_pc, '.1%', '  ---'):>5} "
            f"{_fmt(wr, '.3f'):>6} "
            f"{_fmt(wr_c, '.3f'):>6} "
            f"{_fmt(rally, '.2f'):>6} "
            f"{skill_str}"
        )
    print(f"\n  Rows: {len(rows)}")


# ── CSV ──────────────────────────────────────────────────────────────────────

def _print_csv(path: str) -> None:
    _row_banner(f"CSV: {path}")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        print("  (empty)")
        return

    cols = list(rows[0].keys())
    has_win_rate   = "win_rate" in cols
    has_trunc      = "truncated_episodes" in cols
    has_rally      = "avg_rally_length" in cols
    has_skill      = "most_used_skill" in cols or "dominant_skill" in cols
    has_ep         = "episodes" in cols

    hdr = (f"{'s1':<{_W}} {'s2':<{_W}}")
    if has_ep:        hdr += f" {'ep':>5}"
    if has_trunc:     hdr += f" {'trunc':>6} {'tr%':>5}"
    if has_win_rate:  hdr += f" {'wr':>6}"
    if has_rally:     hdr += f" {'rally':>6}"
    if has_skill:     hdr += f" {'top_skill'}"
    print(hdr)
    print("-" * len(hdr))

    for r in rows:
        line = (f"{str(r.get('strategy1','')):<{_W}} "
                f"{str(r.get('strategy2','')):<{_W}}")
        if has_ep:
            line += f" {_fmt(r.get('episodes'), '.0f'):>5}"
        if has_trunc:
            trunc = _safe_float(r.get("truncated_episodes", 0))
            ep    = _safe_float(r.get("episodes"))
            tr_pc = _trunc_rate(ep, trunc)
            line += f" {_fmt(trunc, '.0f'):>6} {_fmt(tr_pc, '.1%', '  ---'):>5}"
        if has_win_rate:
            line += f" {_fmt(r.get('win_rate'), '.3f'):>6}"
        if has_rally:
            line += f" {_fmt(r.get('avg_rally_length'), '.2f'):>6}"
        if has_skill:
            skill = r.get("most_used_skill") or r.get("dominant_skill") or "---"
            line += f" {skill}"
        print(line)
    print(f"\n  Rows: {len(rows)}")


# ── main ─────────────────────────────────────────────────────────────────────

def analyse_file(path: str) -> None:
    p = Path(path)
    if not p.exists():
        print(f"  [SKIP] not found: {path}")
        return

    if p.suffix.lower() == ".csv":
        _print_csv(str(p))
        return

    with open(p) as f:
        data = json.load(f)

    if "results" in data:
        rows = _parse_format_a(data)
        _print_format_a(rows, str(p))
    elif "per_seed_results" in data or "aggregate_results" in data:
        agg  = _parse_format_b_agg(data)
        seed = _parse_format_b_per_seed(data)
        if agg:
            _print_format_b_agg(agg, str(p))
        if seed:
            _print_format_b_per_seed(seed, str(p))
    else:
        print(f"  [UNKNOWN FORMAT] {path}  top-level keys: {list(data.keys())}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyse existing evaluation JSON/CSV results without running MuJoCo.")
    ap.add_argument("--file", "-f", nargs="*",
                    help="One or more result files to analyse (default: all known paths)")
    args = ap.parse_args()

    files = args.file if args.file else DEFAULT_FILES

    print("Nash Skills — Existing Results Analysis")
    print(f"Working directory: {os.getcwd()}")

    for path in files:
        analyse_file(path)

    print("\n" + "=" * 72)
    print("Legend")
    print("=" * 72)
    print("  ep      — episodes requested (done + may include capped)")
    print("  trunc   — episodes that hit the step cap before a winner was decided")
    print("  tr%     — truncated / (ep + trunc)  [avoids double-counting done episodes]")
    print("  wr      — reported win rate (denominator: done episodes)")
    print("  wr_dec  — decided-only win rate: ego_wins / (ego_wins + opp_wins)")
    print("  wr_c    — win_rate_clean field when available (same as wr_dec if stored)")
    print()
    print("  Truncation bias: if skill pairs that ego wins tend to terminate quickly")
    print("  while pairs ego loses run long and hit the cap, then wr_dec > wr (optimistic).")
    print("  Conversely: if ego-winning pairs run long → wr_dec < wr (pessimistic).")


if __name__ == "__main__":
    main()
