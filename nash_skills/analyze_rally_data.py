"""
Rally dataset analysis for the 5-skill Nash pipeline.

Loads data/rallies_5skill.pkl and reports:
  - total rallies / states
  - rallies and states per skill pair
  - near-empty rally count (arm-settling artifacts)
  - usable rally count per pair (after filtering)
  - imbalance ratio across the 25 pairs
  - per-pair status: ok / weak / critical
  - overall recommendation: train_now / collect_more /
                             topup_weak_pairs / drop_weak_skills

Run:
    python nash_skills/analyze_rally_data.py
    python nash_skills/analyze_rally_data.py --min-states 3 --ok-threshold 5
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────── #
# Thresholds                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

MIN_STATES_DEFAULT    = 3    # rallies with <= this many states are discarded
OK_THRESHOLD          = 5    # >= OK_THRESHOLD usable rallies → "ok"
CRITICAL_THRESHOLD    = 3    # < CRITICAL_THRESHOLD usable rallies → "critical"
IMBALANCE_WARNING     = 10.0 # max/min ratio above this is problematic


# ─────────────────────────────────────────────────────────────────────────── #
# Analysis functions                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def count_rallies(rallies: List[Dict]) -> Dict:
    """
    Count total and per-(skill1, skill2) rallies.

    Returns dict with:
      total    — int
      per_pair — {(skill1, skill2): count}
    """
    per_pair: Dict[Tuple, int] = defaultdict(int)
    for r in rallies:
        per_pair[(r['skill1'], r['skill2'])] += 1
    return {'total': len(rallies), 'per_pair': dict(per_pair)}


def count_states(rallies: List[Dict]) -> Dict:
    """
    Count total and per-(skill1, skill2) states across all rallies.

    Returns dict with:
      total    — int
      per_pair — {(skill1, skill2): state_count}
    """
    per_pair: Dict[Tuple, int] = defaultdict(int)
    total = 0
    for r in rallies:
        n = len(r['states'])
        per_pair[(r['skill1'], r['skill2'])] += n
        total += n
    return {'total': total, 'per_pair': dict(per_pair)}


def usable_rallies(rallies: List[Dict], min_states: int = MIN_STATES_DEFAULT) -> Dict:
    """
    Filter rallies to those with > min_states states (arm settling artifacts removed).

    Returns dict with:
      total_usable  — int
      total_near_empty — int (rallies with <= min_states states)
      per_pair      — {(skill1, skill2): usable_count}
      usable_states — total state count from usable rallies only
    """
    per_pair: Dict[Tuple, int] = defaultdict(int)
    total_usable = 0
    total_near_empty = 0
    usable_states = 0
    for r in rallies:
        n = len(r['states'])
        if n > min_states:
            k = (r['skill1'], r['skill2'])
            per_pair[k] += 1
            total_usable += 1
            usable_states += n
        else:
            total_near_empty += 1
    return {
        'total_usable':     total_usable,
        'total_near_empty': total_near_empty,
        'per_pair':         dict(per_pair),
        'usable_states':    usable_states,
    }


def imbalance_ratio(per_pair: Dict[Tuple, int]) -> float:
    """
    Compute max_count / max(min_count, 1) across all pairs.

    Missing pairs count as 0.
    """
    if not per_pair:
        return 1.0
    counts = list(per_pair.values())
    return max(counts) / max(min(counts), 1)


def pair_recommendation(usable: int,
                         ok_threshold: int = OK_THRESHOLD,
                         critical_threshold: int = CRITICAL_THRESHOLD) -> str:
    """
    Return per-pair status string based on usable rally count.

      'ok'       — >= ok_threshold
      'weak'     — >= critical_threshold but < ok_threshold
      'critical' — < critical_threshold
    """
    if usable >= ok_threshold:
        return 'ok'
    if usable >= critical_threshold:
        return 'weak'
    return 'critical'


def overall_recommendation(total_usable: int,
                            n_critical: int,
                            n_weak: int,
                            imbalance: float) -> str:
    """
    Return one overall action string:

      'train_now'        — data is sufficient and balanced
      'topup_weak_pairs' — a few pairs are weak but no criticals, high imbalance
      'collect_more'     — many weak/critical pairs, collect uniformly
      'drop_weak_skills' — structural problem: specific skills always critical
    """
    # Critical pairs mean we cannot train reliably on those combos
    if n_critical >= 5:
        return 'drop_weak_skills'
    if n_critical > 0:
        return 'collect_more'
    # No criticals: check imbalance and weak counts
    if imbalance > IMBALANCE_WARNING or n_weak >= 6:
        return 'topup_weak_pairs'
    # Enough usable rallies, balanced enough
    if total_usable >= 150 and n_weak <= 5:
        return 'train_now'
    return 'collect_more'


# ─────────────────────────────────────────────────────────────────────────── #
# Full analysis runner                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def analyse(rallies: List[Dict],
            min_states: int = MIN_STATES_DEFAULT,
            ok_threshold: int = OK_THRESHOLD,
            critical_threshold: int = CRITICAL_THRESHOLD) -> Dict:
    """Run the full analysis and return a summary dict."""
    rally_counts  = count_rallies(rallies)
    state_counts  = count_states(rallies)
    usable        = usable_rallies(rallies, min_states=min_states)
    imbal         = imbalance_ratio(usable['per_pair'])

    from nash_skills.skills import SKILL_NAMES
    all_pairs = [(s1, s2) for s1 in SKILL_NAMES for s2 in SKILL_NAMES]

    per_pair_status = {}
    for pair in all_pairs:
        u = usable['per_pair'].get(pair, 0)
        per_pair_status[pair] = {
            'total_rallies':  rally_counts['per_pair'].get(pair, 0),
            'total_states':   state_counts['per_pair'].get(pair, 0),
            'usable_rallies': u,
            'status':         pair_recommendation(u, ok_threshold, critical_threshold),
            'avg_rally_len':  (state_counts['per_pair'].get(pair, 0) /
                               max(rally_counts['per_pair'].get(pair, 1), 1)),
        }

    n_critical = sum(1 for v in per_pair_status.values() if v['status'] == 'critical')
    n_weak     = sum(1 for v in per_pair_status.values() if v['status'] == 'weak')
    n_ok       = sum(1 for v in per_pair_status.values() if v['status'] == 'ok')

    rec = overall_recommendation(
        total_usable=usable['total_usable'],
        n_critical=n_critical,
        n_weak=n_weak,
        imbalance=imbal,
    )

    return {
        'total_rallies':    rally_counts['total'],
        'total_states':     state_counts['total'],
        'total_usable':     usable['total_usable'],
        'total_near_empty': usable['total_near_empty'],
        'usable_states':    usable['usable_states'],
        'imbalance_ratio':  round(imbal, 1),
        'n_ok':             n_ok,
        'n_weak':           n_weak,
        'n_critical':       n_critical,
        'per_pair':         per_pair_status,
        'recommendation':   rec,
    }


# ─────────────────────────────────────────────────────────────────────────── #
# Report printer                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def print_report(summary: Dict, file=None):
    if file is None:
        file = sys.stdout

    p = lambda *a, **kw: print(*a, **kw, file=file)

    p()
    p("=" * 72)
    p("  5-SKILL RALLY DATASET ANALYSIS")
    p("=" * 72)
    p(f"  Total rallies collected : {summary['total_rallies']}")
    p(f"  Total states            : {summary['total_states']}")
    p(f"  Near-empty rallies (<=3 states, discarded) : {summary['total_near_empty']}")
    p(f"  Usable rallies (>3 states) : {summary['total_usable']}")
    p(f"  States from usable rallies : {summary['usable_states']}")
    p(f"  Imbalance ratio (max/min usable) : {summary['imbalance_ratio']:.1f}x")
    p()
    p(f"  Pair status breakdown: "
      f"ok={summary['n_ok']}  "
      f"weak={summary['n_weak']}  "
      f"critical={summary['n_critical']}  "
      f"(out of 25 pairs)")
    p()

    from nash_skills.skills import SKILL_NAMES
    STATUS_ICON = {'ok': '  ', 'weak': '⚠ ', 'critical': '✗ '}
    header = (f"  {'skill1':<14} {'skill2':<14} "
              f"{'rallies':>8} {'usable':>7} {'avg_len':>8} {'status':>10}")
    p(header)
    p("  " + "-" * 66)
    for s1 in SKILL_NAMES:
        for s2 in SKILL_NAMES:
            v = summary['per_pair'][(s1, s2)]
            icon = STATUS_ICON.get(v['status'], '  ')
            p(f"  {s1:<14} {s2:<14} "
              f"{v['total_rallies']:>8} {v['usable_rallies']:>7} "
              f"{v['avg_rally_len']:>8.1f} "
              f"{icon}{v['status']:>8}")

    p()
    p("  " + "=" * 66)
    p(f"  RECOMMENDATION: {summary['recommendation'].upper()}")
    p("  " + "=" * 66)

    rec = summary['recommendation']
    if rec == 'train_now':
        p("  Data is sufficient and balanced. Proceed to training.")
    elif rec == 'topup_weak_pairs':
        p("  Data is highly imbalanced. Re-run collection focusing on")
        p("  pairs marked 'weak' or 'critical' to bring them up to ≥5")
        p("  usable rallies each before training.")
    elif rec == 'collect_more':
        p("  Too many critical pairs (< 3 usable rallies). Collect more")
        p("  data uniformly across all 25 pairs. Aim for ≥5 usable per pair.")
    elif rec == 'drop_weak_skills':
        p("  Structural imbalance: skills with systematically near-zero")
        p("  usable rally counts may be too out-of-distribution for the PPO.")
        p("  Consider dropping the worst-performing skills and reducing to a")
        p("  4-skill (or 3-skill) set before retraining.")
    p()


# ─────────────────────────────────────────────────────────────────────────── #
# Main                                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(description="Analyse 5-skill rally dataset")
    parser.add_argument('--path',          default='data/rallies_5skill.pkl')
    parser.add_argument('--min-states',    type=int, default=MIN_STATES_DEFAULT)
    parser.add_argument('--ok-threshold',  type=int, default=OK_THRESHOLD)
    parser.add_argument('--output-json',   default='skill_eval/rally_data_analysis.json')
    args = parser.parse_args()

    import pickle
    with open(args.path, 'rb') as f:
        rallies = pickle.load(f)

    summary = analyse(rallies,
                      min_states=args.min_states,
                      ok_threshold=args.ok_threshold)
    print_report(summary)

    os.makedirs(
        os.path.dirname(args.output_json)
        if os.path.dirname(args.output_json) else '.', exist_ok=True
    )
    # Convert tuple keys to strings for JSON
    out = {k: v for k, v in summary.items() if k != 'per_pair'}
    out['per_pair'] = {
        f"{s1},{s2}": vv
        for (s1, s2), vv in summary['per_pair'].items()
    }
    with open(args.output_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Saved to: {args.output_json}")


if __name__ == '__main__':
    main()
