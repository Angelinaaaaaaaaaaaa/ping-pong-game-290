"""
Win-oriented discounted-return labeling for the v2 Nash pipeline.

Design rationale
================

The old pipeline assigned labels {-1, 0, +1} only to the last two crossings
in a rally, leaving ~96% of states labeled 0.  This was a debugging artifact,
not an intentional design.  The mentor confirmed that the intended objective
is to WIN and that gamma=0.9 is valid for shorter-horizon strategy learning.

New scheme
----------

Each rally is a sequence of ball crossings (states recorded each time the ball
crosses the net midpoint x=1.5m).  At the end of the rally we know the winner:

  winner = 1  : ego won   (ball exited past opponent's racket)
  winner = 2  : opp won   (ball exited past ego's racket)
  winner = 0  : truncated (step limit hit, no clear winner)

Terminal rewards (assigned to the LAST crossing in the rally):
  ego terminal reward   r1 = +1  if winner==1 else  -1  if winner==2 else  0
  opp terminal reward   r2 = -r1                          (zero-sum)

For all earlier crossings (t < L-1): r = 0 (intermediate steps have no reward).

Discounted returns (backward pass):
  G1[L-1] = r1
  G1[t]   = gamma * G1[t+1]   for t = L-2, ..., 0

G2[t] = -G1[t] at every step (zero-sum property).

This means:
  - Every state in a won rally gets a positive return (decayed by distance)
  - Every state in a lost rally gets a negative return
  - Truncated rallies get zero return throughout

Constants
---------
GAMMA : float — default discount factor (0.9 per mentor guidance)
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

# Default discount factor — use gamma=0.9 for shorter-horizon strategy learning
GAMMA: float = 0.9

# Index into each raw 116-dim state where ball velocity x is stored
_BALL_VEL_X_IDX = 39


def detect_winner(rally: list, done: bool) -> int:
    """
    Infer which player won this rally from the terminal state.

    Parameters
    ----------
    rally : list of np.ndarray (116,)
        Sequence of states recorded at each ball crossing.
        May be empty if no crossings occurred.
    done  : bool
        True if the episode terminated with a real done signal
        (ball exited past a racket by >0.3m).

    Returns
    -------
    int
        1  — ego player won
        2  — opponent won
        0  — truncated / inconclusive
    """
    if not done or len(rally) == 0:
        return 0

    # The terminal ball velocity x-component tells us who missed:
    #   ball_vel[0] > 0  → ball moving toward opponent's side → opp missed → ego wins
    #   ball_vel[0] < 0  → ball moving toward ego's side      → ego missed → opp wins
    last_state = rally[-1]
    ball_vel_x = float(last_state[_BALL_VEL_X_IDX])

    if ball_vel_x > 0:
        return 1   # ego wins
    else:
        return 2   # opp wins


def compute_returns(
    rally: list,
    gamma: float = GAMMA,
    winner: int = 0,
) -> Tuple[List[float], List[float]]:
    """
    Compute discounted returns G1 and G2 for every crossing in a rally.

    Parameters
    ----------
    rally  : list of np.ndarray — crossing states (may be empty)
    gamma  : float — discount factor (default: GAMMA = 0.9)
    winner : int   — 1=ego, 2=opp, 0=truncated

    Returns
    -------
    (G1, G2) : two lists of float, each of length len(rally)
        G1[t] — ego's discounted return from crossing t onward
        G2[t] — opp's discounted return (= -G1[t], zero-sum)
    """
    L = len(rally)
    if L == 0:
        return [], []

    # Terminal reward for ego
    if winner == 1:
        r_terminal = 1.0
    elif winner == 2:
        r_terminal = -1.0
    else:
        r_terminal = 0.0

    # Backward pass
    G1 = [0.0] * L
    G1[L - 1] = r_terminal
    for t in range(L - 2, -1, -1):
        G1[t] = gamma * G1[t + 1]

    G2 = [-v for v in G1]
    return G1, G2


def summarise_balance(rallies: list) -> Dict[Tuple[str, str], int]:
    """
    Count how many rallies exist for each (skill1, skill2) pair.

    Parameters
    ----------
    rallies : list of rally dicts, each with keys 'skill1', 'skill2'

    Returns
    -------
    dict mapping (skill1, skill2) -> count
    """
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for r in rallies:
        key = (r["skill1"], r["skill2"])
        counts[key] += 1
    return dict(counts)


def check_balance(
    rallies: list,
    threshold: float = 10.0,
) -> Tuple[bool, float]:
    """
    Check whether the dataset is balanced across skill pairs.

    Parameters
    ----------
    rallies   : list of rally dicts
    threshold : float — max allowed ratio of max_count / min_count

    Returns
    -------
    (is_balanced, ratio)
        is_balanced : True if max/min <= threshold
        ratio       : actual max/min ratio (inf if any pair has 0 rallies)
    """
    counts = summarise_balance(rallies)
    if not counts:
        return True, 1.0

    max_count = max(counts.values())
    min_count = min(counts.values())

    if min_count == 0:
        return False, float("inf")

    ratio = max_count / min_count
    return ratio <= threshold, ratio
