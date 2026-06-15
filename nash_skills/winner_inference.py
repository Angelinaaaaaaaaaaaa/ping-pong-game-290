"""
Shared terminal winner inference for Nash skill evaluation and labeling.

The competition environment ends a point when the ball travels more than
0.3 m past either racket.  The most reliable signal is therefore an explicit
winner emitted by the env, followed by reconstructing the racket x-positions
from info.  Ball x-velocity is a useful fallback, but should not be the only
source of truth for near-stationary terminal states.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

TABLE_SHIFT = 1.5
BALL_X_IDX = 36
BALL_VEL_X_IDX = 39
TERMINAL_MARGIN = 0.3
VELOCITY_EPS = 1e-6


def _as_float(value) -> Optional[float]:
    try:
        return float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None


def _normalise_winner_value(value) -> Optional[str]:
    """
    Convert common winner encodings to 'ego' / 'opp'.

    Prefer string values in new code. Numeric values are kept for compatibility
    with older experiments: 0/1 player-index style and 1/2 label style both
    appear in this repo, so the only unambiguous numeric opp label is 2.
    """
    if value is None:
        return None

    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"ego", "p0", "player0", "player_0", "left", "agent", "self"}:
            return "ego"
        if v in {"opp", "opponent", "p1", "player1", "player_1", "right"}:
            return "opp"
        if v in {"1", "+1"}:
            return "ego"
        if v in {"2", "-1"}:
            return "opp"
        return None

    numeric = _as_float(value)
    if numeric is None:
        return None
    if abs(numeric - 2.0) < 1e-9:
        return "opp"
    if abs(numeric - 1.0) < 1e-9 or abs(numeric - 0.0) < 1e-9:
        return "ego"
    if abs(numeric + 1.0) < 1e-9:
        return "opp"
    return None


def winner_from_info(info: Optional[dict]) -> Optional[str]:
    """Return an explicit winner from env info when one is available."""
    if not isinstance(info, dict):
        return None

    for key in ("winner", "point_winner", "episode_winner"):
        if key in info:
            winner = _normalise_winner_value(info[key])
            if winner is not None:
                return winner

    return None


def _winner_from_racket_bounds(
    obs,
    info: Optional[dict],
    *,
    table_shift: float,
    terminal_margin: float,
) -> Optional[str]:
    """
    Reconstruct the environment's terminal condition from terminal obs/info.

    info stores:
      diff_pos = target - ego_racket_pos
      diff_pos_opp = target_opp - mirrored_opp_racket_pos

    The opponent racket x-position in world coordinates is the mirror of the
    encoded opponent position around TABLE_SHIFT.
    """
    if not isinstance(info, dict):
        return None

    required = ("target", "diff_pos", "target_opp", "diff_pos_opp")
    if not all(key in info for key in required):
        return None

    ball_x = _as_float(np.asarray(obs)[BALL_X_IDX] if len(obs) > BALL_X_IDX else None)
    if ball_x is None:
        return None

    target = np.asarray(info["target"], dtype=np.float64).reshape(-1)
    diff_pos = np.asarray(info["diff_pos"], dtype=np.float64).reshape(-1)
    target_opp = np.asarray(info["target_opp"], dtype=np.float64).reshape(-1)
    diff_pos_opp = np.asarray(info["diff_pos_opp"], dtype=np.float64).reshape(-1)
    if min(len(target), len(diff_pos), len(target_opp), len(diff_pos_opp)) < 1:
        return None

    ego_racket_x = float(target[0] - diff_pos[0])
    opp_racket_x_mirrored = float(target_opp[0] - diff_pos_opp[0])
    opp_racket_x = 2.0 * table_shift - opp_racket_x_mirrored

    if ball_x < ego_racket_x - terminal_margin:
        return "opp"
    if ball_x > opp_racket_x + terminal_margin:
        return "ego"
    return None


def infer_terminal_winner(
    obs,
    info: Optional[dict] = None,
    *,
    table_shift: float = TABLE_SHIFT,
    terminal_margin: float = TERMINAL_MARGIN,
    velocity_eps: float = VELOCITY_EPS,
    fallback: Optional[str] = "position",
) -> Optional[str]:
    """
    Infer terminal winner as 'ego' / 'opp', or None if inconclusive.

    Order:
      1. explicit env info winner;
      2. exact env terminal boundary reconstructed from racket positions;
      3. ball x-velocity sign;
      4. optional final fallback:
         - 'position': ball_x relative to TABLE_SHIFT;
         - 'ego' / 'opp': fixed fallback;
         - None: return None.
    """
    explicit = winner_from_info(info)
    if explicit is not None:
        return explicit

    bounded = _winner_from_racket_bounds(
        obs,
        info,
        table_shift=table_shift,
        terminal_margin=terminal_margin,
    )
    if bounded is not None:
        return bounded

    arr = np.asarray(obs)
    if len(arr) > BALL_VEL_X_IDX:
        ball_vel_x = _as_float(arr[BALL_VEL_X_IDX])
        if ball_vel_x is not None and abs(ball_vel_x) > velocity_eps:
            return "ego" if ball_vel_x > 0 else "opp"

    if fallback == "position" and len(arr) > BALL_X_IDX:
        ball_x = _as_float(arr[BALL_X_IDX])
        if ball_x is not None:
            if ball_x > table_shift:
                return "ego"
            if ball_x < table_shift:
                return "opp"

    if fallback in {"ego", "opp"}:
        return fallback
    return None


def winner_to_label(winner: Optional[str]) -> int:
    """Convert 'ego' / 'opp' / None to the v2 label convention 1 / 2 / 0."""
    if winner == "ego":
        return 1
    if winner == "opp":
        return 2
    return 0
