"""
Player-swap symmetry of the Design B joint state (state_encoder_attack_defend layout).

Robot 2 is robot 1 rotated 180 degrees about the vertical axis through the table centre
(iiwa14_comp.xml: base_opp at x = 3 with quat "0 0 0 1", identical joint chain), so
relabelling the players is an exact symmetry of the game:
  - robot blocks (qpos, qvel) and racket blocks (target / tracking error, already stored in
    each player's own frame) swap;
  - the ball, in world coordinates, rotates about (TABLE_SHIFT, 0): x -> 2*TABLE_SHIFT - x,
    y -> -y, vx -> -vx, vy -> -vy (z, vz unchanged), which also flips the attacker flag;
  - attack / preparation slots (choices or theta) swap; the winner swaps.
The acting-player slot is left as is (constant for alpha-potential states).

augment_rallies_with_swap doubles a training set with the swapped copy of every rally, which
pushes Q and Phi toward the game's side symmetry instead of learning separate behaviour for
each seat.
"""
import numpy as np

from nash_skills.v2.state_encoder_attack_defend import ENCODER_DIMS, ENCODER_V3

__all__ = ["TABLE_SHIFT", "swap_players", "augment_rallies_with_swap"]

TABLE_SHIFT = 1.5
_PAIRS = ((slice(0, 9), slice(18, 27)), (slice(9, 18), slice(27, 36)), (slice(42, 56), slice(56, 70)),
          (slice(70, 71), slice(71, 72)), (slice(74, 79), slice(79, 84)), (slice(84, 87), slice(87, 90)))


def swap_players(state):
    """Joint state with the two players' roles exchanged; returns a new array."""
    s = np.asarray(state)
    out = s.copy()
    for a, b in _PAIRS:
        out[a], out[b] = s[b], s[a]
    out[36] = 2.0 * TABLE_SHIFT - s[36]
    out[37] = -s[37]
    out[39] = -s[39]
    out[40] = -s[40]
    if len(s) == ENCODER_DIMS[ENCODER_V3]:
        # spin is an axial vector: the 180-degree turn about z flips its x and y components;
        # the last hitter changes sides with the players
        out[90], out[91] = -s[90], -s[91]
        out[93], out[94] = s[94], s[93]
    return out


_SWAPPED_KEYS = (("ego_attack_choices", "opp_attack_choices"), ("ego_defend_choices", "opp_defend_choices"),
                 ("ego_theta", "opp_theta"))


def _swap_rally(rally):
    out = dict(rally)
    out["states"] = [swap_players(s) for s in rally["states"]]
    out["winner"] = {1: 2, 2: 1}.get(rally.get("winner"), rally.get("winner"))
    for a, b in _SWAPPED_KEYS:
        if a in rally or b in rally:
            out[a], out[b] = rally.get(b), rally.get(a)
    return out


def augment_rallies_with_swap(rallies):
    """Original rallies followed by their player-swapped copies (originals are not modified)."""
    return list(rallies) + [_swap_rally(r) for r in rallies]
