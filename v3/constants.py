"""Constants for the isolated v3.0 defender pre-position prototype.

The current MuJoCo environment clips both players' gantry coordinates to:
  x in [-1.0, -0.5]
  y in [-1.0,  1.0]

Ready reset uses raw gantry qpos x=-0.5, y=0.0 for both players. Semantic
defensive targets are represented in physical/world coordinates and converted
to each player's raw gantry frame before action control.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFENSE_POSITIONS = ("left", "center", "right")

GANTRY_X_MIN = -1.0
GANTRY_X_MAX = -0.5
GANTRY_Y_MIN = -1.0
GANTRY_Y_MAX = 1.0

READY_GANTRY_X = -0.5
READY_GANTRY_Y = 0.0

DEFENSE_LATERAL_OFFSET = 0.35


@dataclass(frozen=True)
class GantryTarget:
    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


DEFENSE_GANTRY_TARGETS = {
    # Semantic/world targets for P1.  Position names are from the defender's
    # own physical perspective; P1 faces +x, so its physical left is +world-y.
    "left": GantryTarget(READY_GANTRY_X, DEFENSE_LATERAL_OFFSET),
    "center": GantryTarget(READY_GANTRY_X, READY_GANTRY_Y),
    "right": GantryTarget(READY_GANTRY_X, -DEFENSE_LATERAL_OFFSET),
}


def validate_defense_position(position: str) -> str:
    if position not in DEFENSE_GANTRY_TARGETS:
        choices = ", ".join(DEFENSE_POSITIONS)
        raise ValueError(f"Unknown defense position '{position}'. Choose from: {choices}")
    return position


def target_for_defender(player: int, position: str) -> GantryTarget:
    """Return the desired physical/world target for a defending player.

    The defender position names are defined from the defender's perspective:
    P1 ``left`` is positive world-y and P2 ``left`` is negative world-y.  The
    returned y coordinate is never a raw P2 qpos coordinate.
    """
    if player not in (1, 2):
        raise ValueError("player must be 1 or 2")
    target = DEFENSE_GANTRY_TARGETS[validate_defense_position(position)]
    if player == 1:
        return target
    return GantryTarget(2 * 1.5 - target.x, -target.y)


def world_to_gantry_target(player: int, target: GantryTarget) -> GantryTarget:
    """Convert a physical/world gantry target into raw MuJoCo qpos space.

    P2's base is rotated 180 degrees around world-z.  Its raw lateral qpos
    therefore has the opposite sign from the physical world y coordinate.
    Its x qpos is also expressed in the mirrored local base coordinate.
    """
    if player not in (1, 2):
        raise ValueError("player must be 1 or 2")
    if player == 1:
        return target
    return GantryTarget(2 * 1.5 - target.x, -target.y)


def gantry_to_world_position(player: int, gantry: GantryTarget) -> GantryTarget:
    """Convert raw gantry qpos coordinates into physical/world coordinates."""
    return world_to_gantry_target(player, gantry)


__all__ = [
    "DEFENSE_POSITIONS",
    "DEFENSE_GANTRY_TARGETS",
    "GantryTarget",
    "gantry_to_world_position",
    "target_for_defender",
    "validate_defense_position",
    "world_to_gantry_target",
]
