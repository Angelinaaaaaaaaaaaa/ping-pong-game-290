"""V3 skill geometry using each attacker's own left/right convention.

The legacy five-skill module uses lateral signs that predate the V3 semantic
convention.  V3 keeps the same skill names, depth targets, and one-hot index
ordering, but defines local ``+y`` as the acting player's physical left:

* ``left``/``left_short`` -> local ``+y``
* ``right``/``right_short`` -> local ``-y``

Player 2's local frame is mirrored into world coordinates by the existing
table transform, so its left is world ``-y`` while Player 1's left is world
``+y``.
"""

from __future__ import annotations

from nash_skills.skills import (
    N_SKILLS,
    SIDE_AGGRESSIVE,
    SIDE_CURRENT,
    SIDE_MODERATE,
    SKILL_NAMES,
    SKILL_PROFILE_NAMES,
    TABLE_FAR,
    TABLE_FAR_AGGRESSIVE,
    TABLE_FAR_MODERATE,
    TABLE_MID,
    TABLE_NEAR,
    TABLE_SHORT_MODERATE,
    TABLE_SHORT_AGGRESSIVE,
    TABLE_SHIFT,
    skill_from_index,
    skill_index,
)


SKILL_PROFILES = {
    "current": {
        "left": (SIDE_CURRENT, TABLE_FAR),
        "left_short": (SIDE_CURRENT, TABLE_NEAR),
        "center_safe": (0.0, TABLE_MID),
        "right_short": (-SIDE_CURRENT, TABLE_NEAR),
        "right": (-SIDE_CURRENT, TABLE_FAR),
    },
    "moderate_aggressive": {
        "left": (SIDE_MODERATE, TABLE_FAR_MODERATE),
        "left_short": (SIDE_MODERATE, TABLE_SHORT_MODERATE),
        "center_safe": (0.0, TABLE_MID),
        "right_short": (-SIDE_MODERATE, TABLE_SHORT_MODERATE),
        "right": (-SIDE_MODERATE, TABLE_FAR_MODERATE),
    },
    "aggressive": {
        "left": (SIDE_AGGRESSIVE, TABLE_FAR_AGGRESSIVE),
        "left_short": (SIDE_AGGRESSIVE, TABLE_SHORT_AGGRESSIVE),
        "center_safe": (0.0, TABLE_MID),
        "right_short": (-SIDE_AGGRESSIVE, TABLE_SHORT_AGGRESSIVE),
        "right": (-SIDE_AGGRESSIVE, TABLE_FAR_AGGRESSIVE),
    },
}

SKILLS = SKILL_PROFILES["current"]


def get_skill_profile(profile: str):
    if profile not in SKILL_PROFILES:
        raise ValueError(f"Unknown skill profile '{profile}'. Choose from: {SKILL_PROFILE_NAMES}")
    return SKILL_PROFILES[profile]


def get_skill(name: str, profile: str = "current"):
    skills = get_skill_profile(profile)
    if name not in skills:
        raise ValueError(f"Unknown skill '{name}'. Choose from: {SKILL_NAMES}")
    return skills[name]


def skill_target_xy(name: str, profile: str = "current"):
    side, x_target = get_skill(name, profile=profile)
    return float(x_target), float(side * 0.38)


def world_target_xy(player: int, name: str, profile: str = "current"):
    x_target, y_target = skill_target_xy(name, profile=profile)
    if player == 1:
        return x_target, y_target
    if player == 2:
        return 2 * TABLE_SHIFT - x_target, -y_target
    raise ValueError("player must be 1 or 2")


def skill_player_relative_side(name: str, profile: str = "current") -> str:
    """Return the semantic side of a V3 skill from the attacker's view."""
    side, _x_target = get_skill(name, profile=profile)
    if side > 0:
        return "left"
    if side < 0:
        return "right"
    return "center"


def profile_target_rows(profile: str = "current"):
    rows = []
    for skill in SKILL_NAMES:
        side, x_target = get_skill(skill, profile=profile)
        local_x, local_y = skill_target_xy(skill, profile=profile)
        p1_x, p1_y = world_target_xy(1, skill, profile=profile)
        p2_x, p2_y = world_target_xy(2, skill, profile=profile)
        rows.append({
            "profile": profile,
            "skill": skill,
            "side_target": side,
            "player_local_x": local_x,
            "player_local_y": local_y,
            "p1_world_x": p1_x,
            "p1_world_y": p1_y,
            "p2_world_x": p2_x,
            "p2_world_y": p2_y,
        })
    return rows


__all__ = [
    "N_SKILLS",
    "SKILL_NAMES",
    "SKILL_PROFILE_NAMES",
    "SKILL_PROFILES",
    "SKILLS",
    "get_skill",
    "get_skill_profile",
    "profile_target_rows",
    "skill_from_index",
    "skill_index",
    "skill_player_relative_side",
    "skill_target_xy",
    "world_target_xy",
]
