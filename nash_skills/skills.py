"""
Skill definitions for the expanded Nash equilibrium pipeline.

Each skill is a (side_target, x_target) pair:
  - side_target: lateral aim, multiplied by 0.38m  (negative=left, 0=center, positive=right)
  - x_target:   depth aim on opponent's side of the table (meters, world frame)
                 TABLE_SHIFT=1.5, table runs from 1.5 to 2.87
                 ~1.75 = short (near net), ~2.19 = deep (far end)

The environment's step() already calls:
    update_target_racket_pose(y_target=side_target * 0.38)
We extend this by also passing x_target when applying a skill.
"""

TABLE_SHIFT = 1.5       # net position (x)
TABLE_NEAR  = 1.75      # short return: raised from 1.65 (simulator validation showed
                        # PPO cannot track 1.65; left_short landed 0.20m deeper than left)
TABLE_MID   = 1.85      # center-safe: middle of opponent's half
TABLE_FAR   = TABLE_SHIFT + 1.37 / 2  # ~2.19, deep return (existing default)
TABLE_SHORT_MODERATE = 1.66
TABLE_SHORT_AGGRESSIVE = 1.60
TABLE_FAR_MODERATE = 2.35
TABLE_FAR_AGGRESSIVE = 2.50
SIDE_CURRENT = 1.0
SIDE_MODERATE = 1.45       # y ~= +/-0.55, inside table half-width 0.7625
SIDE_AGGRESSIVE = 1.70     # y ~= +/-0.65, still bounded inside table

# Skill name -> (side_target, x_target)
# Order is semantically left-to-right so normalised indices (idx/(N-1)) are
# geometrically meaningful: 0.0=far-left, 0.5=center, 1.0=far-right.
SKILL_PROFILES = {
    "current": {
        "left":         (-SIDE_CURRENT, TABLE_FAR),   # deep left  (original)
        "left_short":   (-SIDE_CURRENT, TABLE_NEAR),  # short left
        "center_safe":  ( 0.0, TABLE_MID),            # center, mid-depth (conservative)
        "right_short":  ( SIDE_CURRENT, TABLE_NEAR),  # short right
        "right":        ( SIDE_CURRENT, TABLE_FAR),   # deep right (original)
    },
    "moderate_aggressive": {
        "left":         (-SIDE_MODERATE, TABLE_FAR_MODERATE),
        "left_short":   (-SIDE_MODERATE, TABLE_SHORT_MODERATE),
        "center_safe":  ( 0.0, TABLE_MID),
        "right_short":  ( SIDE_MODERATE, TABLE_SHORT_MODERATE),
        "right":        ( SIDE_MODERATE, TABLE_FAR_MODERATE),
    },
    "aggressive": {
        "left":         (-SIDE_AGGRESSIVE, TABLE_FAR_AGGRESSIVE),
        "left_short":   (-SIDE_AGGRESSIVE, TABLE_SHORT_AGGRESSIVE),
        "center_safe":  ( 0.0, TABLE_MID),
        "right_short":  ( SIDE_AGGRESSIVE, TABLE_SHORT_AGGRESSIVE),
        "right":        ( SIDE_AGGRESSIVE, TABLE_FAR_AGGRESSIVE),
    },
}

SKILLS = SKILL_PROFILES["current"]
SKILL_PROFILE_NAMES = list(SKILL_PROFILES.keys())
SKILL_NAMES = list(SKILLS.keys())
N_SKILLS = len(SKILL_NAMES)


def get_skill_profile(profile: str):
    if profile not in SKILL_PROFILES:
        raise ValueError(f"Unknown skill profile '{profile}'. Choose from: {SKILL_PROFILE_NAMES}")
    return SKILL_PROFILES[profile]


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

_SKILL_ORDER_CHECK = {
    "left":         (-1.0, TABLE_FAR),   # deep left  (original)
    "left_short":   (-1.0, TABLE_NEAR),  # short left
    "center_safe":  ( 0.0, TABLE_MID),   # center, mid-depth (conservative)
    "right_short":  ( 1.0, TABLE_NEAR),  # short right
    "right":        ( 1.0, TABLE_FAR),   # deep right (original)
}


def get_skill(name: str, profile: str = "current"):
    """Return (side_target, x_target) for a skill name."""
    skills = get_skill_profile(profile)
    if name not in skills:
        raise ValueError(f"Unknown skill '{name}'. Choose from: {SKILL_NAMES}")
    return skills[name]


def skill_index(name: str) -> int:
    return SKILL_NAMES.index(name)


def skill_from_index(idx: int) -> str:
    return SKILL_NAMES[idx]
