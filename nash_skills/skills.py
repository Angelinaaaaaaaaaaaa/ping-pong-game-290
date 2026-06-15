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

# Skill name -> (side_target, x_target)
# Order is semantically left-to-right so normalised indices (idx/(N-1)) are
# geometrically meaningful: 0.0=far-left, 0.5=center, 1.0=far-right.
SKILLS = {
    "left":         (-1.0, TABLE_FAR),   # deep left  (original)
    "left_short":   (-1.0, TABLE_NEAR),  # short left
    "center_safe":  ( 0.0, TABLE_MID),   # center, mid-depth (conservative)
    "right_short":  ( 1.0, TABLE_NEAR),  # short right
    "right":        ( 1.0, TABLE_FAR),   # deep right (original)
}

SKILL_NAMES = list(SKILLS.keys())
N_SKILLS = len(SKILL_NAMES)


def get_skill(name: str):
    """Return (side_target, x_target) for a skill name."""
    if name not in SKILLS:
        raise ValueError(f"Unknown skill '{name}'. Choose from: {SKILL_NAMES}")
    return SKILLS[name]


def skill_index(name: str) -> int:
    return SKILL_NAMES.index(name)


def skill_from_index(idx: int) -> str:
    return SKILL_NAMES[idx]
