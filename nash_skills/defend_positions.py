"""
Defend-position vocabulary for the attacker/defender extension (Design B).

Every player picks an attack skill (nash_skills.skills, unchanged) AND a
defend position each rally. The environment applies whichever half is
relevant each crossing based on who the ball is currently heading toward --
the receiver's defend-position choice moves their gantry, the hitter's
attack-skill choice becomes their shot. No role is ever declared; it falls
out of ball direction the same way the winner-assignment fix already infers
"receiver" from it.

Values come directly from the feature spec: defend position only moves the
gantry (x, y), nothing else. "Relative to the player's own qpos" is read
here as matching the gantry x-clamp precedent in mujoco_env_comp.py, which
applies the *same* numeric range to player 1's qpos[0:2] and player 2's
qpos[9:11] with no sign-flip or reflection -- unlike skills.world_target_xy,
which DOES mirror racket-aim targets for player 2. Defend position is a
gantry-qpos concept like the clamp, not a racket-aim concept like a skill
target, so no mirroring is applied here. This is the one real assumption in
this module; flagged in tests/test_defend_positions.py and easy to revisit
if the live wiring shows it's wrong.
"""

DEFEND_POSITIONS = {
    "left":   (-0.5,  0.35),
    "right":  (-0.5, -0.35),
    "center": (-0.5,  0.0),
}

DEFEND_POSITION_NAMES = list(DEFEND_POSITIONS.keys())
N_DEFEND_POSITIONS = len(DEFEND_POSITION_NAMES)


def get_defend_position(name: str):
    """Return (x, y) gantry qpos target for a defend-position name."""
    if name not in DEFEND_POSITIONS:
        raise ValueError(
            f"Unknown defend position '{name}'. Choose from: {DEFEND_POSITION_NAMES}"
        )
    return DEFEND_POSITIONS[name]


def defend_position_index(name: str) -> int:
    return DEFEND_POSITION_NAMES.index(name)


def defend_position_from_index(idx: int) -> str:
    return DEFEND_POSITION_NAMES[idx]


def world_defend_qpos(player: int, name: str):
    """
    Return the (x, y) gantry qpos target for a player taking this defend
    position. Unlike skills.world_target_xy, this does NOT mirror for
    player 2 -- see module docstring.
    """
    if player not in (1, 2):
        raise ValueError("player must be 1 or 2")
    return get_defend_position(name)
