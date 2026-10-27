"""Defender positions for the attacker/defender variant.

Per Kexin's spec: at each net crossing both players simultaneously pick an
attacking skill *and* a defender position, regardless of which role they turn
out to hold. The skill applies when you are the one hitting; the position
applies while you are waiting for the ball. Only the gantry moves -- the arm
keeps doing whatever the tracker asks of it.

Naming follows v3: a position is named from the defender's own physical
perspective, so P1's ``left`` is +world-y and P2's ``left`` is -world-y. Both
reduce to qpos y = +0.35, because P2's base is rotated 180 degrees and its raw
lateral qpos already carries the opposite sign from world y.

`nash_skills.skills` predates that convention and gives the *opposite* lateral
sign: its ``left`` lands at -world-y. Rather than renumber skills.py -- which
would invalidate every Phi/Q checkpoint and every collected dataset, all trained
under the legacy signs -- the two conventions are reconciled here, the same way
v3 does it with its own skill_semantics module. Use `attack_side_world()` when
comparing where a shot goes against where a defender stands.
"""

# name -> (gantry x, gantry y) in the player's own qpos frame
DEFENDER_POSITIONS = {
    "def_left":   (-0.5, +0.35),
    "def_center": (-0.5,  0.00),
    "def_right":  (-0.5, -0.35),
}
DEFENDER_NAMES = list(DEFENDER_POSITIONS)
N_DEFENDER = len(DEFENDER_NAMES)


def defender_index(name: str) -> int:
    return DEFENDER_NAMES.index(name)


def defender_from_index(i: int) -> str:
    return DEFENDER_NAMES[int(i) % N_DEFENDER]


def defender_target(name: str):
    return DEFENDER_POSITIONS[name]


def joint_size(n_skills: int) -> int:
    """Number of (skill, position) pairs."""
    return n_skills * N_DEFENDER


def split_joint(idx: int, n_skills: int):
    """Joint action index -> (skill index, defender index).

    Skill is the fast-varying half so that a joint policy initialised uniformly
    still spreads evenly over skills, which keeps the early-training skill
    statistics comparable with the 5-skill runs.
    """
    idx = int(idx)
    return idx % n_skills, idx // n_skills


def join_index(skill_idx: int, defender_idx: int, n_skills: int) -> int:
    return int(defender_idx) * n_skills + int(skill_idx)


# --- reconciling the two lateral conventions ---------------------------------
# skills.py: `left` has side_target < 0, i.e. the ball lands at negative world y.
# Defender names (and v3's skills): `left` means positive world y for P1.
# So a skill's name and a defender position's name refer to opposite sides.

def attack_side_world(skill_name: str, profile: str = "aggressive") -> float:
    """World-y sign where `skill_name` lands the ball, for player 1.

    Positive means the same side that `def_left` stands on. Use this instead of
    comparing the two names directly, which would silently be off by a mirror.
    """
    from nash_skills.env_wrapper import SkillEnv
    # Go through the env's accessor, not skills.get_skill, so this follows
    # PINGPONG_SKILL_SEMANTICS.  Reading skills.py directly would report the
    # legacy sign while the env is actually playing the flipped one, and every
    # "did the shot go where the defender stood" answer would be mirrored.
    side, _x = SkillEnv._get_skill(skill_name, profile)
    return float(side)


def defender_side_world(position: str) -> float:
    """World-y sign of defender position `position`, for player 1."""
    return float(DEFENDER_POSITIONS[position][1])


def attack_covers_defense(skill_name: str, position: str,
                          profile: str = "aggressive") -> bool:
    """Whether a shot is aimed at the half the defender has committed to.

    Both are reduced to a world-y sign first, so the legacy/v3 naming mismatch
    cannot flip the answer.
    """
    a = attack_side_world(skill_name, profile)
    d = defender_side_world(position)
    if a == 0.0 or d == 0.0:
        return False
    return (a > 0) == (d > 0)
