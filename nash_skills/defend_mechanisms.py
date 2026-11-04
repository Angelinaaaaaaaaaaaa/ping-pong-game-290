"""
Switch between the two Design B defend mechanisms, so collection, evaluation
and self-play can run either one under an otherwise identical pipeline.

  reactive     the receiver's interception gets a fading lateral bias toward
               its defend position (nash_skills.defend_reactive_pose)
  preparation  the player who just hit moves to its defend position while the
               ball travels away from it (nash_skills.defend_preparation);
               matches the paper's attacker/defender roles
  preparation_cal  same as preparation, but the poses come from the
               per-profile calibration (nash_skills.prep_pose_tables)
"""

__all__ = [
    "DEFEND_MECHANISMS",
    "DEFAULT_DEFEND_MECHANISM",
    "validate_defend_mechanism",
    "set_defend_choices",
]

DEFEND_MECHANISMS = ("reactive", "preparation", "preparation_cal")
DEFAULT_DEFEND_MECHANISM = "reactive"


def validate_defend_mechanism(mechanism: str) -> str:
    if mechanism not in DEFEND_MECHANISMS:
        raise ValueError(f"Unknown defend mechanism '{mechanism}'. Choose from: {DEFEND_MECHANISMS}")
    return mechanism


def set_defend_choices(env, mechanism: str, pos1, pos2):
    validate_defend_mechanism(mechanism)
    if mechanism == "reactive":
        env.set_defend_reactive_positions(pos1, pos2)
    elif mechanism == "preparation":
        env.set_defend_preparation_positions(pos1, pos2)
    else:
        env.set_defend_preparation_positions(pos1, pos2, calibrated=True)
