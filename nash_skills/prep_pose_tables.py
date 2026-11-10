"""
Calibrated preparation poses, one table per skill profile.

Each position is the median racket target at contact when the receiver
returns the matching attack (left -> "left", center_safe -> "center",
right -> "right"), measured with scripts/calibrate_prep_poses.py using the
controller whose md5 is PREP_POSE_CONTROLLER_MD5 (the retrained tracker the
server runs use). Raw results: analysis/prep_pose_calibration/<profile>.json.

The poses are left/right asymmetric because the shots are: with symmetric
aim (+/-side * 0.38 m) the controller lands right shots wider than left shots,
and center_safe lands slightly right of 0.

Used by the "preparation_cal" defend mechanism. The legacy "preparation"
mechanism keeps nash_skills.defend_target_pose's single (aggressive-profile,
April-controller) table so existing data and models stay reproducible.
"""
import numpy as np

from nash_skills.defend_positions import DEFEND_POSITION_NAMES
from nash_skills.defend_target_pose import _DEFEND_QUAT

__all__ = ["CALIBRATED_PREP_POSES", "CALIBRATED_PREP_QUATS", "PREP_ORIENTATIONS", "PREP_POSE_CONTROLLER_MD5",
           "calibrated_prep_pose"]

PREP_ORIENTATIONS = ("fixed", "calibrated")

PREP_POSE_CONTROLLER_MD5 = "db5df579165070c7db59481776221bc2"

# (x, y, z) in the player-1 frame, see analysis/prep_pose_calibration/.
CALIBRATED_PREP_POSES = {
    "aggressive": {
        "left": (0.006, 0.664, 0.880),
        "center": (-0.133, 0.124, 0.804),
        "right": (-0.044, -0.743, 0.874),
    },
    "moderate_aggressive": {
        "left": (-0.289, 0.792, 0.804),
        "center": (-0.133, 0.124, 0.804),
        "right": (-0.200, -0.872, 0.838),
    },
}


# Mean racket orientation at contact (scalar-last quaternion), see
# analysis/prep_pose_calibration/<profile>_orient.json. Stored in the env's sign convention
# (the form R.from_matrix(...).as_quat() gives, largest component positive): the controller
# network reads the raw quaternion, and the same rotation with the opposite sign is an input
# it never saw (tracking error ~77 degrees instead of ~3).
CALIBRATED_PREP_QUATS = {
    "aggressive": {
        "left": (0.69984244, 0.07758186, 0.57944247, 0.41042423),
        "center": (0.71304180, 0.02338964, 0.69602838, 0.08104817),
        "right": (0.69540316, -0.08689889, 0.57034397, -0.42845160),
    },
    "moderate_aggressive": {
        "left": (0.70771460, 0.11875307, 0.54312326, 0.43595284),
        "center": (0.71304180, 0.02338964, 0.69602838, 0.08104817),
        "right": (0.70718088, -0.11409511, 0.53375420, -0.44942626),
    },
}


def calibrated_prep_pose(position_name: str, profile: str, orientation: str = "fixed") -> np.ndarray:
    """7-dim [x, y, z, qx, qy, qz, qw] preparation pose, player-1 frame.

    orientation "fixed": the face-forward quaternion of nash_skills.defend_target_pose;
    "calibrated": the mean racket orientation at contact for this position
    (CALIBRATED_PREP_QUATS, scalar-last like the env's targets)."""
    if orientation not in PREP_ORIENTATIONS:
        raise ValueError(f"Unknown orientation '{orientation}'. Choose from: {PREP_ORIENTATIONS}")
    if profile not in CALIBRATED_PREP_POSES:
        raise ValueError(f"No calibrated preparation poses for skill profile '{profile}'. "
                         f"Calibrated: {sorted(CALIBRATED_PREP_POSES)}")
    if position_name not in DEFEND_POSITION_NAMES:
        raise ValueError(f"Unknown defend position '{position_name}'. Choose from: {DEFEND_POSITION_NAMES}")
    quat = _DEFEND_QUAT if orientation == "fixed" else np.array(CALIBRATED_PREP_QUATS[profile][position_name], dtype=float)
    return np.concatenate([np.array(CALIBRATED_PREP_POSES[profile][position_name], dtype=float), quat])
