"""Tests for scripts/calibrate_prep_poses.py (pure aggregation only)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from calibrate_prep_poses import ATTACK_FOR_POSITION, summarize_contacts  # noqa: E402


def _rec(attack, xyz, err=0.01):
    return {"attack": attack, "target": list(xyz), "track_err": err}


def test_attack_for_position_covers_all_three_positions():
    assert ATTACK_FOR_POSITION == {"left": "left", "center": "center_safe", "right": "right"}


def test_summary_is_median_per_position_over_converged_contacts():
    recs = [_rec("left", (-0.1, 0.6, 0.8)), _rec("left", (-0.2, 0.7, 0.9)), _rec("left", (-0.3, 0.8, 1.0)),
            _rec("right", (-0.1, -0.7, 0.8)), _rec("center_safe", (-0.1, 0.05, 0.85))]
    out = summarize_contacts(recs, max_track_err=0.05)
    assert out["left"]["y"] == pytest.approx(0.7)
    assert out["left"]["x"] == pytest.approx(-0.2)
    assert out["left"]["n"] == 3
    assert out["right"]["y"] == pytest.approx(-0.7)
    assert out["center"]["y"] == pytest.approx(0.05)


def test_unconverged_contacts_are_dropped():
    recs = [_rec("left", (0, 0.6, 0.8)), _rec("left", (0, -0.5, 0.8), err=0.2),
            _rec("center_safe", (0, 0, 0.8)), _rec("right", (0, -0.7, 0.8))]
    out = summarize_contacts(recs, max_track_err=0.05)
    assert out["left"]["n"] == 1
    assert out["left"]["y"] == pytest.approx(0.6)


def test_shared_x_z_are_medians_over_all_positions():
    recs = [_rec("left", (-0.1, 0.6, 0.8)), _rec("right", (-0.3, -0.7, 0.9)), _rec("center_safe", (-0.2, 0, 1.0))]
    out = summarize_contacts(recs, max_track_err=0.05)
    assert out["shared"]["x"] == pytest.approx(-0.2)
    assert out["shared"]["z"] == pytest.approx(0.9)


def test_mean_orientation_per_position_when_quaternions_are_recorded():
    from scipy.spatial.transform import Rotation as R
    q_left = R.from_euler("z", 40, degrees=True).as_quat().tolist()
    q_right = R.from_euler("z", -40, degrees=True).as_quat().tolist()
    recs = [dict(_rec("left", (0, 0.6, 0.8)), quat=q_left), dict(_rec("left", (0, 0.6, 0.8)), quat=q_left),
            dict(_rec("right", (0, -0.7, 0.8)), quat=q_right), dict(_rec("center_safe", (0, 0, 0.8)), quat=[0, 0, 0, 1])]
    out = summarize_contacts(recs, max_track_err=0.05)
    angle = R.from_quat(out["left"]["quat"]).as_euler("xyz", degrees=True)[2]
    assert angle == pytest.approx(40, abs=1e-6)
    assert out["left"]["quat_spread_deg"] == pytest.approx(0.0, abs=1e-6)
    assert R.from_quat(out["right"]["quat"]).as_euler("xyz", degrees=True)[2] == pytest.approx(-40, abs=1e-6)
    q = out["left"]["quat"]
    np.testing.assert_allclose(q, R.from_matrix(R.from_quat(q).as_matrix()).as_quat(), atol=1e-9)


def test_missing_position_raises():
    with pytest.raises(ValueError, match="right"):
        summarize_contacts([_rec("left", (0, 0.6, 0.8)), _rec("center_safe", (0, 0, 0.8))], max_track_err=0.05)
