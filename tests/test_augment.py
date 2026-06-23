"""
TDD regression tests for nash_skills/v2/augment.py

Tests written FIRST (RED phase) before implementation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from nash_skills.v2.augment import mirror_skill, flip_rally, augment_with_flip


# ─── helpers ────────────────────────────────────────────────────────────────

def _make_raw_obs(ego_gantry=(0.1, 0.2), opp_gantry=(0.3, 0.4),
                  ego_arm=None, opp_arm=None,
                  ball_pos=(1.5, 0.0, 0.3), ball_vel=(1.0, 0.0, -1.0),
                  ego_skill_val=0.0, opp_skill_val=1.0) -> np.ndarray:
    """Build a minimal 116-dim raw obs for testing."""
    obs = np.zeros(116, dtype=np.float32)
    obs[0:2]   = ego_gantry
    obs[2:9]   = ego_arm if ego_arm is not None else np.ones(7) * 0.1
    obs[9:18]  = np.ones(9) * 0.05   # ego qvel
    obs[18:20] = opp_gantry
    obs[20:27] = opp_arm if opp_arm is not None else np.ones(7) * 0.2
    obs[27:36] = np.ones(9) * 0.06   # opp qvel
    obs[36:39] = ball_pos
    obs[39:42] = ball_vel
    obs[42:78] = np.ones(36) * 0.11  # ego pose history
    obs[78:114] = np.ones(36) * 0.22  # opp pose history
    obs[114]   = ego_skill_val
    obs[115]   = opp_skill_val
    return obs


def _make_rally(skill1="left", skill2="right", winner=1, n_steps=2):
    raw_obs_list = [_make_raw_obs(ego_skill_val=0.0, opp_skill_val=1.0)
                    for _ in range(n_steps)]
    states = [np.zeros(76, dtype=np.float32) for _ in range(n_steps)]
    return {"skill1": skill1, "skill2": skill2,
            "states": states, "raw_obs": raw_obs_list, "winner": winner}


# ─── mirror_skill ────────────────────────────────────────────────────────────

class TestMirrorSkill:
    def test_left_mirrors_to_right(self):
        assert mirror_skill("left") == "right"

    def test_right_mirrors_to_left(self):
        assert mirror_skill("right") == "left"

    def test_left_short_mirrors_to_right_short(self):
        assert mirror_skill("left_short") == "right_short"

    def test_right_short_mirrors_to_left_short(self):
        assert mirror_skill("right_short") == "left_short"

    def test_center_safe_is_self_mirror(self):
        assert mirror_skill("center_safe") == "center_safe"

    def test_double_mirror_is_identity(self):
        for skill in ["left", "left_short", "center_safe", "right_short", "right"]:
            assert mirror_skill(mirror_skill(skill)) == skill, skill

    def test_unknown_skill_raises(self):
        with pytest.raises(KeyError):
            mirror_skill("bogus")


# ─── flip_rally ──────────────────────────────────────────────────────────────

class TestFlipRally:
    def test_returns_none_for_truncated_winner_zero(self):
        rally = _make_rally(winner=0)
        assert flip_rally(rally) is None

    def test_returns_none_for_none_winner(self):
        rally = _make_rally(winner=None)
        assert flip_rally(rally) is None

    def test_winner_1_becomes_2(self):
        rally = _make_rally(winner=1)
        flipped = flip_rally(rally)
        assert flipped["winner"] == 2

    def test_winner_2_becomes_1(self):
        rally = _make_rally(winner=2)
        flipped = flip_rally(rally)
        assert flipped["winner"] == 1

    def test_skills_are_mirrored_and_swapped(self):
        # ego=left (0.0), opp=right (1.0) → flip: ego=mirror(right)=left, opp=mirror(left)=right
        # But also skill1/skill2 swap: new_skill1=mirror(skill2), new_skill2=mirror(skill1)
        rally = _make_rally(skill1="left", skill2="right_short")
        flipped = flip_rally(rally)
        assert flipped["skill1"] == mirror_skill("right_short")  # = left_short
        assert flipped["skill2"] == mirror_skill("left")         # = right

    def test_flipped_state_has_correct_shape(self):
        rally = _make_rally(n_steps=3)
        flipped = flip_rally(rally)
        assert len(flipped["states"]) == 3
        assert np.array(flipped["states"][0]).shape == (76,)

    def test_flipped_ego_gantry_equals_original_opp_gantry(self):
        """Flipped ego kinematic state should come from original opp (obs[18:20])."""
        raw = _make_raw_obs(ego_gantry=(0.1, 0.2), opp_gantry=(0.3, 0.4))
        rally = {"skill1": "left", "skill2": "right",
                 "states": [np.zeros(76)], "raw_obs": [raw], "winner": 1}
        flipped = flip_rally(rally)
        state = np.array(flipped["states"][0])
        np.testing.assert_allclose(state[0:2], [0.3, 0.4], atol=1e-6)

    def test_flipped_ego_history_equals_original_opp_history(self):
        """Flipped ego history (dims 38:74) should come from original opp history (obs[78:114])."""
        raw = _make_raw_obs()
        rally = {"skill1": "left", "skill2": "right",
                 "states": [np.zeros(76)], "raw_obs": [raw], "winner": 1}
        flipped = flip_rally(rally)
        state = np.array(flipped["states"][0])
        np.testing.assert_allclose(state[38:74], raw[78:114], atol=1e-6)

    def test_flipped_ball_pos_unchanged(self):
        """Ball position is NOT mirrored (encode_opp uses same absolute coords)."""
        ball = (1.8, 0.2, 0.4)
        raw = _make_raw_obs(ball_pos=ball)
        rally = {"skill1": "left", "skill2": "right",
                 "states": [np.zeros(76)], "raw_obs": [raw], "winner": 1}
        flipped = flip_rally(rally)
        state = np.array(flipped["states"][0])
        np.testing.assert_allclose(state[18:21], ball, atol=1e-6)

    def test_flipped_skill_ego_is_mirrored_original_opp_skill(self):
        """Skill_ego in flipped state = mirror(original opp skill index)."""
        # opp skill was "right" (index 4, value 1.0) → mirror → "left" (index 0, value 0.0)
        raw = _make_raw_obs(ego_skill_val=0.0, opp_skill_val=1.0)
        rally = {"skill1": "left", "skill2": "right",
                 "states": [np.zeros(76)], "raw_obs": [raw], "winner": 1}
        flipped = flip_rally(rally)
        state = np.array(flipped["states"][0])
        # new ego skill = mirror("right") = "left" → index 0 → value 0.0
        assert state[74] == pytest.approx(0.0, abs=1e-5)

    def test_flipped_skill_opp_is_mirrored_original_ego_skill(self):
        """Skill_opp in flipped state = mirror(original ego skill index)."""
        # ego skill was "left_short" (index 1, value 0.25) → mirror → "right_short" (index 3, val 0.75)
        raw = _make_raw_obs(ego_skill_val=0.25, opp_skill_val=0.75)
        rally = {"skill1": "left_short", "skill2": "right_short",
                 "states": [np.zeros(76)], "raw_obs": [raw], "winner": 2}
        flipped = flip_rally(rally)
        state = np.array(flipped["states"][0])
        # new opp skill = mirror("left_short") = "right_short" → index 3 → value 0.75
        assert state[75] == pytest.approx(0.75, abs=1e-5)


# ─── augment_with_flip ───────────────────────────────────────────────────────

class TestAugmentWithFlip:
    def test_decided_rallies_are_doubled(self):
        rallies = [_make_rally(winner=1), _make_rally(winner=2)]
        result = augment_with_flip(rallies)
        assert len(result) == 4  # 2 original + 2 flipped

    def test_truncated_rally_kept_once(self):
        rallies = [_make_rally(winner=0)]
        result = augment_with_flip(rallies)
        assert len(result) == 1  # no flip for truncated

    def test_mixed_rallies(self):
        rallies = [
            _make_rally(winner=1),
            _make_rally(winner=0),
            _make_rally(winner=2),
        ]
        result = augment_with_flip(rallies)
        assert len(result) == 5  # 2 decided*2 + 1 truncated*1

    def test_ego_opp_wins_balanced_after_augmentation(self):
        """After augmenting symmetric data, ego wins should equal opp wins."""
        rallies = [
            _make_rally(skill1="left", skill2="right", winner=1),
            _make_rally(skill1="right", skill2="left", winner=1),
        ]
        result = augment_with_flip(rallies)
        ego_wins = sum(1 for r in result if r["winner"] == 1)
        opp_wins = sum(1 for r in result if r["winner"] == 2)
        assert ego_wins == opp_wins

    def test_original_rallies_unmodified(self):
        rally = _make_rally(skill1="left", skill2="right", winner=1)
        original_skill1 = rally["skill1"]
        augment_with_flip([rally])
        assert rally["skill1"] == original_skill1  # immutable: originals untouched


# ─── H1: raw_obs shape validation ───────────────────────────────────────────

class TestFlipRallyShapeValidation:
    def test_short_raw_obs_raises_value_error(self):
        """H1: raw_obs with fewer than 116 elements must raise ValueError."""
        bad_obs = np.zeros(100, dtype=np.float32)  # too short
        rally = {"skill1": "left", "skill2": "right",
                 "states": [np.zeros(76)], "raw_obs": [bad_obs], "winner": 1}
        with pytest.raises(ValueError, match="116"):
            flip_rally(rally)

    def test_exact_116_dim_does_not_raise(self):
        """H1: exactly 116-dim raw_obs must not raise."""
        good_obs = _make_raw_obs()
        assert good_obs.shape == (116,)
        rally = {"skill1": "left", "skill2": "right",
                 "states": [np.zeros(76)], "raw_obs": [good_obs], "winner": 1}
        result = flip_rally(rally)
        assert result is not None


# ─── L1: empty raw_obs list ──────────────────────────────────────────────────

class TestFlipRallyEmptyRawObs:
    def test_empty_raw_obs_returns_empty_states(self):
        """L1: rally with no crossing observations → flipped rally has no states."""
        rally = {"skill1": "left", "skill2": "right",
                 "states": [], "raw_obs": [], "winner": 1}
        flipped = flip_rally(rally)
        assert flipped is not None
        assert flipped["states"] == []
        assert flipped["winner"] == 2


# ─── H2: deep copy of raw_obs arrays ─────────────────────────────────────────

class TestFlipRallyDeepCopy:
    def test_mutating_flipped_raw_obs_does_not_affect_original(self):
        """H2: mutating the returned raw_obs must not affect the original rally."""
        raw = _make_raw_obs()
        rally = {"skill1": "left", "skill2": "right",
                 "states": [np.zeros(76)], "raw_obs": [raw], "winner": 1}
        flipped = flip_rally(rally)
        flipped["raw_obs"][0][0] = 999.0  # mutate the flipped copy
        assert rally["raw_obs"][0][0] != 999.0  # original must be unchanged


# ─── L2: tracking-error dims zeroed in augmented state ───────────────────────

class TestFlipRallyZeroFillTrackingDims:
    def test_tracking_error_dims_are_zero(self):
        """L2: dims [24:38] (diff_pos/diff_quat/target) must be zero in flipped state."""
        raw = _make_raw_obs()
        rally = {"skill1": "left", "skill2": "right",
                 "states": [np.zeros(76)], "raw_obs": [raw], "winner": 1}
        flipped = flip_rally(rally)
        state = np.array(flipped["states"][0])
        np.testing.assert_array_equal(state[24:38], np.zeros(14))
