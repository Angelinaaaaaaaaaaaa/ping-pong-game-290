"""Tests for nash_skills.defend_soft_bias.blend_gantry_command.

TDD: written before implementation.

The hard-override mechanism (nash_skills.defend_steering +
defend_engagement, bangbang_release -- the confirmed-best of 4 tested
controllers) was found net-harmful across all 12 cells of a 1200-rally
matched-pair experiment (avg_delta_tested_wr=-0.243,
avg_delta_contacts=-2.065), harm persisting even when the override
released control back to the PPO well before the rally ended. This
module tests a structurally different design: ADD a gentle proportional
pull toward the defend target to the PPO's own gantry command, instead
of REPLACING it.

final_command = clip(ppo_command + alpha * Kp * (target - current), -1, 1)

alpha=0 must recover pure PPO control exactly -- a built-in sanity
check for the alpha sweep this feeds into.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from nash_skills.defend_soft_bias import blend_gantry_command, DEFAULT_KP


class TestAlphaZeroRecoversPurePPO:
    def test_final_command_equals_ppo_command_at_alpha_zero(self):
        final, mag = blend_gantry_command(
            ppo_command=(0.3, -0.2), current_qpos=(-0.5, 0.0), target_qpos=(-0.5, 0.35), alpha=0.0,
        )
        assert final == (0.3, -0.2)
        assert mag == 0.0

    def test_alpha_zero_regardless_of_how_far_from_target(self):
        final, mag = blend_gantry_command(
            ppo_command=(0.1, 0.1), current_qpos=(-0.5, -5.0), target_qpos=(-0.5, 5.0), alpha=0.0,
        )
        assert final == (0.1, 0.1)
        assert mag == 0.0


class TestBlendDirection:
    def test_bias_pulls_toward_target_not_away(self):
        # target is +y of current -- bias on y must be positive
        final, _ = blend_gantry_command(
            ppo_command=(0.0, 0.0), current_qpos=(-0.5, 0.0), target_qpos=(-0.5, 0.35), alpha=0.1,
        )
        assert final[1] > 0.0

    def test_already_at_target_gives_zero_bias(self):
        final, mag = blend_gantry_command(
            ppo_command=(0.2, -0.1), current_qpos=(-0.5, 0.35), target_qpos=(-0.5, 0.35), alpha=0.5,
        )
        assert final == pytest.approx((0.2, -0.1))
        assert mag == pytest.approx(0.0)


class TestAlphaMonotonicity:
    def test_larger_alpha_gives_larger_intervention_magnitude(self):
        # Fixed error, increasing alpha -- intervention should grow
        # (while unclipped).
        _, mag_small = blend_gantry_command(
            ppo_command=(0.0, 0.0), current_qpos=(-0.5, 0.0), target_qpos=(-0.5, 0.1), alpha=0.1,
        )
        _, mag_large = blend_gantry_command(
            ppo_command=(0.0, 0.0), current_qpos=(-0.5, 0.0), target_qpos=(-0.5, 0.1), alpha=0.25,
        )
        assert mag_large > mag_small


class TestClipping:
    def test_final_command_never_exceeds_unit_bounds(self):
        final, _ = blend_gantry_command(
            ppo_command=(0.9, 0.9), current_qpos=(-0.5, -10.0), target_qpos=(-0.5, 10.0), alpha=0.5,
        )
        assert -1.0 <= final[0] <= 1.0
        assert -1.0 <= final[1] <= 1.0

    def test_intervention_magnitude_reflects_post_clip_deviation_not_raw_bias(self):
        # A huge raw bias that saturates the clip must report the ACTUAL
        # (post-clip) deviation from ppo_command, not the unclipped
        # wished-for bias -- this is what "how hard did it actually
        # push" needs to mean for the alpha sweep to be interpretable.
        final, mag = blend_gantry_command(
            ppo_command=(0.0, 0.0), current_qpos=(-0.5, -100.0), target_qpos=(-0.5, 100.0), alpha=1.0,
        )
        assert final[1] == pytest.approx(1.0)  # saturated
        assert mag == pytest.approx(1.0)  # actual deviation from ppo_command=(0,0) is exactly 1.0, not huge


class TestDefaultKp:
    def test_default_kp_matches_validated_proportional_gain(self):
        assert DEFAULT_KP == 8.0

    def test_kp_is_overridable(self):
        final_default, _ = blend_gantry_command(
            ppo_command=(0.0, 0.0), current_qpos=(-0.5, 0.0), target_qpos=(-0.5, 0.05), alpha=0.1,
        )
        final_custom, _ = blend_gantry_command(
            ppo_command=(0.0, 0.0), current_qpos=(-0.5, 0.0), target_qpos=(-0.5, 0.05), alpha=0.1, kp=16.0,
        )
        assert final_custom[1] > final_default[1]
