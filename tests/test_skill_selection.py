"""Tests for nash_skills.v2.skill_selection — probabilistic skill selection.

TDD: written before implementation. All tests should FAIL until
nash_skills/v2/skill_selection.py is created.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from nash_skills.v2.skill_selection import (
    softmax_probs,
    epsilon_mix_probs,
    select_skill_from_values,
)


class TestSoftmaxProbs:
    def test_output_sums_to_one(self):
        probs = softmax_probs(np.array([1.0, 2.0, 3.0]))
        assert abs(probs.sum() - 1.0) < 1e-9

    def test_output_shape_matches_input(self):
        probs = softmax_probs(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert probs.shape == (5,)

    def test_higher_value_gets_higher_prob(self):
        probs = softmax_probs(np.array([1.0, 3.0]))
        assert probs[1] > probs[0]

    def test_equal_values_give_uniform(self):
        probs = softmax_probs(np.array([2.0, 2.0, 2.0]))
        np.testing.assert_allclose(probs, [1 / 3, 1 / 3, 1 / 3], atol=1e-9)

    def test_very_low_temperature_approaches_argmax(self):
        probs = softmax_probs(np.array([1.0, 10.0, 2.0]), temperature=0.01)
        assert probs[1] > 0.999

    def test_very_high_temperature_approaches_uniform(self):
        probs = softmax_probs(np.array([1.0, 10.0, 2.0]), temperature=1000.0)
        np.testing.assert_allclose(probs, [1 / 3, 1 / 3, 1 / 3], atol=0.01)

    def test_temperature_one_is_standard_softmax(self):
        vals = np.array([1.0, 2.0, 3.0])
        probs = softmax_probs(vals, temperature=1.0)
        shifted = vals - vals.max()
        exp = np.exp(shifted)
        expected = exp / exp.sum()
        np.testing.assert_allclose(probs, expected, rtol=1e-6)

    def test_accepts_list_input(self):
        probs = softmax_probs([1.0, 2.0, 3.0])
        assert abs(probs.sum() - 1.0) < 1e-9

    def test_raises_for_zero_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            softmax_probs(np.array([1.0, 2.0]), temperature=0.0)

    def test_raises_for_negative_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            softmax_probs(np.array([1.0, 2.0]), temperature=-1.0)

    def test_numerical_stability_large_values(self):
        probs = softmax_probs(np.array([1000.0, 1001.0, 1002.0]))
        assert abs(probs.sum() - 1.0) < 1e-9
        assert all(probs >= 0)


class TestEpsilonMixProbs:
    def test_epsilon_zero_returns_base_probs(self):
        base = np.array([0.8, 0.1, 0.1])
        result = epsilon_mix_probs(base, epsilon=0.0, num_skills=3)
        np.testing.assert_allclose(result, base, rtol=1e-9)

    def test_epsilon_one_returns_uniform(self):
        base = np.array([0.9, 0.05, 0.05])
        result = epsilon_mix_probs(base, epsilon=1.0, num_skills=3)
        np.testing.assert_allclose(result, [1 / 3, 1 / 3, 1 / 3], atol=1e-9)

    def test_epsilon_half_is_midpoint(self):
        base = np.array([1.0, 0.0])
        result = epsilon_mix_probs(base, epsilon=0.5, num_skills=2)
        np.testing.assert_allclose(result, [0.75, 0.25], atol=1e-9)

    def test_output_sums_to_one(self):
        base = np.array([0.6, 0.3, 0.1])
        result = epsilon_mix_probs(base, epsilon=0.3, num_skills=3)
        assert abs(result.sum() - 1.0) < 1e-9

    def test_raises_for_negative_epsilon(self):
        with pytest.raises(ValueError, match="epsilon"):
            epsilon_mix_probs(np.array([0.5, 0.5]), epsilon=-0.1, num_skills=2)

    def test_raises_for_epsilon_above_one(self):
        with pytest.raises(ValueError, match="epsilon"):
            epsilon_mix_probs(np.array([0.5, 0.5]), epsilon=1.1, num_skills=2)

    def test_output_shape(self):
        base = np.array([0.2, 0.5, 0.3])
        result = epsilon_mix_probs(base, epsilon=0.5, num_skills=3)
        assert result.shape == (3,)


class TestSelectSkillFromValues:
    def test_argmax_returns_highest_index(self):
        vals = np.array([0.1, 0.9, 0.3, 0.2, 0.5])
        idx = select_skill_from_values(vals, mode="argmax")
        assert idx == 1

    def test_argmax_is_default_mode(self):
        vals = np.array([0.1, 0.9, 0.3])
        idx = select_skill_from_values(vals)
        assert idx == 1

    def test_argmax_is_deterministic(self):
        vals = np.array([1.0, 5.0, 2.0])
        results = {select_skill_from_values(vals, mode="argmax") for _ in range(20)}
        assert results == {1}

    def test_softmax_returns_valid_index(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rng = np.random.default_rng(42)
        for _ in range(20):
            idx = select_skill_from_values(vals, mode="softmax", rng=rng)
            assert 0 <= idx < 5

    def test_softmax_samples_all_skills_with_uniform_values(self):
        vals = np.array([1.0, 1.0, 1.0])
        rng = np.random.default_rng(0)
        counts = [0, 0, 0]
        for _ in range(300):
            counts[select_skill_from_values(vals, mode="softmax", rng=rng)] += 1
        for c in counts:
            assert 30 < c < 200, f"Expected ~100, got {c}"

    def test_epsilon_argmax_with_zero_epsilon_is_argmax(self):
        vals = np.array([1.0, 5.0, 2.0])
        rng = np.random.default_rng(0)
        results = {
            select_skill_from_values(vals, mode="epsilon_argmax", epsilon=0.0, rng=rng)
            for _ in range(20)
        }
        assert results == {1}

    def test_epsilon_argmax_with_full_epsilon_is_random(self):
        vals = np.array([1.0, 100.0, 1.0])
        rng = np.random.default_rng(0)
        seen = set()
        for _ in range(200):
            seen.add(select_skill_from_values(vals, mode="epsilon_argmax", epsilon=1.0, rng=rng))
        assert len(seen) > 1

    def test_epsilon_softmax_with_zero_epsilon_concentrates_on_best(self):
        vals = np.array([0.0, 0.0, 10.0])
        rng = np.random.default_rng(0)
        counts = [0, 0, 0]
        for _ in range(200):
            counts[select_skill_from_values(vals, mode="epsilon_softmax", temperature=0.1, epsilon=0.0, rng=rng)] += 1
        assert counts[2] > 180

    def test_epsilon_softmax_with_full_epsilon_is_uniform(self):
        vals = np.array([1.0, 100.0, 2.0])
        rng = np.random.default_rng(42)
        counts = [0, 0, 0]
        for _ in range(300):
            counts[select_skill_from_values(vals, mode="epsilon_softmax", epsilon=1.0, rng=rng)] += 1
        for c in counts:
            assert 30 < c < 200, f"Expected ~100, got {c}"

    def test_accepts_list_input(self):
        idx = select_skill_from_values([1.0, 3.0, 2.0], mode="argmax")
        assert idx == 1

    def test_accepts_torch_tensor(self):
        try:
            import torch
            vals = torch.tensor([1.0, 5.0, 2.0])
            idx = select_skill_from_values(vals, mode="argmax")
            assert idx == 1
        except ImportError:
            pytest.skip("torch not available")

    def test_unknown_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="mode"):
            select_skill_from_values(np.array([1.0, 2.0]), mode="unknown")

    def test_rng_none_uses_internal_default(self):
        vals = np.array([1.0, 5.0, 2.0])
        idx = select_skill_from_values(vals, mode="argmax", rng=None)
        assert idx == 1

    def test_softmax_with_same_seed_is_reproducible(self):
        vals = np.array([1.0, 2.0, 3.0])
        idx1 = select_skill_from_values(vals, mode="softmax", rng=np.random.default_rng(7))
        idx2 = select_skill_from_values(vals, mode="softmax", rng=np.random.default_rng(7))
        assert idx1 == idx2

    def test_returns_plain_int(self):
        idx = select_skill_from_values(np.array([1.0, 2.0, 3.0]), mode="argmax")
        assert type(idx) is int
