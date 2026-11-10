"""Tests for nash_skills.v2.mixed_strategy and
joint_action.write_mixed_strategy_pair.

TDD: written before implementation.

Paper Sec 4.1 compact policy family: theta^i = (w^{A,i}, w^{D,i}) in
Delta_5 x Delta_3. The existing 90-dim layout's one-hot action blocks are
exactly theta evaluated at simplex VERTICES -- writing probability vectors
into the same slots is the generalization. Pure Python/torch, no MuJoCo.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from nash_skills.v2.joint_action import (
    write_joint_action_pair,
    write_mixed_strategy_pair,
)
from nash_skills.v2.mixed_strategy import (
    project_to_simplex,
    sample_theta,
    sample_action_from_theta,
    pga_maximize,
    theta_vertex,
)
from nash_skills.v2.state_encoder_attack_defend import STATE_DIM_ATTACK_DEFEND
from nash_skills.skills import N_SKILLS
from nash_skills.defend_positions import N_DEFEND_POSITIONS


def _onehot(n, i):
    v = torch.zeros(n)
    v[i] = 1.0
    return v


class TestWriteMixedStrategyPair:
    def test_one_hot_inputs_reproduce_write_joint_action_pair_exactly(self):
        # Consistency guard: the pure-action writer must be the vertex
        # special case of the mixed writer, or the two would silently
        # disagree on the slot layout.
        base = torch.full((STATE_DIM_ATTACK_DEFEND,), 7.0)
        pure = write_joint_action_pair(base, 2, 1, 4, 0)
        mixed = write_mixed_strategy_pair(
            base, _onehot(N_SKILLS, 2), _onehot(N_DEFEND_POSITIONS, 1),
            _onehot(N_SKILLS, 4), _onehot(N_DEFEND_POSITIONS, 0))
        torch.testing.assert_close(pure, mixed)

    def test_probability_vectors_land_in_the_right_slots(self):
        base = torch.zeros(STATE_DIM_ATTACK_DEFEND)
        wA1 = torch.tensor([0.1, 0.2, 0.3, 0.2, 0.2])
        wD1 = torch.tensor([0.5, 0.25, 0.25])
        wA2 = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
        wD2 = torch.tensor([0.2, 0.3, 0.5])
        out = write_mixed_strategy_pair(base, wA1, wD1, wA2, wD2)
        torch.testing.assert_close(out[74:79], wA1)
        torch.testing.assert_close(out[79:84], wA2)
        torch.testing.assert_close(out[84:87], wD1)
        torch.testing.assert_close(out[87:90], wD2)

    def test_no_mutation_and_batched(self):
        base = torch.full((4, STATE_DIM_ATTACK_DEFEND), 3.0)
        before = base.clone()
        out = write_mixed_strategy_pair(base, torch.full((5,), 0.2), torch.full((3,), 1 / 3),
                                        torch.full((5,), 0.2), torch.full((3,), 1 / 3))
        torch.testing.assert_close(base, before)
        assert out.shape == base.shape
        assert torch.allclose(out[:, 74:79].sum(dim=1), torch.ones(4))

    def test_gradient_flows_to_theta(self):
        # PGA needs d Phi / d theta through the written state.
        base = torch.zeros(STATE_DIM_ATTACK_DEFEND)
        wA1 = torch.full((5,), 0.2, requires_grad=True)
        out = write_mixed_strategy_pair(base, wA1, torch.full((3,), 1 / 3),
                                        torch.full((5,), 0.2), torch.full((3,), 1 / 3))
        out[74:79].sum().backward()
        assert wA1.grad is not None and torch.allclose(wA1.grad, torch.ones(5))


class TestProjectToSimplex:
    def test_known_cases(self):
        torch.testing.assert_close(project_to_simplex(torch.tensor([1.0, 1.0, 0.0])),
                                   torch.tensor([0.5, 0.5, 0.0]))
        torch.testing.assert_close(project_to_simplex(torch.tensor([3.0, 0.0, 0.0])),
                                   torch.tensor([1.0, 0.0, 0.0]))
        torch.testing.assert_close(project_to_simplex(torch.tensor([0.5, 0.5, 0.5])),
                                   torch.full((3,), 1 / 3))

    def test_output_is_on_the_simplex(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            v = torch.tensor(rng.normal(size=5) * 3, dtype=torch.float32)
            w = project_to_simplex(v)
            assert (w >= -1e-7).all()
            assert w.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_idempotent_on_simplex_points(self):
        p = torch.tensor([0.1, 0.6, 0.3])
        torch.testing.assert_close(project_to_simplex(p), p)


class TestSampleTheta:
    def test_valid_simplex_vectors_of_the_right_sizes(self):
        rng = np.random.default_rng(0)
        for _ in range(30):
            wA, wD = sample_theta(rng, p_vertex=0.3)
            assert wA.shape == (N_SKILLS,) and wD.shape == (N_DEFEND_POSITIONS,)
            assert wA.sum() == pytest.approx(1.0) and wD.sum() == pytest.approx(1.0)
            assert (wA >= 0).all() and (wD >= 0).all()

    def test_p_vertex_one_always_gives_vertices(self):
        # Paper: data must "cover both the fifteen fixed target-preparation
        # pairings and randomized play" -- vertices are the pure pairings.
        rng = np.random.default_rng(1)
        for _ in range(20):
            wA, wD = sample_theta(rng, p_vertex=1.0)
            assert wA.max() == 1.0 and wD.max() == 1.0

    def test_p_vertex_zero_gives_interior_points(self):
        rng = np.random.default_rng(2)
        for _ in range(20):
            wA, wD = sample_theta(rng, p_vertex=0.0)
            assert wA.max() < 1.0 and wD.max() < 1.0

    def test_deterministic_given_seed(self):
        a = sample_theta(np.random.default_rng(7), p_vertex=0.5)
        b = sample_theta(np.random.default_rng(7), p_vertex=0.5)
        np.testing.assert_allclose(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1])

    def test_theta_vertex_is_one_hot(self):
        wA, wD = theta_vertex(3, 2)
        assert wA[3] == 1.0 and wA.sum() == 1.0
        assert wD[2] == 1.0 and wD.sum() == 1.0


class TestSampleActionFromTheta:
    def test_vertex_theta_always_returns_its_index(self):
        rng = np.random.default_rng(0)
        wA, wD = theta_vertex(4, 1)
        for _ in range(20):
            assert sample_action_from_theta(rng, wA, wD) == (4, 1)

    def test_frequencies_track_the_distribution(self):
        rng = np.random.default_rng(0)
        wA = np.array([0.7, 0.3, 0.0, 0.0, 0.0])
        wD = np.array([0.0, 0.0, 1.0])
        draws = [sample_action_from_theta(rng, wA, wD)[0] for _ in range(4000)]
        assert np.mean(np.array(draws) == 0) == pytest.approx(0.7, abs=0.03)


def _uniform_theta():
    return (torch.full((N_SKILLS,), 1 / N_SKILLS), torch.full((N_DEFEND_POSITIONS,), 1 / N_DEFEND_POSITIONS),
            torch.full((N_SKILLS,), 1 / N_SKILLS), torch.full((N_DEFEND_POSITIONS,), 1 / N_DEFEND_POSITIONS))


class TestPGAMaximize:
    def test_linear_objective_converges_to_the_best_vertex(self):
        # A function linear in one block: max over the simplex is a vertex.
        c = torch.tensor([0.1, -0.3, 0.9, 0.2, 0.0])

        def phi(wA1, _wD1, _wA2, _wD2):
            return (c * wA1).sum()

        res = pga_maximize(phi, init=_uniform_theta(), steps=200, lr=0.2, n_restarts=2,
                           rng=np.random.default_rng(0))
        assert res.theta[0].argmax().item() == 2
        assert res.value == pytest.approx(0.9, abs=1e-3)

    def test_finds_an_interior_maximum_that_discrete_argmax_cannot(self):
        # The case the continuous formulation exists for: a genuinely
        # non-multilinear Phi with an INTERIOR optimum. No vertex attains it.
        target = torch.tensor([0.2, 0.3, 0.1, 0.25, 0.15])

        def phi(wA1, _wD1, _wA2, _wD2):
            return -((wA1 - target) ** 2).sum()

        res = pga_maximize(phi, init=_uniform_theta(), steps=400, lr=0.2, n_restarts=2,
                           rng=np.random.default_rng(0))
        torch.testing.assert_close(res.theta[0], target, atol=1e-3, rtol=0)
        best_vertex_value = max(phi(torch.eye(5)[k], None, None, None).item() for k in range(5))
        assert res.value > best_vertex_value + 0.1

    def test_warm_start_at_the_optimum_stays_there(self):
        target = torch.tensor([0.5, 0.25, 0.25])

        def phi(_wA1, wD1, _wA2, _wD2):
            return -((wD1 - target) ** 2).sum()

        init = (_uniform_theta()[0], target.clone(), _uniform_theta()[2], _uniform_theta()[3])
        res = pga_maximize(phi, init=init, steps=5, lr=0.1, n_restarts=0,
                           rng=np.random.default_rng(0))
        torch.testing.assert_close(res.theta[1], target, atol=1e-6, rtol=0)

    def test_reports_every_starts_value_for_local_optima_sensitivity(self):
        def phi(wA1, wD1, wA2, wD2):
            return wA1[0] * wA2[0] + wD1[0] * wD2[0]

        res = pga_maximize(phi, init=_uniform_theta(), steps=50, lr=0.1, n_restarts=3,
                           rng=np.random.default_rng(0))
        assert len(res.start_values) == 4  # warm start + 3 restarts
        assert res.value == pytest.approx(max(res.start_values))

    def test_returned_theta_is_on_the_product_of_simplexes(self):
        def phi(wA1, wD1, wA2, wD2):
            return wA1[1] - wD1[2] + wA2[4] * wD2[0]

        res = pga_maximize(phi, init=None, steps=60, lr=0.3, n_restarts=3,
                           rng=np.random.default_rng(0))
        for w in res.theta:
            assert (w >= -1e-6).all()
            assert w.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_requires_at_least_one_start(self):
        with pytest.raises(ValueError):
            pga_maximize(lambda *_: torch.tensor(0.0), init=None, steps=1, lr=0.1,
                         n_restarts=0, rng=np.random.default_rng(0))
