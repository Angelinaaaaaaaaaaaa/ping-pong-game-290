"""Tests for nash_skills.v2.train_q_model_5skill_v3_attack_defend.

TDD: written before implementation.

apply_joint_pair is the one piece of actual new logic in this trainer
(everything else -- build_dataset, the Q-training loop, the potential
same-state constraint-training loop -- is unchanged from
train_q_model_5skill_v3_one_hot.py, generic over state dimension and
action-index range). It decomposes a pair of flat per-player joint
indices (0..14) and delegates the one-hot write to
nash_skills.v2.joint_action.write_joint_action_pair -- tested directly
here (not just indirectly through a full training run) since a wrong
decomposition would silently corrupt every counterfactual-constraint
query the potential model trains against, the same failure mode already
caught twice this session.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from nash_skills.v2.train_q_model_5skill_v3_attack_defend import apply_joint_pair
from nash_skills.v2.joint_action import N_JOINT_ACTIONS, joint_action_index
from nash_skills.v2.state_encoder_attack_defend import STATE_DIM_ATTACK_DEFEND, STATE_ENCODER_NAME


def _base_batch(n=3):
    return torch.full((n, STATE_DIM_ATTACK_DEFEND), 7.0, dtype=torch.float32)


class TestAppliesCorrectDecomposedBlocks:
    def test_ego_joint_index_decomposes_to_correct_attack_and_defend_blocks(self):
        x = _base_batch()
        # joint index for (attack=2, defend=1) -- see joint_action_index's
        # row-major convention (attack_idx * N_DEFEND_POSITIONS + defend_idx)
        ego_joint = joint_action_index(2, 1)
        out = apply_joint_pair(x, ego_joint, opp_joint=0)
        assert out[0, 74:79].argmax().item() == 2   # ego attack block
        assert out[0, 84:87].argmax().item() == 1   # ego defend block

    def test_opp_joint_index_decomposes_to_correct_attack_and_defend_blocks(self):
        x = _base_batch()
        opp_joint = joint_action_index(3, 2)
        out = apply_joint_pair(x, ego_joint=0, opp_joint=opp_joint)
        assert out[0, 79:84].argmax().item() == 3   # opp attack block
        assert out[0, 87:90].argmax().item() == 2   # opp defend block

    def test_every_joint_index_pair_round_trips(self):
        x = _base_batch(n=1)
        for ego_joint in range(N_JOINT_ACTIONS):
            for opp_joint in range(N_JOINT_ACTIONS):
                out = apply_joint_pair(x, ego_joint, opp_joint)
                assert out[0, 74:79].sum().item() == 1.0
                assert out[0, 79:84].sum().item() == 1.0
                assert out[0, 84:87].sum().item() == 1.0
                assert out[0, 87:90].sum().item() == 1.0

    def test_output_shape_and_no_mutation(self):
        x = _base_batch(n=4)
        x_before = x.clone()
        out = apply_joint_pair(x, 5, 10)
        assert out.shape == x.shape == (4, STATE_DIM_ATTACK_DEFEND)
        torch.testing.assert_close(x, x_before)


# --------------------------------------------------------------------------- #
# Paper Sec 4.1 alignment: rally-level split, held-out residuals (empirical
# alpha), max-discrepancy objective, nested data-volume subsets.
# --------------------------------------------------------------------------- #
import numpy as np
import pytest
import torch.nn as nn

from nash_skills.v2.train_q_model_5skill_v3_attack_defend import (
    split_rallies,
    limit_rallies,
    phi_discrepancy_loss,
    residual_stats,
    build_constraints,
    unilateral_residuals,
)


def _fake_rallies(n):
    return [{"id": k, "states": [np.zeros(STATE_DIM_ATTACK_DEFEND, dtype=np.float32)]} for k in range(n)]


class TestSplitRallies:
    """Paper Sec 4.1: 'Split data by independently generated rallies, so
    that correlated states from one rally do not appear in both training
    and evaluation sets.' The trainer previously had no held-out set at
    all -- every reported loss was a training loss."""

    def test_no_rally_appears_in_both_sides(self):
        train, hold = split_rallies(_fake_rallies(50), holdout_frac=0.2, seed=0)
        train_ids = {r["id"] for r in train}
        hold_ids = {r["id"] for r in hold}
        assert train_ids.isdisjoint(hold_ids)
        assert len(train_ids) + len(hold_ids) == 50

    def test_holdout_size_matches_fraction(self):
        _train, hold = split_rallies(_fake_rallies(50), holdout_frac=0.2, seed=0)
        assert len(hold) == 10

    def test_deterministic_for_same_seed_and_varies_across_seeds(self):
        a = [r["id"] for r in split_rallies(_fake_rallies(40), 0.25, seed=1)[1]]
        b = [r["id"] for r in split_rallies(_fake_rallies(40), 0.25, seed=1)[1]]
        c = [r["id"] for r in split_rallies(_fake_rallies(40), 0.25, seed=2)[1]]
        assert a == b
        assert a != c

    def test_both_sides_nonempty_for_tiny_datasets(self):
        train, hold = split_rallies(_fake_rallies(2), holdout_frac=0.01, seed=0)
        assert len(train) == 1 and len(hold) == 1

    def test_zero_fraction_means_no_holdout(self):
        train, hold = split_rallies(_fake_rallies(10), holdout_frac=0.0, seed=0)
        assert len(hold) == 0 and len(train) == 10

    def test_invalid_fraction_raises(self):
        with pytest.raises(ValueError):
            split_rallies(_fake_rallies(10), holdout_frac=1.0, seed=0)


class TestLimitRalliesIsNested:
    """Paper Sec 5.2: 'Train ... with NESTED subsets containing different
    numbers M of independently generated rallies.'"""

    def test_smaller_limit_is_prefix_of_larger(self):
        train, _ = split_rallies(_fake_rallies(30), 0.2, seed=0)
        small = [r["id"] for r in limit_rallies(train, 5)]
        large = [r["id"] for r in limit_rallies(train, 12)]
        assert large[:5] == small

    def test_none_means_all(self):
        train, _ = split_rallies(_fake_rallies(30), 0.2, seed=0)
        assert len(limit_rallies(train, None)) == len(train)

    def test_nonpositive_limit_raises(self):
        with pytest.raises(ValueError):
            limit_rallies(_fake_rallies(5), 0)


class TestPhiDiscrepancyLoss:
    """Paper Eq. (potentialtraining): minimize the LARGEST sampled
    discrepancy. 'lse' is a smooth version: (1/beta)*[logsumexp(beta|e|)
    - log N], which equals c when all |e|=c, tends to mean|e| as beta->0
    and to max|e| as beta->inf."""

    def test_mse_matches_mean_squared_error(self):
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 1.0, 1.0])
        assert phi_discrepancy_loss(pred, target, kind="mse").item() == pytest.approx((0 + 1 + 4) / 3)

    def test_lse_equals_common_value_when_all_residuals_equal(self):
        pred = torch.tensor([0.5, -0.5, 0.5])
        target = torch.zeros(3)
        assert phi_discrepancy_loss(pred, target, kind="lse", beta=7.0).item() == pytest.approx(0.5, abs=1e-5)

    def test_lse_approaches_max_for_large_beta(self):
        pred = torch.tensor([0.1, 0.2, 0.9])
        target = torch.zeros(3)
        val = phi_discrepancy_loss(pred, target, kind="lse", beta=500.0).item()
        assert val == pytest.approx(0.9, abs=0.01)

    def test_lse_is_between_mean_and_max(self):
        pred = torch.tensor([0.1, 0.2, 0.9])
        target = torch.zeros(3)
        val = phi_discrepancy_loss(pred, target, kind="lse", beta=5.0).item()
        assert (0.1 + 0.2 + 0.9) / 3 <= val <= 0.9

    def test_lse_has_finite_gradient_at_zero_residual(self):
        pred = torch.zeros(4, requires_grad=True)
        loss = phi_discrepancy_loss(pred, torch.zeros(4), kind="lse", beta=10.0)
        loss.backward()
        assert torch.isfinite(pred.grad).all()

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            phi_discrepancy_loss(torch.zeros(2), torch.zeros(2), kind="huber")


class TestResidualStats:
    def test_reports_max_mean_p95_of_absolute_residuals(self):
        stats = residual_stats(np.array([-0.4, 0.1, 0.2, -0.1]))
        assert stats["max_abs"] == pytest.approx(0.4)
        assert stats["mean_abs"] == pytest.approx(0.2)
        assert stats["n"] == 4
        assert 0.2 <= stats["p95_abs"] <= 0.4


class TestBuildConstraints:
    def test_count_matches_unilateral_deviation_structure(self):
        # 15x15 base pairs, 14 alternatives per player, 2 players
        assert len(build_constraints()) == 15 * 15 * 14 * 2

    def test_each_constraint_changes_exactly_one_players_joint_action(self):
        for kind, i, j, i2, j2 in build_constraints()[:500]:
            if kind == "p1":
                assert i2 != i and j2 == j
            else:
                assert i2 == i and j2 != j


class _ExactPotential(nn.Module):
    """Phi that EXACTLY tracks the value models' unilateral differences:
    Phi(x) = V1(x) + V2(x) is NOT generally exact, but if V1 depends only
    on ego's action and V2 only on opp's action, then Phi = V1 + V2 is an
    exact potential -- every residual must be ~0."""

    def __init__(self, v1, v2):
        super().__init__()
        self.v1, self.v2 = v1, v2

    def forward(self, x):
        return self.v1(x) + self.v2(x)


class _EgoActionValue(nn.Module):
    def forward(self, x):
        return x[:, 74:79].argmax(dim=1, keepdim=True).float() * 0.1


class _OppActionValue(nn.Module):
    def forward(self, x):
        return x[:, 79:84].argmax(dim=1, keepdim=True).float() * -0.2


class TestUnilateralResiduals:
    def test_exact_potential_gives_zero_residuals(self):
        v1, v2 = _EgoActionValue(), _OppActionValue()
        phi = _ExactPotential(v1, v2)
        x = torch.zeros(16, STATE_DIM_ATTACK_DEFEND)
        res = unilateral_residuals(phi, v1, v2, x, build_constraints(),
                                   rng=np.random.default_rng(0), n_constraints=50, batch_size=8)
        assert np.abs(res).max() == pytest.approx(0.0, abs=1e-6)

    def test_constant_potential_gives_nonzero_residuals(self):
        v1, v2 = _EgoActionValue(), _OppActionValue()

        class Const(nn.Module):
            def forward(self, x):
                return torch.zeros(x.shape[0], 1)

        x = torch.zeros(16, STATE_DIM_ATTACK_DEFEND)
        res = unilateral_residuals(Const(), v1, v2, x, build_constraints(),
                                   rng=np.random.default_rng(0), n_constraints=50, batch_size=8)
        assert np.abs(res).max() > 0.05


# --------------------------------------------------------------------------- #
# theta-mode "mixed": unilateral deviations in the compact family
# theta^i = (w^A, w^D) in Delta_5 x Delta_3 (paper Sec 4.1), instead of in
# pure joint actions. Requires data from collect_data_attack_defend.py
# --sampling mixed -- otherwise the value models have never seen interior
# theta and every theta-deviation target would be pure extrapolation.
# --------------------------------------------------------------------------- #
from nash_skills.v2.train_q_model_5skill_v3_attack_defend import (
    sample_theta_deviation,
    theta_unilateral_residuals,
    THETA_MODES,
    train as train_fn,
)


class _LinearEgoAttack(nn.Module):
    def forward(self, x):
        return x[:, 74:79] @ torch.tensor([[0.3], [-0.1], [0.5], [0.0], [0.2]])


class _LinearOppAttack(nn.Module):
    def forward(self, x):
        return x[:, 79:84] @ torch.tensor([[-0.4], [0.2], [0.1], [0.3], [0.0]])


class TestSampleThetaDeviation:
    def test_valid_player_and_thetas(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            player, th1, th2, th_alt = sample_theta_deviation(rng, p_vertex=0.3)
            assert player in (1, 2)
            for wA, wD in (th1, th2, th_alt):
                assert wA.sum() == pytest.approx(1.0) and wD.sum() == pytest.approx(1.0)

    def test_both_players_get_deviations(self):
        rng = np.random.default_rng(1)
        players = {sample_theta_deviation(rng, p_vertex=0.3)[0] for _ in range(40)}
        assert players == {1, 2}


class TestThetaUnilateralResiduals:
    def test_exact_potential_gives_zero_residuals_at_interior_theta(self):
        v1, v2 = _LinearEgoAttack(), _LinearOppAttack()
        phi = _ExactPotential(v1, v2)
        x = torch.zeros(16, STATE_DIM_ATTACK_DEFEND)
        res = theta_unilateral_residuals(phi, v1, v2, x, rng=np.random.default_rng(0),
                                         n_samples=40, batch_size=8, p_vertex=0.0)
        assert np.abs(res).max() == pytest.approx(0.0, abs=1e-6)

    def test_constant_potential_gives_nonzero_residuals(self):
        v1, v2 = _LinearEgoAttack(), _LinearOppAttack()

        class Const(nn.Module):
            def forward(self, x):
                return torch.zeros(x.shape[0], 1)

        x = torch.zeros(16, STATE_DIM_ATTACK_DEFEND)
        res = theta_unilateral_residuals(Const(), v1, v2, x, rng=np.random.default_rng(0),
                                         n_samples=40, batch_size=8, p_vertex=0.0)
        assert np.abs(res).max() > 0.01


class TestThetaModeValidation:
    def test_modes(self):
        assert THETA_MODES == ("pure", "mixed")

    def test_mixed_mode_rejects_pure_only_data(self, tmp_path):
        import pickle
        rallies = [{"states": [np.zeros(STATE_DIM_ATTACK_DEFEND, dtype=np.float32)] * 3,
                    "ego_attack_choices": ["left"] * 3, "ego_defend_choices": ["left"] * 3,
                    "opp_attack_choices": ["left"] * 3, "opp_defend_choices": ["left"] * 3,
                    "winner": 1, "encoder": STATE_ENCODER_NAME} for _ in range(4)]
        path = tmp_path / "pure.pkl"
        with open(path, "wb") as f:
            pickle.dump(rallies, f)
        with pytest.raises(ValueError, match="mixed"):
            train_fn(str(path), n_epochs=1, lr=1e-3, theta_mode="mixed",
                     heldout_out=str(tmp_path / "h.json"))


class TestOutDir:
    """--out-dir keeps parallel runs from overwriting each other's models and logs."""

    def _write_rallies(self, path, n_rallies=6, n_decisions=3, encoder=None):
        from nash_skills.v2.state_encoder_attack_defend import STATE_ENCODER_NAME
        encoder = encoder or STATE_ENCODER_NAME
        import pickle
        rng = np.random.default_rng(0)
        rallies = []
        for i in range(n_rallies):
            states = [rng.random(STATE_DIM_ATTACK_DEFEND).astype(np.float32) for _ in range(n_decisions)]
            rallies.append({
                "states": states,
                "ego_attack_choices": ["left"] * n_decisions, "ego_defend_choices": ["center"] * n_decisions,
                "opp_attack_choices": ["right"] * n_decisions, "opp_defend_choices": ["left"] * n_decisions,
                "winner": 1 + i % 2, "encoder": encoder,
            })
        with open(path, "wb") as f:
            pickle.dump(rallies, f)

    def test_all_outputs_land_in_out_dir(self, tmp_path):
        from nash_skills.v2 import train_q_model_5skill_v3_attack_defend as tq
        rally_path = tmp_path / "rallies.pkl"
        self._write_rallies(rally_path)
        out_dir = tmp_path / "run_a"
        tq.train(str(rally_path), n_epochs=2, lr=1e-3, out_dir=str(out_dir))
        expected = {
            "model1_attack_defend.pth", "model2_attack_defend.pth", "model_p_attack_defend.pth",
            "train_q_attack_defend.csv", "train_p_attack_defend.csv", "heldout_attack_defend.json",
        }
        assert expected <= set(os.listdir(out_dir))


class TestEncoderVersionGuard:
    def test_data_from_the_old_encoder_is_rejected(self, tmp_path):
        from nash_skills.v2 import train_q_model_5skill_v3_attack_defend as tq
        path = tmp_path / "old.pkl"
        TestOutDir()._write_rallies(path, encoder="attack_defend")
        with pytest.raises(ValueError, match="encoder"):
            tq.train(str(path), n_epochs=1, lr=1e-3, out_dir=str(tmp_path / "run"))

    def test_held_out_report_records_the_encoder(self, tmp_path):
        import json
        from nash_skills.v2 import train_q_model_5skill_v3_attack_defend as tq
        from nash_skills.v2.state_encoder_attack_defend import STATE_ENCODER_NAME
        path = tmp_path / "new.pkl"
        TestOutDir()._write_rallies(path)
        tq.train(str(path), n_epochs=2, lr=1e-3, out_dir=str(tmp_path / "run"))
        report = json.load(open(tmp_path / "run" / "heldout_attack_defend.json"))
        assert report["encoder"] == STATE_ENCODER_NAME
