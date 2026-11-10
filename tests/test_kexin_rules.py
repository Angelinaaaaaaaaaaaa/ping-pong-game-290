"""Decision rules matching Kexin's V3 table rows, on the Design B 15x15 joint game:
nash-q (zero-sum LP on Q), br-q (Q best response to the opponent's previous pair),
nash-p-ibr (coordinate ascent on Phi); plus the zero-sum solver they use."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch
import torch.nn as nn

from nash_skills import eval_matchup_attack_defend as emad
from nash_skills.v2.joint_action import N_JOINT_ACTIONS, joint_action_index
from nash_skills.v2.matrix_game import solve_zero_sum


class TestSolveZeroSum:
    def test_matching_pennies(self):
        p, q, v = solve_zero_sum(np.array([[1.0, -1.0], [-1.0, 1.0]]))
        np.testing.assert_allclose(p, [0.5, 0.5], atol=1e-6)
        np.testing.assert_allclose(q, [0.5, 0.5], atol=1e-6)
        assert v == pytest.approx(0.0, abs=1e-6)

    def test_saddle_point_is_pure(self):
        M = np.array([[3.0, 1.0, 4.0], [2.0, 0.0, 1.0]])
        p, q, v = solve_zero_sum(M)
        np.testing.assert_allclose(p, [1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(q, [0.0, 1.0, 0.0], atol=1e-6)
        assert v == pytest.approx(1.0, abs=1e-6)

    def test_rock_paper_scissors_is_uniform(self):
        M = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
        p, q, v = solve_zero_sum(M)
        np.testing.assert_allclose(p, [1 / 3] * 3, atol=1e-6)
        assert v == pytest.approx(0.0, abs=1e-6)


def _decode(x):
    a1, a2 = x[:, 74:79].argmax(1), x[:, 79:84].argmax(1)
    d1, d2 = x[:, 84:87].argmax(1), x[:, 87:90].argmax(1)
    i = torch.tensor([joint_action_index(int(a), int(d)) for a, d in zip(a1, d1)])
    j = torch.tensor([joint_action_index(int(a), int(d)) for a, d in zip(a2, d2)])
    return i, j


class TableModel(nn.Module):
    """Returns table[i, j] for the joint pair encoded in the state's action slots."""

    def __init__(self, table):
        super().__init__()
        self.table = torch.tensor(table, dtype=torch.float32)

    def forward(self, x):
        i, j = _decode(x)
        return self.table[i, j].unsqueeze(-1)


def _info(decision_index=1, current=(0, 0)):
    return {"decision_index": decision_index, "current_joints": current}


def _base():
    return np.zeros(90, dtype=np.float32)


def test_br_q_best_responds_to_the_opponents_previous_pair():
    q1 = np.zeros((N_JOINT_ACTIONS, N_JOINT_ACTIONS))
    q1[7, 4] = 1.0          # vs opponent pair 4, player 1's best pair is 7
    q2 = np.zeros_like(q1)
    q2[2, 11] = 1.0         # vs player-1 pair 2, player 2's best pair is 11
    pick = emad.make_picker_joint("br-q", None, state_encoder_fn=lambda o, i, p: _base(),
                                  model1=TableModel(q1), model2=TableModel(q2))
    assert pick(1, _base(), 4, _info()) == 7
    assert pick(2, _base(), 2, _info()) == 11


def test_nash_q_plays_the_saddle_point_of_the_zero_sum_q_game():
    q1 = -np.ones((N_JOINT_ACTIONS, N_JOINT_ACTIONS))
    q1[5, :] = 0.5           # player 1's pair 5 guarantees 0.5
    q1[5, 9] = 0.2           # ... except against player 2's pair 9 -> saddle (5, 9)
    q2 = -q1
    pick = emad.make_picker_joint("nash-q", None, state_encoder_fn=lambda o, i, p: _base(),
                                  model1=TableModel(q1), model2=TableModel(q2))
    assert {pick(1, _base(), 0, _info()) for _ in range(10)} == {5}
    assert {pick(2, _base(), 0, _info()) for _ in range(10)} == {9}


def test_br_q_uses_nash_on_the_first_decision_of_a_rally():
    q1 = -np.ones((N_JOINT_ACTIONS, N_JOINT_ACTIONS))
    q1[5, :] = 0.5
    q1[5, 9] = 0.2
    pick = emad.make_picker_joint("br-q", None, state_encoder_fn=lambda o, i, p: _base(),
                                  model1=TableModel(q1), model2=TableModel(-q1))
    assert pick(1, _base(), 0, _info(decision_index=0)) == 5


def test_potential_ibr_climbs_phi_by_alternating_moves():
    phi = np.zeros((N_JOINT_ACTIONS, N_JOINT_ACTIONS))
    start = joint_action_index(2, 2)  # center_safe / center
    phi[start, start] = 0.1
    phi[3, start] = 0.5               # player 1 improves to 3
    phi[3, 12] = 0.9                  # then player 2 improves to 12 (a local max)
    pick = emad.make_picker_joint("nash-p-ibr", TableModel(phi), state_encoder_fn=lambda o, i, p: _base())
    assert pick(1, _base(), 0, _info(decision_index=0)) == 3
    assert pick(2, _base(), 0, _info(decision_index=0)) == 12


def test_potential_ibr_starts_from_the_current_pairs_after_the_first_decision():
    phi = np.zeros((N_JOINT_ACTIONS, N_JOINT_ACTIONS))
    phi[6, 8] = 1.0
    phi[6, 1] = 0.5
    pick = emad.make_picker_joint("nash-p-ibr", TableModel(phi), state_encoder_fn=lambda o, i, p: _base())
    # current pairs (6, 1): player 2 improves to 8 while player 1 stays at 6
    assert pick(1, _base(), 1, _info(current=(6, 1))) == 6
    assert pick(2, _base(), 6, _info(current=(6, 1))) == 8


def test_q_rules_need_q_models():
    with pytest.raises(ValueError):
        emad.make_picker_joint("nash-q", None)
    with pytest.raises(ValueError):
        emad.make_picker_joint("br-q", None)


def test_run_matchup_passes_the_decision_index(monkeypatch):
    from tests.test_eval_decision_timing import TimedEnv
    from tests.test_eval_matchup_attack_defend_run import FakeModel, TWO_DECISION_SCRIPT
    env = TimedEnv(TWO_DECISION_SCRIPT)
    monkeypatch.setattr(emad, "SkillEnv", lambda **_kw: env)
    seen = []

    def fake_make_picker(_strategy, *_a, **_kw):
        def pick(p, _obs, _other, info=None):
            seen.append(info["decision_index"])
            return 0
        return pick

    monkeypatch.setattr(emad, "make_picker_joint", fake_make_picker)
    emad.run_matchup_joint(1, 2, FakeModel(), None, n_episodes=1, max_steps_per_episode=10,
                           defend_mechanism="preparation", decision_timing="contact")
    assert seen == [0, 0, 1, 1]
