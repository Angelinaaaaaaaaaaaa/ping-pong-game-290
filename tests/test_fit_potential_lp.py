"""scripts/fit_potential_lp.py: Kexin's V3 way of fitting Phi (per-state LP labels, then regression) on the
Design B joint game."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

import fit_potential_lp as fl
from nash_skills.table_rules import joint_value_table
from nash_skills.v2.joint_action import write_joint_pair_by_index


class _Linear(torch.nn.Module):
    """A value model whose output depends on the state and on the choice slots."""

    def __init__(self, dim):
        super().__init__()
        self.w = torch.linspace(-1.0, 1.0, dim)

    def forward(self, x):
        return (x * self.w).sum(dim=1, keepdim=True)


def test_with_profiles_matches_the_one_pair_writer_row_by_row():
    x = torch.rand(6, 97)
    i, j = torch.tensor([0, 3, 14, 7, 7, 2]), torch.tensor([14, 0, 5, 7, 1, 9])
    out = fl.with_profiles(x, i, j)
    for r in range(6):
        torch.testing.assert_close(out[r], write_joint_pair_by_index(x[r:r + 1], int(i[r]), int(j[r]))[0])
    assert not torch.equal(out, x)  # a new tensor; the input is left alone


def test_value_tables_match_the_decision_rules_tables():
    x = torch.rand(5, 97)
    q1, q2 = fl.value_tables(_Linear(97), _Linear(97), x, chunk=2)
    assert q1.shape == (5, 15, 15)
    for s in range(5):
        np.testing.assert_allclose(q1[s], joint_value_table(x[s].numpy(), _Linear(97)), rtol=1e-5, atol=1e-5)


def test_lp_labels_solve_each_state():
    rng = np.random.default_rng(0)
    q1 = rng.normal(size=(3, 15, 15))
    phi, alpha = fl.lp_labels(q1, -q1, workers=1)
    assert phi.shape == (3, 15, 15) and alpha.shape == (3,) and np.all(alpha > 0)
    assert np.all(phi[:, 0, 0] == 0.0)
