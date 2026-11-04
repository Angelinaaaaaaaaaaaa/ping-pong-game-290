"""Tests for scripts/phi_sign_agreement.py (does Phi move the same way as the deviator's Q?)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch
import torch.nn as nn

from phi_sign_agreement import deviation_diffs, sign_agreement


def test_sign_agreement_ignores_negligible_value_changes():
    dphi = np.array([0.5, -0.2, 0.3, 0.1])
    dq = np.array([0.4, 0.3, 1e-6, -0.2])
    out = sign_agreement(dphi, dq, eps=1e-3)
    assert out["n"] == 3
    assert out["agree"] == pytest.approx(1 / 3)


class _Lin(nn.Module):
    def __init__(self, sl, w):
        super().__init__()
        self.sl, self.w = sl, torch.tensor(w, dtype=torch.float32)

    def forward(self, x):
        return (x[:, self.sl] @ self.w).unsqueeze(-1)


def test_exact_potential_agrees_everywhere_for_both_halves():
    # V1 depends on P1's attack + defend blocks, V2 on P2's; Phi = V1 + V2 is exact.
    v1 = _Lin(slice(74, 90), [0.3, -0.1, 0.5, 0.0, 0.2] + [0.0] * 5 + [0.4, -0.3, 0.1] + [0.0] * 3)
    v2 = _Lin(slice(74, 90), [0.0] * 5 + [-0.4, 0.2, 0.1, 0.3, 0.0] + [0.0] * 3 + [0.2, 0.6, -0.5])

    class Phi(nn.Module):
        def forward(self, x):
            return v1(x) + v2(x)

    x = torch.zeros(32, 90)
    rng = np.random.default_rng(0)
    for player in (1, 2):
        for half in ("attack", "defend"):
            dphi, dq = deviation_diffs(Phi(), v1, v2, x, player, half, rng)
            assert sign_agreement(dphi, dq, 1e-6)["agree"] == pytest.approx(1.0)
