"""scripts/diagnose_potential_equilibrium.py: equilibrium structure of the learned game vs Eq. 12."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

import diagnose_potential_equilibrium as dg

PENNIES = np.array([[1.0, -1.0], [-1.0, 1.0]])
SADDLE = np.array([[3.0, 1.0], [4.0, 2.0]])  # pure saddle point at (1, 1), value 2
PHI_AT_SADDLE = np.array([[0.0, 0.0], [0.0, 1.0]])


def test_pure_security_levels():
    assert dg.pure_security(PENNIES) == (-1.0, 1.0)
    assert dg.pure_security(SADDLE) == (2.0, 2.0)


def test_profile_gains_are_unilateral_improvements():
    assert dg.profile_gains(PENNIES, 0, 0) == (0.0, 2.0)  # player 2 gains 2 by switching
    assert dg.profile_gains(SADDLE, 1, 1) == (0.0, 0.0)
    assert dg.profile_gains(SADDLE, 0, 0) == (1.0, 2.0)


def test_analyze_state_on_matching_pennies():
    # Q1 = M, Q2 = -M; a potential that prefers (0, 0)
    phi = np.array([[1.0, 0.0], [0.0, 0.0]])
    r = dg.analyze_state(PENNIES, -PENNIES, phi)
    assert not r["pure_saddle"] and r["support1"] == 2 and r["support2"] == 2
    assert r["value"] == pytest.approx(0.0, abs=1e-6)
    assert r["randomization_gain1"] == pytest.approx(1.0) and r["randomization_gain2"] == pytest.approx(1.0)
    assert (r["phi_i"], r["phi_j"]) == (0, 0) and r["phi_gain2"] == pytest.approx(2.0)
    assert r["phi_nashconv"] == pytest.approx(2.0)


def test_analyze_state_with_a_pure_saddle():
    r = dg.analyze_state(SADDLE, -SADDLE, PHI_AT_SADDLE)
    assert r["pure_saddle"] and r["support1"] == 1 and r["support2"] == 1
    assert r["phi_nashconv"] == pytest.approx(0.0, abs=1e-9)


def test_summarize_reports_shares_and_choice_frequencies():
    rows = [dg.analyze_state(PENNIES, -PENNIES, np.array([[1.0, 0.0], [0.0, 0.0]])),
            dg.analyze_state(SADDLE, -SADDLE, PHI_AT_SADDLE)]
    s = dg.summarize(rows)
    assert s["n_states"] == 2 and s["pure_saddle_share"] == 0.5
    assert s["phi_nashconv_mean"] == pytest.approx(1.0)
