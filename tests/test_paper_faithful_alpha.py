"""Paper-faithful alpha-potential (Sec. 4.1): continuation length H in the value targets (Eq. 9),
training on fixed-pair (pure) rallies only (Eq. 7), and the exact 225-profile argmax rule (Eq. 12)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from nash_skills.v2 import train_q_model_5skill_v3_attack_defend as tq
from nash_skills.v2.labeling import compute_returns


class TestHorizonReturns:
    def test_long_horizon_equals_the_full_discounted_return(self):
        states = [np.zeros(3)] * 5
        g1, _ = compute_returns(states, gamma=0.9, winner=1)
        np.testing.assert_allclose(tq.returns_with_horizon(5, winner=1, gamma=0.9, horizon=10), g1)

    def test_rewards_beyond_the_cutoff_are_dropped(self):
        # decision t sees the point after (n - 1 - t) more exchanges; kept only if that is < H
        out = tq.returns_with_horizon(5, winner=2, gamma=0.5, horizon=2)
        np.testing.assert_allclose(out, [0.0, 0.0, 0.0, -0.5, -1.0])

    def test_unresolved_rally_is_zero(self):
        np.testing.assert_allclose(tq.returns_with_horizon(3, winner=0, gamma=0.9, horizon=5), [0, 0, 0])


def _write(path, n_rallies=12, mixed_every=2):
    from tests.test_train_q_model_5skill_v3_attack_defend import TestOutDir
    TestOutDir()._write_rallies(path, n_rallies=n_rallies)
    import pickle
    rallies = pickle.load(open(path, "rb"))
    for i, r in enumerate(rallies):
        if i % mixed_every == 0:
            r["theta_mode"] = "mixed"
    pickle.dump(rallies, open(path, "wb"))


def test_pure_data_only_drops_mixed_rallies_and_records_settings(tmp_path):
    path = tmp_path / "r.pkl"
    _write(path)
    rep = tq.train(str(path), n_epochs=2, lr=1e-3, out_dir=str(tmp_path / "run"), pure_data_only=True, horizon=4)
    saved = json.load(open(tmp_path / "run" / "heldout_attack_defend.json"))
    assert saved["pure_data_only"] is True and saved["horizon"] == 4
    assert rep["n_train_rallies"] + rep["n_holdout_rallies"] + saved["n_val_rallies"] == 6


def test_pure_data_only_needs_pure_theta_mode(tmp_path):
    path = tmp_path / "r.pkl"
    _write(path)
    with pytest.raises(ValueError, match="pure"):
        tq.train(str(path), n_epochs=1, lr=1e-3, out_dir=str(tmp_path / "run"), pure_data_only=True,
                 theta_mode="mixed")


def test_exact_argmax_rule_plays_its_half_of_the_best_joint_pair():
    from nash_skills import eval_matchup_attack_defend as emad
    from tests.test_kexin_rules import TableModel, _base, _info
    from nash_skills.v2.joint_action import N_JOINT_ACTIONS
    phi = np.zeros((N_JOINT_ACTIONS, N_JOINT_ACTIONS))
    phi[4, 11] = 2.0     # global best profile
    phi[4, 0] = 1.9      # player 1's row max is shared; player 2 must pick 11, not 0
    phi[9, 0] = 1.95
    pick = emad.make_picker_joint("nash-p-argmax", TableModel(phi), state_encoder_fn=lambda o, i, p: _base())
    assert pick(1, _base(), 0, _info()) == 4
    assert pick(2, _base(), 0, _info()) == 11
