"""The alpha-potential trainer takes its state size from the rallies' encoder tag (v2: 90, v3: 97)."""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from nash_skills.v2 import train_q_model_5skill_v3_attack_defend as tq
from nash_skills.v2.state_encoder_attack_defend import ENCODER_V2, ENCODER_V3


def _rallies(encoder, dim, n=12, k=3):
    rng = np.random.default_rng(0)
    return [{"states": [rng.random(dim).astype(np.float32) for _ in range(k)],
             "ego_attack_choices": ["left"] * k, "ego_defend_choices": ["center"] * k,
             "opp_attack_choices": ["right"] * k, "opp_defend_choices": ["left"] * k,
             "winner": 1 + i % 2, "encoder": encoder} for i in range(n)]


def test_rally_encoder_requires_one_known_encoder():
    assert tq.rally_encoder(_rallies(ENCODER_V3, 97, n=2)) == ENCODER_V3
    assert tq.rally_encoder(_rallies(ENCODER_V2, 90, n=2)) == ENCODER_V2
    with pytest.raises(ValueError, match="one state encoder"):
        tq.rally_encoder(_rallies(ENCODER_V2, 90, n=1) + _rallies(ENCODER_V3, 97, n=1))
    with pytest.raises(ValueError, match="Recollect"):
        tq.rally_encoder(_rallies("one_hot_v1", 76, n=1))


def test_v3_rallies_train_97_input_models(tmp_path):
    path = tmp_path / "r.pkl"
    pickle.dump(_rallies(ENCODER_V3, 97), open(path, "wb"))
    tq.train(str(path), n_epochs=2, lr=1e-3, out_dir=str(tmp_path / "run"))
    saved = json.load(open(tmp_path / "run" / "heldout_attack_defend.json"))
    assert saved["encoder"] == ENCODER_V3
    for name in ("model1_attack_defend.pth", "model_p_attack_defend.pth"):
        weights = torch.load(tmp_path / "run" / name, map_location="cpu")
        assert weights["batch_norm.running_mean"].shape == (97,)


def test_state_length_must_match_the_tag(tmp_path):
    path = tmp_path / "r.pkl"
    pickle.dump(_rallies(ENCODER_V3, 90), open(path, "wb"))
    with pytest.raises(ValueError, match="97"):
        tq.train(str(path), n_epochs=1, lr=1e-3, out_dir=str(tmp_path / "run"))
