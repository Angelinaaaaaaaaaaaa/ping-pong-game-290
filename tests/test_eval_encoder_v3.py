"""The Table 1 evaluator follows the models' state encoder (v2: 90 inputs, v3: 97 with spin and status)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

import eval_table1_attack_defend as t1
from model_arch import SimpleModel
from nash_skills.v2.state_encoder_attack_defend import ENCODER_V2, ENCODER_V3
from tests.test_state_encoder_v3 import _info, _obs


def _model_dir(tmp_path, encoder, dim):
    d = tmp_path / encoder
    d.mkdir()
    (d / "heldout_attack_defend.json").write_text(json.dumps({"encoder": encoder}))
    for key, name in t1.MODEL_NAMES.items():
        model = SimpleModel(dim, [64, 32, 16], 1, last_layer_activation=None if key == "model_p" else "tanh")
        torch.save(model.state_dict(), d / name)
    return str(d)


def test_check_model_encoder_returns_the_encoder(tmp_path):
    assert t1.check_model_encoder(_model_dir(tmp_path, ENCODER_V3, 97)) == ENCODER_V3
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "heldout_attack_defend.json").write_text(json.dumps({"encoder": "one_hot_v1"}))
    with pytest.raises(ValueError, match="one_hot_v1"):
        t1.check_model_encoder(str(legacy))


def test_load_models_sizes_networks_by_encoder(tmp_path):
    models = t1._load_models(_model_dir(tmp_path, ENCODER_V3, 97))
    assert models["encoder"] == ENCODER_V3
    with torch.no_grad():
        assert models["model_p"](torch.zeros(2, 97)).shape == (2, 1)
    assert t1._load_models(None)["encoder"] == ENCODER_V2


def test_phi_encoder_matches_the_model(tmp_path):
    state = t1.make_phi_encoder(ENCODER_V3)(_obs(), _info(), 1)
    assert state.shape == (97,)
    assert t1.make_phi_encoder(ENCODER_V2)(_obs(), _info(), 2).shape == (90,)
    np.testing.assert_array_equal(t1._encode_for_phi(_obs(), _info(), 1), t1.make_phi_encoder(ENCODER_V2)(_obs(), _info(), 1))
