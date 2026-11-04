"""Tests for nash_skills.v2.diag_skill_selection_probs._validate_rally_schema.

TDD: written before implementation. Covers the code-review HIGH finding --
main() crashed with an opaque AttributeError when pointed at a legacy
non-dict rally pickle (e.g. data_new/rallies.pkl, a flat list of raw state
arrays). This validator fails fast with a clear ValueError instead.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from nash_skills.v2.diag_skill_selection_probs import _validate_rally_schema


class TestValidateRallySchema:
    def test_accepts_dict_entries(self):
        rallies = [{"skill1": "left", "skill2": "right", "states": [], "winner": 1}]
        _validate_rally_schema(rallies, "some_file.pkl")  # should not raise

    def test_rejects_non_dict_first_entry(self):
        rallies = [np.zeros(116, dtype=np.float32), np.zeros(116, dtype=np.float32)]
        with pytest.raises(ValueError, match="rally-dict schema"):
            _validate_rally_schema(rallies, "rallies.pkl")

    def test_error_message_includes_source(self):
        rallies = [np.zeros(116, dtype=np.float32)]
        with pytest.raises(ValueError, match="rallies.pkl"):
            _validate_rally_schema(rallies, "rallies.pkl")

    def test_empty_list_does_not_raise(self):
        _validate_rally_schema([], "empty.pkl")  # should not raise

    def test_rejects_plain_list_entries(self):
        rallies = [[1, 2, 3], [4, 5, 6]]
        with pytest.raises(ValueError, match="rally-dict schema"):
            _validate_rally_schema(rallies, "rallies_backup.pkl")
