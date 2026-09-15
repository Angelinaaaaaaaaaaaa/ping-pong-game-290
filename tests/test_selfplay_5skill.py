"""Tests for selfplay_5skill.py.

TDD: written before the fix. Pure Python, no MuJoCo -- covers the
per-iteration snapshot path derivation used by --save-history, which lets
performance be tracked over training (checkpoint at iter 100, 250, 500, ...)
instead of only ever having the single final/overwritten checkpoint.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from selfplay_5skill import history_checkpoint_path


class TestHistoryCheckpointPath:
    def test_inserts_iter_before_suffix(self):
        assert history_checkpoint_path("models/foo.pth", 100) == "models/foo_iter100.pth"

    def test_preserves_directory(self):
        path = history_checkpoint_path("models/nested/dir/selfplay_5skill_full2.pth", 250)
        assert path == "models/nested/dir/selfplay_5skill_full2_iter250.pth"

    def test_different_iterations_produce_different_paths(self):
        p1 = history_checkpoint_path("models/foo.pth", 50)
        p2 = history_checkpoint_path("models/foo.pth", 500)
        assert p1 != p2
        assert p1 == "models/foo_iter50.pth"
        assert p2 == "models/foo_iter500.pth"

    def test_iteration_zero(self):
        assert history_checkpoint_path("models/foo.pth", 0) == "models/foo_iter0.pth"
