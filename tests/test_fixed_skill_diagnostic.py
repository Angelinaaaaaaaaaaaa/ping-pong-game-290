import csv

import numpy as np
import pytest

import diagnostic_rendering as rendering
import diagnostic_fixed_skill as diag
from nash_skills.skills import SKILL_NAMES


def _row(seed, skill, episode_index, winner, steps=600, decisions=9):
    return {
        "key": diag.episode_key(seed, skill, episode_index),
        "seed": seed,
        "episode_index": episode_index,
        "player1_skill": skill,
        "player2_skill": "center_safe",
        "winner": winner,
        "termination_reason": "step_limit" if winner == "truncated" else "env_done",
        "reached_step_limit": winner == "truncated",
        "physics_steps": steps,
        "decision_state_count": decisions,
        "raw_obs_ids": "[0.0, 0.5]",
        "state_ids": "[0.0, 0.5]",
        "player1_target_xy": "[2.185, -0.38]",
        "player2_target_xy": "[1.85, 0.0]",
        "validation_ok": True,
        "validation_errors": "[]",
    }


def test_episode_key_is_seed_skill_episode():
    assert diag.episode_key(2, "left_short", 7) == "2:left_short:7"


def test_fixed_skill_render_argument_parsing_defaults():
    args = diag.parse_args(["--fixed-player", "2", "--fixed-skill", "center_safe"])
    assert args.render is False
    assert args.render_episodes is None
    assert args.render_truncated_only is False
    assert args.save_video is False
    assert args.video_dir == "data/rendered_rallies"
    assert args.video_fps == 60
    assert args.capture_every == 1


def test_fixed_skill_render_argument_parsing_flags():
    args = diag.parse_args([
        "--fixed-player",
        "1",
        "--fixed-skill",
        "left",
        "--render",
        "--render-episodes",
        "4",
        "--render-truncated-only",
        "--save-video",
        "--video-dir",
        "videos",
        "--video-fps",
        "30",
        "--capture-every",
        "4",
    ])
    assert args.render is True
    assert args.render_episodes == "4"
    assert args.render_truncated_only is True
    assert args.save_video is True
    assert args.video_dir == "videos"
    assert args.video_fps == 30
    assert args.capture_every == 4


def test_fixed_skill_manual_render_episode_flag_parses_without_value():
    args = diag.parse_args(["--fixed-player", "2", "--fixed-skill", "center_safe", "--render-episodes"])
    assert args.render_episodes == "manual"
    assert rendering.manual_render_requested(args) is True


def test_fixed_skill_truncated_only_video_selection_respects_limit():
    args = diag.parse_args([
        "--fixed-player",
        "2",
        "--fixed-skill",
        "right",
        "--save-video",
        "--render-truncated-only",
        "--render-episodes",
        "1",
    ])
    assert rendering.should_save_video(args, selected_count=0, truncated=False) is False
    assert rendering.should_save_video(args, selected_count=0, truncated=True) is True
    assert rendering.should_save_video(args, selected_count=1, truncated=True) is False


def test_fixed_skill_truncated_replay_selection_only_returns_step_limit_rows():
    rows = [
        {"episode_id": 1, "reached_step_limit": False},
        {"episode_id": 2, "reached_step_limit": True},
        {"episode_id": 3, "reached_step_limit": "true"},
    ]
    assert [row["episode_id"] for row in rendering.select_truncated_replays(rows)] == [2, 3]


def test_fixed_skill_manual_replay_selection_only_returns_requested_ids():
    rows = [
        {"episode_id": 10, "winner": "player1", "physics_steps": 10, "reached_step_limit": False},
        {"episode_id": 11, "winner": "truncated", "physics_steps": 600, "reached_step_limit": True},
    ]
    assert [row["episode_id"] for row in rendering.select_manual_replays(rows, ["11"])] == [11]


def test_fixed_skill_video_attempts_stop_after_save_limit():
    args = diag.parse_args([
        "--fixed-player",
        "2",
        "--fixed-skill",
        "right",
        "--save-video",
        "--render-episodes",
        "1",
    ])
    assert rendering.should_attempt_video(args, saved_count=0) is True
    assert rendering.should_save_video(args, selected_count=0, truncated=False) is True
    assert rendering.should_attempt_video(args, saved_count=1) is False


def test_episode_video_recorder_moves_kept_temp_file(tmp_path, monkeypatch):
    class FakeWriter:
        def __init__(self, path):
            self.path = path

        def append_data(self, _frame):
            self.path.write_bytes(b"frame")

        def close(self):
            pass

    class FakeEnv:
        def render(self, mode="human"):
            assert mode == "rgb_array"
            return np.zeros((1, 1, 3), dtype=np.uint8)

    def fake_ensure_writer(self):
        if self._writer is None:
            self._writer = FakeWriter(self.temp_path)
        return self._writer

    monkeypatch.setattr(rendering.EpisodeVideoRecorder, "_ensure_writer", fake_ensure_writer)
    recorder = rendering.EpisodeVideoRecorder(FakeEnv(), tmp_path, fps=30, capture_every=2)
    temp_path = recorder.temp_path
    final_path = tmp_path / "final.mp4"
    recorder.capture(1)
    recorder.capture(2)

    assert recorder.finish(True, final_path) == final_path
    assert not temp_path.exists()
    assert final_path.read_bytes() == b"frame"


def test_episode_rng_seed_is_resume_stable_and_skill_specific():
    first = diag.episode_rng_seed(0, "left", 3)
    assert first == diag.episode_rng_seed(0, "left", 3)
    assert first != diag.episode_rng_seed(0, "left_short", 3)
    assert first != diag.episode_rng_seed(1, "left", 3)
    assert first != diag.episode_rng_seed(0, "left", 4)


def test_planned_keys_cover_all_player1_skills():
    keys = diag.planned_keys([0, 1], 3)
    assert len(keys) == len(SKILL_NAMES) * 2 * 3
    assert "0:left:0" in keys
    assert "1:right:2" in keys


def test_duplicate_episode_keys_raise():
    rows = [_row(0, "left", 0, "player1"), _row(0, "left", 0, "player2")]
    with pytest.raises(ValueError, match="Duplicate episode key"):
        diag.dedupe_rows(rows)


def test_existing_keys_detects_duplicate_resume_rows():
    rows = [_row(0, "left", 0, "player1"), _row(0, "left", 0, "player2")]
    with pytest.raises(ValueError, match="Duplicate episode key"):
        diag.existing_keys(rows)


def test_resume_rows_must_match_requested_fixed_skill_and_plan():
    diag.validate_resume_rows([_row(0, "left", 0, "player1")], "center_safe", [0], 1)

    with pytest.raises(ValueError, match="player2_skill"):
        diag.validate_resume_rows([_row(0, "left", 0, "player1")], "right", [0], 1)

    with pytest.raises(ValueError, match="outside requested seeds"):
        diag.validate_resume_rows([_row(1, "left", 0, "player1")], "center_safe", [0], 1)

    with pytest.raises(ValueError, match="outside requested range"):
        diag.validate_resume_rows([_row(0, "left", 2, "player1")], "center_safe", [0], 1)


def test_aggregate_does_not_count_truncations_as_losses():
    rows = [
        _row(0, "left", 0, "player1", steps=71, decisions=1),
        _row(0, "left", 1, "player2", steps=72, decisions=1),
        _row(0, "left", 2, "truncated", steps=600, decisions=10),
    ]
    summary = {row["player1_skill"]: row for row in diag.aggregate_rows(rows, "center_safe")}
    left = summary["left"]
    assert left["episode_count"] == 3
    assert left["player1_wins"] == 1
    assert left["player2_wins"] == 1
    assert left["step_limit_count"] == 1
    assert left["completed_count"] == 2
    assert left["completed_player1_win_rate"] == 0.5
    assert left["step_limit_rate"] == pytest.approx(1 / 3)


def test_aggregate_handles_no_completed_rallies_without_division_by_zero():
    rows = [_row(0, "right", i, "truncated", steps=600, decisions=9) for i in range(3)]
    summary = {row["player1_skill"]: row for row in diag.aggregate_rows(rows, "center_safe")}
    right = summary["right"]
    assert right["completed_count"] == 0
    assert right["completed_player1_win_rate"] is None
    assert right["completed_player1_win_ci_low"] is None
    assert right["completed_player1_win_ci_high"] is None


def test_write_csv_creates_header_for_empty_rows(tmp_path):
    path = tmp_path / "empty.csv"
    diag.write_csv(path, [], ["a", "b"])
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    assert rows == [["a", "b"]]


def test_plot_outputs_handles_zero_sample_categories(tmp_path):
    rows = [_row(0, "left", 0, "truncated", steps=600, decisions=9)]
    summary = diag.aggregate_rows(rows, "center_safe")
    plots = diag.plot_outputs(tmp_path, rows, summary, [], steps=600)
    assert {path.rsplit("/", 1)[-1] for path in plots} == {
        "outcome_distribution.png",
        "truncation_rate.png",
        "rally_duration_distribution.png",
        "completed_win_rate.png",
        "target_coordinates.png",
        "representative_long_rallies.png",
    }
