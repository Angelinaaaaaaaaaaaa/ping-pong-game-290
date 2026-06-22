import csv

import pytest

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
