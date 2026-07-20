import csv

import pytest

import analyze_randomized_skill_diagnostics as analysis
import diagnostic_randomized_skills as diag


def test_policy_types_for_modes():
    assert diag.policy_types("random_vs_random", None) == ("random", "random")
    assert diag.policy_types("fixed_vs_random", 1) == ("fixed", "random")
    assert diag.policy_types("fixed_vs_random", 2) == ("random", "fixed")


def test_rally_id_is_stable():
    assert diag.make_rally_id(0, "fixed_vs_random", "p1_fixed_left", 7) == "0:fixed_vs_random:p1_fixed_left:7"


def test_write_csv_creates_header_for_empty_rows(tmp_path):
    path = tmp_path / "empty.csv"
    diag.write_csv(path, [], ["a", "b"])
    with path.open(newline="") as f:
        assert list(csv.reader(f)) == [["a", "b"]]


def test_fixed_vs_random_summary_and_distributions():
    rallies = analysis.normalize_rallies([
        {
            "rally_id": "r1",
            "seed": "0",
            "mode": "fixed_vs_random",
            "fixed_player": "1",
            "fixed_skill": "left",
            "p1_policy_type": "fixed",
            "p2_policy_type": "random",
            "winner": "player1",
            "truncated": "False",
            "rally_length": "10",
            "num_decisions": "2",
            "max_steps": "600",
        },
        {
            "rally_id": "r2",
            "seed": "0",
            "mode": "fixed_vs_random",
            "fixed_player": "1",
            "fixed_skill": "left",
            "p1_policy_type": "fixed",
            "p2_policy_type": "random",
            "winner": "player2",
            "truncated": "False",
            "rally_length": "20",
            "num_decisions": "2",
            "max_steps": "600",
        },
        {
            "rally_id": "r3",
            "seed": "0",
            "mode": "fixed_vs_random",
            "fixed_player": "1",
            "fixed_skill": "left",
            "p1_policy_type": "fixed",
            "p2_policy_type": "random",
            "winner": "truncated",
            "truncated": "True",
            "rally_length": "600",
            "num_decisions": "1",
            "max_steps": "600",
        },
    ])
    decisions = analysis.normalize_decisions([
        {"rally_id": "r1", "decision_t": "0", "player": "2", "chosen_skill": "right", "winner": "player1", "truncated": "False", "rally_length": "10", "mode": "fixed_vs_random", "fixed_player": "1", "fixed_skill": "left"},
        {"rally_id": "r1", "decision_t": "1", "player": "2", "chosen_skill": "left_short", "winner": "player1", "truncated": "False", "rally_length": "10", "mode": "fixed_vs_random", "fixed_player": "1", "fixed_skill": "left"},
        {"rally_id": "r2", "decision_t": "0", "player": "2", "chosen_skill": "center_safe", "winner": "player2", "truncated": "False", "rally_length": "20", "mode": "fixed_vs_random", "fixed_player": "1", "fixed_skill": "left"},
        {"rally_id": "r3", "decision_t": "0", "player": "2", "chosen_skill": "right_short", "winner": "truncated", "truncated": "True", "rally_length": "600", "mode": "fixed_vs_random", "fixed_player": "1", "fixed_skill": "left"},
    ])

    summary, distributions, last_skills = analysis.summarize_fixed_vs_random(rallies, decisions)
    left = next(row for row in summary if row["fixed_player"] == 1 and row["fixed_skill"] == "left")
    assert left["fixed_player_win_rate"] == pytest.approx(1 / 3)
    assert left["random_player_win_rate"] == pytest.approx(1 / 3)
    assert left["truncation_rate"] == pytest.approx(1 / 3)
    assert left["median_rally_length"] == 20

    win_scope = "fixed_player=1;fixed_skill=left;random_wins"
    win_dist = {row["skill"]: row for row in distributions if row["scope"] == win_scope}
    assert win_dist["center_safe"]["count"] == 1

    last_scope = "fixed_player=1;fixed_skill=left;random_last_before_terminal"
    last_dist = {row["skill"]: row for row in last_skills if row["scope"] == last_scope}
    assert last_dist["left_short"]["count"] == 1
    assert last_dist["center_safe"]["count"] == 1
    assert last_dist["right_short"]["count"] == 1


def test_random_vs_random_summary_counts_player_distributions():
    rallies = analysis.normalize_rallies([
        {"rally_id": "r1", "seed": "0", "mode": "random_vs_random", "fixed_player": "", "fixed_skill": "", "p1_policy_type": "random", "p2_policy_type": "random", "winner": "player1", "truncated": "False", "rally_length": "10", "num_decisions": "1", "max_steps": "600"},
        {"rally_id": "r2", "seed": "0", "mode": "random_vs_random", "fixed_player": "", "fixed_skill": "", "p1_policy_type": "random", "p2_policy_type": "random", "winner": "truncated", "truncated": "True", "rally_length": "600", "num_decisions": "1", "max_steps": "600"},
    ])
    decisions = analysis.normalize_decisions([
        {"rally_id": "r1", "decision_t": "0", "player": "1", "chosen_skill": "left", "winner": "player1", "truncated": "False", "rally_length": "10", "mode": "random_vs_random", "fixed_player": "", "fixed_skill": ""},
        {"rally_id": "r1", "decision_t": "0", "player": "2", "chosen_skill": "right", "winner": "player1", "truncated": "False", "rally_length": "10", "mode": "random_vs_random", "fixed_player": "", "fixed_skill": ""},
        {"rally_id": "r2", "decision_t": "0", "player": "1", "chosen_skill": "center_safe", "winner": "truncated", "truncated": "True", "rally_length": "600", "mode": "random_vs_random", "fixed_player": "", "fixed_skill": ""},
        {"rally_id": "r2", "decision_t": "0", "player": "2", "chosen_skill": "left_short", "winner": "truncated", "truncated": "True", "rally_length": "600", "mode": "random_vs_random", "fixed_player": "", "fixed_skill": ""},
    ])

    summary, distributions, last_skills = analysis.summarize_random_vs_random(rallies, decisions)
    assert summary[0]["p1_win_rate"] == pytest.approx(0.5)
    assert summary[0]["truncation_rate"] == pytest.approx(0.5)
    p1_scope = "random_vs_random;player=1;overall"
    p1_dist = {row["skill"]: row for row in distributions if row["scope"] == p1_scope}
    assert p1_dist["left"]["count"] == 1
    assert p1_dist["center_safe"]["count"] == 1
    last_dist = {row["skill"]: row for row in last_skills}
    assert last_dist["left"]["count"] == 1
    assert last_dist["right"]["count"] == 1
