import csv

import numpy as np
import pytest

import analyze_randomized_skill_diagnostics as analysis
import diagnostic_rendering as rendering
import diagnostic_randomized_skills as diag


def test_policy_types_for_modes():
    assert diag.policy_types("random_vs_random", None) == ("random", "random")
    assert diag.policy_types("fixed_vs_random", 1) == ("fixed", "random")
    assert diag.policy_types("fixed_vs_random", 2) == ("random", "fixed")


def test_rally_id_is_stable():
    assert diag.make_rally_id(0, "fixed_vs_random", "p1_fixed_left", 7) == "0:fixed_vs_random:p1_fixed_left:7"


def test_randomized_render_argument_parsing_defaults():
    args = diag.parse_args(["--mode", "random_vs_random", "--output-dir", "out"])
    assert args.render is False
    assert args.render_episodes is None
    assert args.render_truncated_only is False
    assert args.save_video is False
    assert args.video_dir == "data/rendered_rallies"
    assert args.video_fps == 60
    assert args.capture_every == 1


def test_randomized_render_argument_parsing_flags():
    args = diag.parse_args([
        "--mode",
        "fixed_vs_random",
        "--output-dir",
        "out",
        "--render",
        "--render-episodes",
        "3",
        "--render-truncated-only",
        "--save-video",
        "--video-dir",
        "videos",
        "--video-fps",
        "24",
        "--capture-every",
        "5",
    ])
    assert args.render is True
    assert args.render_episodes == "3"
    assert args.render_truncated_only is True
    assert args.save_video is True
    assert args.video_dir == "videos"
    assert args.video_fps == 24
    assert args.capture_every == 5


def test_randomized_manual_render_episode_flag_parses_without_value():
    args = diag.parse_args(["--mode", "random_vs_random", "--output-dir", "out", "--render-episodes"])
    assert args.render_episodes == "manual"
    assert rendering.manual_render_requested(args) is True


def test_render_truncated_only_selects_correct_episodes():
    args = diag.parse_args([
        "--mode",
        "random_vs_random",
        "--output-dir",
        "out",
        "--save-video",
        "--render-episodes",
        "2",
        "--render-truncated-only",
    ])
    saved = 0
    selections = []
    for truncated in [False, True, False, True, True]:
        selected = rendering.should_save_video(args, saved, truncated)
        selections.append(selected)
        if selected:
            saved += 1
    assert selections == [False, True, False, True, False]


def test_truncated_replay_selection_only_returns_truncated_rows():
    rows = [
        {"episode_id": 1, "truncated": False},
        {"episode_id": 2, "truncated": True},
        {"episode_id": 3, "truncated": "True"},
    ]
    assert [row["episode_id"] for row in rendering.select_truncated_replays(rows)] == [2, 3]
    assert [row["episode_id"] for row in rendering.select_truncated_replays(rows, limit=1)] == [2]


def test_manual_replay_selection_returns_only_requested_ids():
    rows = [
        {"episode_id": 318, "winner": "player1", "rally_length": 20, "truncated": False},
        {"episode_id": 319, "winner": "truncated", "rally_length": 600, "truncated": True},
        {"episode_id": 320, "winner": "player2", "rally_length": 12, "truncated": False},
    ]
    selected = rendering.select_manual_replays(rows, ["318", "320"])
    assert [row["episode_id"] for row in selected] == [318, 320]


def test_video_attempts_continue_until_save_limit_reached():
    args = diag.parse_args([
        "--mode",
        "random_vs_random",
        "--output-dir",
        "out",
        "--save-video",
        "--render-episodes",
        "2",
        "--render-truncated-only",
    ])
    saved = 0
    attempts = []
    for truncated in [False, True, False, True, True]:
        attempts.append(rendering.should_attempt_video(args, saved))
        if rendering.should_save_video(args, saved, truncated):
            saved += 1
    assert attempts == [True, True, True, True, False]


def test_randomized_evaluation_does_not_render_by_default():
    class FakeEnv:
        def __init__(self):
            self.render_calls = 0

        def set_skills(self, _p1, _p2):
            pass

        def reset(self):
            obs = np.zeros(120, dtype=np.float32)
            info = {
                "diff_pos": np.zeros(3),
                "diff_quat": np.zeros(4),
                "target": np.zeros(7),
                "diff_pos_opp": np.zeros(3),
                "diff_quat_opp": np.zeros(4),
                "target_opp": np.zeros(7),
                "initial_state": {},
            }
            return obs, info

        def step(self, _action):
            obs = np.zeros(120, dtype=np.float32)
            info = {"winner": "ego"}
            return obs, 0.0, True, False, info

        def render(self, mode="human"):
            self.render_calls += 1
            return np.zeros((1, 1, 3), dtype=np.uint8)

    class FakeModel:
        def predict(self, _obs, deterministic=True):
            return np.zeros(9, dtype=np.float32), None

    env = FakeEnv()
    diag.run_rally(
        env,
        FakeModel(),
        rally_id="r",
        seed=0,
        mode="random_vs_random",
        fixed_player=None,
        fixed_skill=None,
        max_steps=1,
        rng=np.random.default_rng(0),
        skill_profile="current",
    )
    assert env.render_calls == 0


def test_episode_video_recorder_deletes_unkept_temp_file(tmp_path, monkeypatch):
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
    recorder = rendering.EpisodeVideoRecorder(FakeEnv(), tmp_path, fps=30, capture_every=1)
    temp_path = recorder.temp_path
    recorder.capture(1)
    assert temp_path.exists()

    assert recorder.finish(False, tmp_path / "final.mp4") is None
    assert not temp_path.exists()
    assert not (tmp_path / "final.mp4").exists()


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
