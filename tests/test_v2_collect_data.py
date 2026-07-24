import pickle
import json

import numpy as np

import diagnostic_randomized_skills as diag
from nash_skills.skills import SKILL_NAMES, N_SKILLS, skill_index
from nash_skills.v2 import collect_data as collect
from nash_skills.v2.labeling import compute_returns
from nash_skills.v2.train_q_model_5skill_v2 import state_skill_pair_indices


def _skill_value(skill: str) -> float:
    return skill_index(skill) / (N_SKILLS - 1)


def _info():
    return {
        "diff_pos": np.zeros(3),
        "diff_quat": np.zeros(4),
        "target": np.zeros(7),
        "diff_pos_opp": np.zeros(3),
        "diff_quat_opp": np.zeros(4),
        "target_opp": np.zeros(7),
    }


def _obs(ball_x: float, skill1: str, skill2: str) -> np.ndarray:
    obs = np.zeros(116, dtype=np.float32)
    obs[36] = ball_x
    obs[-2] = _skill_value(skill1)
    obs[-1] = _skill_value(skill2)
    return obs


def test_random_initial_and_crossing_skill_selection_matches_diagnostic_seed():
    seed = 7
    setting_index = 0
    episode_index = 3
    c_rng = collect.episode_rng(seed, 1, setting_index, episode_index)
    d_rng = diag.episode_rng(seed, 1, setting_index, episode_index)
    setting = collect.settings_for_mode("random", fixed_player=1, fixed_skill=None)[0]

    assert collect.initial_skill_pair("random", setting, c_rng) == (
        diag.choose_skill("random", None, d_rng),
        diag.choose_skill("random", None, d_rng),
    )
    assert collect.next_skill_pair("random", setting, c_rng, ("left", "right")) == (
        diag.choose_skill("random", None, d_rng),
        diag.choose_skill("random", None, d_rng),
    )


def test_fixed_random_fixed_player_never_changes_and_matches_diagnostic_seed():
    seed = 11
    fixed_skill = "left"
    settings = collect.settings_for_mode("fixed_random", fixed_player=1, fixed_skill=None)
    setting_index = next(i for i, s in enumerate(settings) if s["fixed_skill"] == fixed_skill)
    setting = settings[setting_index]
    c_rng = collect.episode_rng(seed, 0, setting_index, 0)
    d_rng = diag.episode_rng(seed, 0, setting_index, 0)

    pairs = [
        collect.initial_skill_pair("fixed_random", setting, c_rng),
        collect.next_skill_pair("fixed_random", setting, c_rng, ("left", "right")),
        collect.next_skill_pair("fixed_random", setting, c_rng, ("left", "center_safe")),
    ]
    expected = [
        ("left", diag.choose_skill("random", None, d_rng)),
        ("left", diag.choose_skill("random", None, d_rng)),
        ("left", diag.choose_skill("random", None, d_rng)),
    ]
    assert pairs == expected
    assert {p1 for p1, _p2 in pairs} == {"left"}


def test_fixed_random_single_skill_keeps_diagnostic_setting_index():
    setting = collect.settings_for_mode("fixed_random", fixed_player=1, fixed_skill="right")[0]
    assert setting["setting_index"] == SKILL_NAMES.index("right")


def test_collect_passes_env_options_and_aligns_skill_pairs(tmp_path, monkeypatch):
    created = {}

    class FakeEnv:
        def __init__(self, proc_id, history, reset_mode, skill_profile, gantry_speed_scale):
            created.update({
                "proc_id": proc_id,
                "history": history,
                "reset_mode": reset_mode,
                "skill_profile": skill_profile,
                "gantry_speed_scale": gantry_speed_scale,
            })
            self.skill1 = "left"
            self.skill2 = "right"

        def set_skills(self, skill1, skill2):
            self.skill1 = skill1
            self.skill2 = skill2

        def reset(self):
            return _obs(1.0, self.skill1, self.skill2), _info()

        def step(self, _action):
            info = _info()
            info["winner"] = "ego"
            return _obs(2.0, self.skill1, self.skill2), 0.0, True, False, info

        def close(self):
            pass

    class FakeModel:
        def predict(self, _obs, deterministic=True):
            assert deterministic is True
            return np.zeros(9, dtype=np.float32), None

    class FakePPO:
        @staticmethod
        def load(_path):
            return FakeModel()

    monkeypatch.setattr(collect, "SkillEnv", FakeEnv)
    monkeypatch.setattr(collect, "PPO", FakePPO)
    monkeypatch.setattr(collect, "detect_winner", lambda _raw, done, info=None: 1 if done else 0)

    output = tmp_path / "rallies.pkl"
    rallies = collect.collect(
        target_rallies=1,
        output_path=str(output),
        ppo_path="fake",
        max_steps_per_episode=5,
        max_attempts_per_pair=1,
        progress_every=0,
        mode="fixed_random",
        fixed_player=1,
        fixed_skill="left",
        reset_mode="ready",
        skill_profile="aggressive",
        gantry_speed_scale=0.5,
        seed=5,
    )

    assert created["reset_mode"] == "ready"
    assert created["skill_profile"] == "aggressive"
    assert created["gantry_speed_scale"] == 0.5
    assert len(rallies) == 1
    rally = rallies[0]
    assert len(rally["states"]) == len(rally["skill_pairs"]) == 1
    assert set(rally) == {"skill1", "skill2", "skill_pairs", "states", "raw_obs", "winner"}
    assert rally["skill_pairs"][0][0] == "left"
    assert rally["states"][0][-2] == _skill_value(rally["skill_pairs"][0][0])
    assert rally["states"][0][-1] == _skill_value(rally["skill_pairs"][0][1])
    saved_rally = pickle.load(output.open("rb"))[0]
    assert saved_rally["skill_pairs"] == rally["skill_pairs"]
    assert saved_rally["winner"] == rally["winner"] == 1
    np.testing.assert_allclose(saved_rally["states"][0], rally["states"][0])
    assert compute_returns(saved_rally["states"], gamma=0.9, winner=saved_rally["winner"]) == compute_returns(
        rally["states"], gamma=0.9, winner=rally["winner"]
    )
    assert (tmp_path / "rallies_metadata.csv").exists()
    assert (tmp_path / "rallies_metadata.json").exists()
    metadata = json.loads((tmp_path / "rallies_metadata.json").read_text())
    attempt = metadata["attempts"][0]
    assert attempt["net_crossings"] == 1
    assert attempt["recorded_states"] == 1
    assert attempt["steps"] == 1
    assert attempt["accepted"] is True
    assert attempt["discard_reason"] == ""
    assert attempt["winner"] == 1


def test_training_grouping_prefers_skill_pairs_over_rally_level_skills():
    state = np.zeros(76, dtype=np.float32)
    state[-2] = _skill_value("right")
    state[-1] = _skill_value("center_safe")
    entry = {
        "skill1": "left",
        "skill2": "left",
        "skill_pairs": [("right", "center_safe")],
        "states": [state],
    }
    assert state_skill_pair_indices(entry, 0) == (skill_index("right"), skill_index("center_safe"))


def test_training_grouping_falls_back_to_encoded_state_skill_fields():
    state = np.zeros(76, dtype=np.float32)
    state[-2] = _skill_value("right_short")
    state[-1] = _skill_value("left_short")
    entry = {"skill1": "left", "skill2": "right", "states": [state]}
    assert state_skill_pair_indices(entry, 0) == (skill_index("right_short"), skill_index("left_short"))


def test_active_pair_counts_use_every_aligned_skill_pair():
    rallies = [
        {
            "skill1": "left",
            "skill2": "right",
            "states": [np.zeros(76), np.ones(76), np.full(76, 2.0)],
            "skill_pairs": [
                ("left", "right"),
                ("center_safe", "right_short"),
                ("left", "right"),
                ("ignored_extra_pair", "ignored_extra_pair"),
            ],
            "winner": 1,
        }
    ]

    counts = collect.active_decision_state_pair_counts(rallies)

    assert counts[("left", "right")] == 2
    assert counts[("center_safe", "right_short")] == 1
    assert ("ignored_extra_pair", "ignored_extra_pair") not in counts


def test_initial_rally_counts_and_decision_state_counts_are_separate(capsys):
    rallies = [
        {
            "skill1": "left",
            "skill2": "right",
            "states": [np.zeros(76), np.ones(76)],
            "skill_pairs": [("left", "right"), ("center_safe", "right_short")],
            "winner": 1,
        },
        {
            "skill1": "left",
            "skill2": "right",
            "states": [np.zeros(76)],
            "skill_pairs": [("right", "left")],
            "winner": 2,
        },
    ]

    initial_counts = collect.accepted_initial_pair_counts(rallies)
    active_counts = collect.active_decision_state_pair_counts(rallies)

    assert initial_counts == {("left", "right"): 2}
    assert active_counts == {
        ("left", "right"): 1,
        ("center_safe", "right_short"): 1,
        ("right", "left"): 1,
    }

    for title, counts in [
        ("Accepted rallies by initial skill pair:", initial_counts),
        ("Recorded decision states by active skill pair:", active_counts),
    ]:
        print(title)
        for (s1, s2), count in sorted(counts.items()):
            print(f"  {s1} vs {s2}: {count}")

    output = capsys.readouterr().out
    assert "Accepted rallies by initial skill pair:" in output
    assert "Recorded decision states by active skill pair:" in output


def test_crossing_bucket_percentages_are_correct():
    stats = collect.crossing_bucket_percentages([0, 1, 2, 3])

    assert stats["crossings_0_pct"] == 25.0
    assert stats["crossings_1_pct"] == 25.0
    assert stats["crossings_2plus_pct"] == 50.0
