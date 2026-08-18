import pickle

import numpy as np

from nash_skills.skills import N_SKILLS, skill_index
from nash_skills.v2 import collect_data_v3_aligned as aligned


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
        "winner": "ego",
    }


def _obs(ball_x: float, skill1: str, skill2: str) -> np.ndarray:
    obs = np.zeros(116, dtype=np.float32)
    obs[36] = ball_x
    obs[-2] = _skill_value(skill1)
    obs[-1] = _skill_value(skill2)
    return obs


class FakeEnv:
    def __init__(self, proc_id, history, reset_mode, skill_profile, gantry_speed_scale):
        self.skill1 = "left"
        self.skill2 = "left"
        self.step_count = 0
        self.closed = False

    def set_skills(self, skill1, skill2):
        self.skill1 = skill1
        self.skill2 = skill2

    def reset(self):
        self.step_count = 0
        return _obs(1.0, self.skill1, self.skill2), _info()

    def step(self, _action):
        self.step_count += 1
        ball_x = 2.0 if self.step_count % 2 == 1 else 1.0
        done = self.step_count == 3
        return _obs(ball_x, self.skill1, self.skill2), 0.0, done, False, _info()

    def close(self):
        self.closed = True


class FakeModel:
    def predict(self, _obs, deterministic=True):
        assert deterministic is True
        return np.zeros(9, dtype=np.float32), None


class FakePPO:
    @staticmethod
    def load(_path):
        return FakeModel()


def test_aligned_collector_stores_next_skill_pair_for_crossing_state(tmp_path, monkeypatch, capsys):
    sampled_pairs = iter(
        [
            ("left", "right"),
            ("left_short", "right"),
            ("left_short", "center_safe"),
        ]
    )

    monkeypatch.setattr(aligned, "SkillEnv", FakeEnv)
    monkeypatch.setattr(aligned, "PPO", FakePPO)
    monkeypatch.setattr(aligned, "detect_winner", lambda _raw, done, info=None: 1 if done else 0)
    monkeypatch.setattr(aligned, "initial_skill_pair", lambda *_args, **_kwargs: ("left", "left"))
    monkeypatch.setattr(aligned, "next_skill_pair", lambda *_args, **_kwargs: next(sampled_pairs))

    output = tmp_path / "aligned.pkl"
    rallies = aligned.collect(
        target_rallies=1,
        output_path=str(output),
        ppo_path="fake",
        max_steps_per_episode=5,
        max_attempts_per_pair=1,
        progress_every=0,
        mode="random",
        seed=0,
        quiet=False,
        debug_crossings=True,
    )

    assert len(rallies) == 1
    rally = rallies[0]
    assert len(rally["states"]) == len(rally["skill_pairs"]) == 3
    assert rally["skill_pairs"] == [
        ("left", "right"),
        ("left_short", "right"),
        ("left_short", "center_safe"),
    ]
    assert rally["skill1"] == "left"
    assert rally["skill2"] == "right"
    assert rally["winner"] == 1

    for state, raw_obs, (p1_skill, p2_skill) in zip(
        rally["states"], rally["raw_obs"], rally["skill_pairs"]
    ):
        assert state[-2] == _skill_value(p1_skill)
        assert state[-1] == _skill_value(p2_skill)
        assert raw_obs[-2] == _skill_value(p1_skill)
        assert raw_obs[-1] == _skill_value(p2_skill)

    saved = pickle.load(output.open("rb"))
    assert saved[0]["skill_pairs"] == rally["skill_pairs"]
    assert len(saved[0]["states"]) == len(saved[0]["skill_pairs"])

    out = capsys.readouterr().out
    assert "crossing attempt=0 index=0" in out
    assert "old_pair=('left', 'left')" in out
    assert "new_pair=('left', 'right')" in out
    assert "stored_pair=('left', 'right')" in out


def test_aligned_helpers_keep_p1_p2_tuple_order():
    obs = _obs(1.5, "right", "left")
    state = aligned._encode_state_for_pair(obs, _info(), ("left_short", "center_safe"))
    raw = aligned._raw_obs_for_pair(obs, ("left_short", "center_safe"))

    assert state[-2] == _skill_value("left_short")
    assert state[-1] == _skill_value("center_safe")
    assert raw[-2] == _skill_value("left_short")
    assert raw[-1] == _skill_value("center_safe")
