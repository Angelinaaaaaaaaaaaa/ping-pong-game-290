"""SkillEnv adds the ball's spin and the rally status to info (inputs of the v3 state encoder)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import contextlib
import io

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from nash_skills.rally_status import rally_status


@pytest.fixture(scope="module")
def env():
    from nash_skills.env_wrapper import SkillEnv
    with contextlib.redirect_stdout(io.StringIO()):
        e = SkillEnv(proc_id=1, history=4, reset_mode="ready", skill_profile="aggressive", reset_targets=True)
    yield e
    e.close()


def _check(env, info):
    np.testing.assert_allclose(info["ball_angvel"], env._env.data.qvel[-3:])
    assert info["rally_status"] == rally_status(env._env.rally_state)


def test_reset_and_step_report_spin_and_status(env):
    with contextlib.redirect_stdout(io.StringIO()):
        _obs, info = env.reset()
        _check(env, info)
        for _ in range(30):
            _obs, _r, done, _t, info = env.step(np.zeros(18, dtype=np.float32))
            _check(env, info)
            if done:
                break


def test_reported_spin_is_a_copy(env):
    with contextlib.redirect_stdout(io.StringIO()):
        _obs, info = env.reset()
    info["ball_angvel"][:] = 12345.0
    assert not np.any(env._env.data.qvel[-3:] == 12345.0)


def test_inner_env_without_a_rally_tracker_passes_info_through():
    from nash_skills.env_wrapper import SkillEnv

    class _Bare:
        pass
    wrapper = SkillEnv.__new__(SkillEnv)
    wrapper._env = _Bare()
    info = {"target": 1}
    assert wrapper._with_ball_status(info) == {"target": 1}
