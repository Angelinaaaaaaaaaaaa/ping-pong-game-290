"""nash_skills.sim_clone must reproduce a rally exactly from a mid-rally snapshot (real MuJoCo)."""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")


def _step(env, obs, info):
    from nash_skills.v2.selfplay_attack_defend import _build_ppo_obs
    act = np.zeros(18, dtype=np.float32)
    # A fixed, state-dependent action so the rally actually evolves.
    act[:9] = np.tanh(_build_ppo_obs(obs, info, 1)[18:27])
    act[9:] = np.tanh(_build_ppo_obs(obs, info, 2)[18:27])
    with contextlib.redirect_stdout(io.StringIO()):
        return env.step(act)


def test_restored_clone_replays_the_rally_bit_for_bit():
    from nash_skills.env_wrapper import SkillEnv
    from nash_skills.defend_mechanisms import set_defend_choices
    from nash_skills.sim_clone import restore_env, snapshot_env
    real = SkillEnv(proc_id=1, history=4, reset_mode="ready", skill_profile="aggressive", reset_targets=True)
    sim = SkillEnv(proc_id=1, history=4, reset_mode="ready", skill_profile="aggressive", reset_targets=True)
    np.random.seed(1)
    real.set_skills("left", "right")
    set_defend_choices(real, "preparation_cal", "left", "center")
    obs, info = real.reset()
    for _ in range(60):
        obs, _r, _d, _t, info = _step(real, obs, info)
    snap = snapshot_env(real, obs, info)
    ref = []
    for _ in range(200):
        obs, _r, done, _t, info = _step(real, obs, info)
        ref.append(real._env.data.body("ball").xpos.copy())
        if done:
            break
    obs2, info2 = restore_env(sim, snap)
    got = []
    for _ in range(len(ref)):
        obs2, _r, _d, _t, info2 = _step(sim, obs2, info2)
        got.append(sim._env.data.body("ball").xpos.copy())
    np.testing.assert_array_equal(np.array(ref), np.array(got))
    # restoring must not alias the snapshot: a second restore replays again
    obs3, _ = restore_env(sim, snap)
    np.testing.assert_array_equal(obs3, snap.obs)
    real.close()
    sim.close()
