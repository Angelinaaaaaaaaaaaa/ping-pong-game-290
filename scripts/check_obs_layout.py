"""
Check what the raw observation and the meta-level state encoders really contain.

KukaTennisEnv builds obs as concat(qpos[:18], qvel[:18], ...) (mujoco_env_comp.py:601, 738),
i.e. [P1 qpos, P2 qpos, P1 qvel, P2 qvel, ball, ...]. nash_skills/v2/state_encoder.py
assumes [P1 qpos, P1 qvel, P2 qpos, P2 qvel] (kept for old checkpoints); the Design B
joint-state encoder (state_encoder_attack_defend) reads the blocks correctly. This steps
a real env and compares each slice against data.qpos / data.qvel directly.

    MUJOCO_GL=cgl PYTHONPATH=. python scripts/check_obs_layout.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import contextlib
import io

import numpy as np

from nash_skills.env_wrapper import SkillEnv
from nash_skills.v2.state_encoder import encode_ego, encode_opp
from nash_skills.v2 import state_encoder_attack_defend as joint


def main():
    env = SkillEnv(proc_id=1, history=4, reset_mode="ready", skill_profile="aggressive")
    np.random.seed(0)
    obs, info = env.reset()
    rng = np.random.default_rng(0)
    for _ in range(40):  # random actions so velocities are non-zero
        with contextlib.redirect_stdout(io.StringIO()):
            obs, _r, _done, _t, info = env.step(rng.uniform(-1, 1, 18).astype(np.float32))
    d = env._env.data
    truth = {"P1 qpos": d.qpos[0:9], "P2 qpos": d.qpos[9:18], "P1 qvel": d.qvel[0:9], "P2 qvel": d.qvel[9:18]}

    def what(x):
        hits = [name for name, v in truth.items() if np.allclose(x, v, atol=1e-5)]
        return hits[0] if hits else "none of the four"

    print("raw obs:")
    for lo in (0, 9, 18, 27):
        print(f"  obs[{lo}:{lo + 9}] = {what(obs[lo:lo + 9])}")
    e, o = encode_ego(obs, info), encode_opp(obs, info)
    print("encode_ego (P1's state):")
    print(f"  [0:9]  labelled 'ego gantry + arm'      = {what(e[0:9])}")
    print(f"  [9:18] labelled 'ego joint velocities'  = {what(e[9:18])}")
    print("encode_opp (P2's state):")
    print(f"  [0:9]  labelled 'opp gantry + arm'      = {what(o[0:9])}")
    print(f"  [9:18] labelled 'opp joint velocities'  = {what(o[9:18])}")
    j = joint.encode_ego(obs, info, "left", "right")
    print(f"Design B joint state ({joint.STATE_ENCODER_NAME}):")
    for lo, name in ((0, "P1 qpos"), (9, "P1 qvel"), (18, "P2 qpos"), (27, "P2 qvel")):
        got = what(j[lo:lo + 9])
        print(f"  [{lo}:{lo + 9}] should be {name:<8} = {got}  {'OK' if got == name else 'WRONG'}")
    env.close()


if __name__ == "__main__":
    main()
