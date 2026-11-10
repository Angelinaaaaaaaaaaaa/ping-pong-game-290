"""
Exact snapshots of a SkillEnv mid-rally, restorable into a second SkillEnv
(same model), for simulated look-ahead (simulation-based IBR).

MjData is copied whole with mj_copyData. mj_getState/mj_setState + mj_forward
is not enough: KukaTennisEnv reads quantities (contacts) left over from the
previous mj_step, which mj_forward recomputes at the new positions, and the
replay then drifts. The Python-side state of both the inner env and the
wrapper (targets, rally state, action history, role trackers, choices) is
deep-copied; MuJoCo handles and bound methods are not.

Stepping is deterministic (randomness only enters at reset), so a restored
clone replays the original exactly under the same actions.
"""
import copy
import dataclasses
import types

import mujoco

__all__ = ["EnvSnapshot", "snapshot_env", "restore_env"]

_INNER_SKIP = {"model", "data", "viewer", "_rgb_renderer", "action_space", "observation_space"}


@dataclasses.dataclass
class EnvSnapshot:
    data: object
    inner: dict
    outer: dict
    obs: object
    info: dict


def _copyable(key, value, skip):
    return key not in skip and not isinstance(value, types.MethodType)


def snapshot_env(env, obs, info) -> EnvSnapshot:
    inner = env._env
    data = mujoco.MjData(inner.model)
    mujoco.mj_copyData(data, inner.model, inner.data)
    return EnvSnapshot(
        data=data,
        inner={k: copy.deepcopy(v) for k, v in vars(inner).items() if _copyable(k, v, _INNER_SKIP)},
        outer={k: copy.deepcopy(v) for k, v in vars(env).items() if _copyable(k, v, {"_env"})},
        obs=copy.deepcopy(obs), info=copy.deepcopy(info),
    )


def restore_env(env, snap: EnvSnapshot):
    """Put `env` into the snapshot's state; returns fresh copies of (obs, info)."""
    inner = env._env
    mujoco.mj_copyData(inner.data, inner.model, snap.data)
    for k, v in snap.inner.items():
        setattr(inner, k, copy.deepcopy(v))
    for k, v in snap.outer.items():
        setattr(env, k, copy.deepcopy(v))
    return copy.deepcopy(snap.obs), copy.deepcopy(snap.info)
