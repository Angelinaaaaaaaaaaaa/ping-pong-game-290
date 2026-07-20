import sys
import types


def _install_import_stubs():
    mujoco = types.ModuleType("mujoco")
    mujoco.glfw = types.ModuleType("mujoco.glfw")
    mujoco.glfw.glfw = object()
    mujoco.MjModel = object
    mujoco.MjData = object
    mujoco.mjtObj = types.SimpleNamespace(mjOBJ_GEOM=0, mjOBJ_BODY=1)
    mujoco.mjtCamera = types.SimpleNamespace(mjCAMERA_FIXED=0)
    mujoco.mj_name2id = lambda *args, **kwargs: 0
    mujoco.mj_id2name = lambda *args, **kwargs: None
    mujoco.mj_step = lambda *args, **kwargs: None
    mujoco.mj_jac = lambda *args, **kwargs: None
    sys.modules.setdefault("mujoco", mujoco)
    sys.modules.setdefault("mujoco.glfw", mujoco.glfw)

    gymnasium = types.ModuleType("gymnasium")
    gymnasium.Env = object
    gymnasium.spaces = types.SimpleNamespace(Box=lambda *args, **kwargs: None)
    sys.modules.setdefault("gymnasium", gymnasium)


_install_import_stubs()

import mujoco_env_comp


def test_terminal_state_keeps_existing_x_boundary_winners():
    done, winner, reason = mujoco_env_comp.infer_terminal_state(
        ball_pos=[-0.9, 0.0, 0.8],
        ball_vel=[-4.0, 0.0, 0.0],
        ego_racket_x=-0.5,
        opp_racket_x=3.5,
    )
    assert done is True
    assert winner == "opp"
    assert reason == "ball_past_ego_racket"

    done, winner, reason = mujoco_env_comp.infer_terminal_state(
        ball_pos=[3.9, 0.0, 0.8],
        ball_vel=[4.0, 0.0, 0.0],
        ego_racket_x=-0.5,
        opp_racket_x=3.5,
    )
    assert done is True
    assert winner == "ego"
    assert reason == "ball_past_opp_racket"


def test_terminal_state_does_not_end_near_table_width():
    done, winner, reason = mujoco_env_comp.infer_terminal_state(
        ball_pos=[1.5, 0.9, 0.8],
        ball_vel=[4.0, 0.0, 0.0],
        ego_racket_x=-0.5,
        opp_racket_x=3.5,
    )
    assert done is False
    assert winner is None
    assert reason is None


def test_terminal_state_ends_lateral_flyaway_and_assigns_last_hitter_loss():
    done, winner, reason = mujoco_env_comp.infer_terminal_state(
        ball_pos=[1.5, 1.4, 0.8],
        ball_vel=[4.0, 0.0, 0.0],
        ego_racket_x=-0.5,
        opp_racket_x=3.5,
    )
    assert done is True
    assert winner == "opp"
    assert reason == "ball_lateral_out_toward_opp"

    done, winner, reason = mujoco_env_comp.infer_terminal_state(
        ball_pos=[1.5, -1.4, 0.8],
        ball_vel=[-4.0, 0.0, 0.0],
        ego_racket_x=-0.5,
        opp_racket_x=3.5,
    )
    assert done is True
    assert winner == "ego"
    assert reason == "ball_lateral_out_toward_ego"
