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


def test_valid_opponent_side_bounce_then_receiver_miss_awards_hitter():
    rally_state = mujoco_env_comp.new_rally_state()
    mujoco_env_comp._record_racket_hit(rally_state, "ego")
    mujoco_env_comp._record_table_bounce(rally_state, [2.0, 0.2, 0.56])

    done, winner, reason = mujoco_env_comp.infer_terminal_state(
        ball_pos=[2.1, 1.4, 0.8],
        ball_vel=[4.0, 0.0, 0.0],
        ego_racket_x=-0.5,
        opp_racket_x=3.5,
        rally_state=rally_state,
    )
    assert done is True
    assert winner == "ego"
    assert reason == "missed_opp_return_after_valid_bounce"
    assert rally_state["last_hitter"] == "ego"
    assert rally_state["last_table_bounce_side"] == "opp"
    assert rally_state["opp_table_bounces"] == 1
    assert rally_state["receiver_had_return_opportunity"] is True


def test_shot_that_goes_out_before_opponent_table_bounce_awards_receiver():
    rally_state = mujoco_env_comp.new_rally_state()
    mujoco_env_comp._record_racket_hit(rally_state, "ego")

    done, winner, reason = mujoco_env_comp.infer_terminal_state(
        ball_pos=[2.1, 1.4, 0.8],
        ball_vel=[4.0, 0.0, 0.0],
        ego_racket_x=-0.5,
        opp_racket_x=3.5,
        rally_state=rally_state,
    )
    assert done is True
    assert winner == "opp"
    assert reason == "ego_shot_out_before_valid_opp_bounce"
    assert rally_state["receiver_had_return_opportunity"] is False


def test_receiver_contact_after_valid_bounce_continues_rally_state():
    rally_state = mujoco_env_comp.new_rally_state()
    mujoco_env_comp._record_racket_hit(rally_state, "ego")
    mujoco_env_comp._record_table_bounce(rally_state, [2.0, 0.2, 0.56])
    mujoco_env_comp._record_racket_hit(rally_state, "opp")

    done, winner, reason = mujoco_env_comp.infer_terminal_state(
        ball_pos=[1.4, 0.2, 0.8],
        ball_vel=[-4.0, 0.0, 0.0],
        ego_racket_x=-0.5,
        opp_racket_x=3.5,
        rally_state=rally_state,
    )
    assert done is False
    assert winner is None
    assert reason is None
    assert rally_state["last_hitter"] == "opp"
    assert rally_state["receiver_had_return_opportunity"] is False
