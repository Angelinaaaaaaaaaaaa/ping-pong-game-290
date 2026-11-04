"""
Steering math for the attacker/defender extension (Design B).

Drives a defending player's gantry toward their chosen defend position by
reusing the existing per-step gantry_step scale (qpos += gantry_step *
action), rather than teleporting qpos directly -- so the gantry still moves
through the same physical update path as normal play, just with its action
computed from a target instead of a policy.

"Who is defending" is NOT decided here. It used to be derived from a
single instantaneous ball_vel_x reading, but a live PPO-driven trace
showed that reading goes through a small/noisy value right at every
racket hit (observed: -0.0805, settling to -3.7992 ten steps later) --
exactly the kind of transient a one-shot sign check can't distinguish
from a real direction change. That determination is now a separate,
stateful concern owned by nash_skills.defend_role.DefenderRoleTracker,
which latches the initial defender only once velocity clearly exceeds a
threshold and only flips on an actual net crossing. This module just
takes the already-resolved `defending_player` (1, 2, or None) and
answers "given who's defending and where they want to be, what action
gets them there."

This module is pure Python with no MuJoCo dependency, so it's fully unit
tested (tests/test_defend_steering.py). The live wiring that reads real
qpos/qvel/ball position from the simulator, drives the DefenderRoleTracker,
and calls this each step (nash_skills/env_wrapper.py) has been verified
against a live PPO-driven trace (scripts/trace_defend_engagement.py):
steering converges smoothly to the unmirrored target with no overshoot,
across repeated engage/disengage cycles within a single rally.
"""
from nash_skills.defend_positions import world_defend_qpos

__all__ = ["compute_defend_steering_action"]


def compute_defend_steering_action(defending_player, qpos1, qpos2, defend1, defend2, gantry_step):
    """
    Return (action1, action2): each is either None (this player isn't
    defending this step, or is defending but has no defend position set --
    no override either way) or an (x, y) pair in [-1, 1] that steers their
    gantry toward their chosen defend position this step.

    Parameters
    ----------
    defending_player : int or None -- 1, 2, or None if no defender has
                        latched yet (e.g. right at reset/serve, per
                        DefenderRoleTracker); None steers no one.
    qpos1, qpos2 : (x, y) -- each player's current gantry qpos
    defend1, defend2 : str or None -- each player's chosen defend position
                        name, or None if they didn't set one
    gantry_step : float -- the same per-step scale mujoco_env_comp.py uses
                  (gantry_speed_scale * 10.0 / 1000.0)
    """
    action1 = None
    action2 = None

    if defending_player == 1 and defend1 is not None:
        action1 = _steer_toward(qpos1, world_defend_qpos(1, defend1), gantry_step)
    if defending_player == 2 and defend2 is not None:
        action2 = _steer_toward(qpos2, world_defend_qpos(2, defend2), gantry_step)

    return action1, action2


def _steer_toward(current_qpos, target_qpos, gantry_step):
    cx, cy = current_qpos
    tx, ty = target_qpos
    ax = _clip((tx - cx) / gantry_step, -1.0, 1.0)
    ay = _clip((ty - cy) / gantry_step, -1.0, 1.0)
    return (ax, ay)


def _clip(value, lo, hi):
    return max(lo, min(hi, value))
