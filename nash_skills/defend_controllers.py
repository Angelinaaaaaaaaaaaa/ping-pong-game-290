"""
Controller-design family for the attacker/defender defend-steering A/B.

Isolates two independent design choices in how a defending player's
gantry is driven toward their chosen pre-position target, to find which
one explains why this project's own override suppressed return ability
(ego_wins 9-10/10 in every tested combo, even where the target was
reached 100% of the time) while a teammate's structurally similar "V3"
pipeline did not (sustained rallies averaging 5.6 decisions, 100%
return-success-per-attempt in a smoke test) -- despite both overriding
the exact same two gantry action components:

  - approach law: how the per-step action is computed while still short
    of the target.
      "bangbang"     -- this project's original _steer_toward (see
                        nash_skills.defend_steering): full speed
                        (action = clip(error / gantry_step, -1, 1)) for
                        the entire approach, landing exactly on target
                        in one discrete step once within gantry_step.
      "proportional" -- the teammate's V3 style (v3/env_wrapper.py):
                        action = clip(error * gain, -1, 1), decelerating
                        smoothly over roughly 1/gain meters before
                        settling.

  - post-tolerance behavior: what happens once the target has been
    reached (within tolerance).
      "release" -- this project's current nash_skills.defend_engagement
                   design: stop overriding entirely (return None), hand
                   full gantry control back to the PPO controller for
                   the rest of the defending phase.
      "freeze"  -- the teammate's V3 style: keep overriding with an
                   explicit (0, 0) command for the rest of the
                   defending phase, rather than releasing control.

Four named controllers cover the full 2x2: bangbang_release (this
project's current production design), proportional_release,
bangbang_freeze, proportional_freeze (closest replica of the teammate's
V3 design). Pure Python, no MuJoCo dependency -- fully unit tested
(tests/test_defend_controllers.py). The live, instrumented A/B harness
lives in scripts/ab_defend_controllers.py.

This module is deliberately NOT wired into the production path
(nash_skills.env_wrapper / nash_skills.defend_steering /
nash_skills.defend_engagement) -- it is a diagnostic tool to decide
whether/how to change that production design, not yet a replacement for
it.
"""
from __future__ import annotations

__all__ = ["CONTROLLER_NAMES", "compute_action", "PROPORTIONAL_GAIN"]

CONTROLLER_NAMES = (
    "bangbang_release",
    "proportional_release",
    "bangbang_freeze",
    "proportional_freeze",
)

PROPORTIONAL_GAIN = 8.0  # matches the teammate's V3 default position_gain


def _clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def compute_action(controller: str, current_qpos, target_qpos, gantry_step: float,
                    tolerance: float, reached_before: bool):
    """
    Returns (action_or_none, reached_now).

    action_or_none is the (x, y) gantry action to apply this step: a
    real pair while still approaching or frozen, or None if control
    should be released to the PPO controller entirely (only possible for
    a "*_release" controller, and only once reached).

    reached_now is True from the first step the target is within
    tolerance onward. Callers pass the previous call's `reached_before`
    so a "*_release" controller can keep returning None even if the
    PPO's own subsequent motion drags current_qpos back out of
    tolerance (distinguishing "arrived at some point this phase" from
    "currently within tolerance this exact step").
    """
    if controller not in CONTROLLER_NAMES:
        raise ValueError(f"Unknown controller '{controller}'. Choose from: {CONTROLLER_NAMES}")

    approach, post = controller.split("_", 1)

    ex = target_qpos[0] - current_qpos[0]
    ey = target_qpos[1] - current_qpos[1]
    within_tolerance_now = (ex * ex + ey * ey) ** 0.5 <= tolerance

    if reached_before or within_tolerance_now:
        if post == "release":
            return None, True
        return (0.0, 0.0), True  # "freeze"

    if approach == "bangbang":
        action = (_clip(ex / gantry_step), _clip(ey / gantry_step))
    else:  # "proportional"
        action = (_clip(ex * PROPORTIONAL_GAIN), _clip(ey * PROPORTIONAL_GAIN))
    return action, False
