"""
Soft-bias defend-steering: adds a gentle proportional pull toward the
chosen defend target to the PPO's own gantry command, instead of
REPLACING it.

The hard-override mechanism (nash_skills.defend_steering +
defend_engagement, bangbang_release -- the confirmed-best of 4 tested
controllers in an earlier A/B) was found net-harmful across all 12
cells of a 1200-rally matched-pair experiment
(scripts/mechanism_validation_matched.py): avg_delta_tested_wr=-0.243,
avg_delta_contacts=-2.065, with harm persisting even when the override
released control back to the PPO well before the rally ended 94%+ of
the time. That rules out "release happens too late" as the explanation
-- the leading hypothesis is that the PPO was trained under its own
continuous, reactive gantry-tracking policy and never on recovering
from being artificially placed at one of 3 fixed discrete targets, so
a hard override pushes it out of its training distribution in a way
release alone can't undo.

This module tests a structurally different design instead of tuning
the hard override's parameters further:

    final_command = clip(ppo_command + alpha * Kp * (target - current), -1, 1)

alpha in [0, 1] controls how strongly the defend bias is blended in.
alpha=0 recovers pure PPO control exactly -- a built-in sanity check
for the alpha sweep this feeds into (scripts/soft_bias_sweep.py). Kp
reuses the gain already validated in the earlier "proportional"
approach law (nash_skills.defend_controllers.PROPORTIONAL_GAIN, 8.0)
rather than introducing a second new free parameter to tune at the same
time as alpha.

There is deliberately no release/cutoff logic here (unlike
nash_skills.defend_engagement) -- at the alpha values being swept
(<=0.5), the bias is gentle by construction and naturally shrinks
toward zero as the gantry approaches target, rather than needing an
explicit stop condition. Applied continuously for as long as a player
remains the defender (nash_skills.defend_role.DefenderRoleTracker,
unchanged, still the sole source of role determination).

Pure Python, no MuJoCo dependency -- fully unit tested
(tests/test_defend_soft_bias.py). The live sweep harness lives in
scripts/soft_bias_sweep.py.
"""
from __future__ import annotations

__all__ = ["blend_gantry_command", "DEFAULT_KP"]

DEFAULT_KP = 8.0  # matches nash_skills.defend_controllers.PROPORTIONAL_GAIN


def _clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def blend_gantry_command(ppo_command, current_qpos, target_qpos, alpha: float, kp: float = DEFAULT_KP):
    """
    Returns (final_command, intervention_magnitude).

    ppo_command : (x, y) -- the PPO's own, unmodified gantry action for
                  this step
    current_qpos, target_qpos : (x, y) each
    alpha : blend strength, 0 (pure PPO) to 1 (bias as strong as the
            raw proportional term itself)
    kp : proportional gain on the defend-position error

    intervention_magnitude is the Euclidean distance between the final
    blended command and the original PPO command -- the ACTUAL
    deviation actually applied (computed post-clip, not from the raw
    unclipped bias) -- "how hard did this override actually push,"
    tracked so the alpha sweep can report exactly how much intervention
    starts to hurt.
    """
    bias_x = alpha * kp * (target_qpos[0] - current_qpos[0])
    bias_y = alpha * kp * (target_qpos[1] - current_qpos[1])

    final_x = _clip(ppo_command[0] + bias_x)
    final_y = _clip(ppo_command[1] + bias_y)

    intervention_magnitude = (
        (final_x - ppo_command[0]) ** 2 + (final_y - ppo_command[1]) ** 2
    ) ** 0.5
    return (final_x, final_y), intervention_magnitude
