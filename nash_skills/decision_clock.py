"""
When a Design B strategic decision happens, and who is the attacker at it.

  crossing  Kexin-compatible net-plane crossing: initialize from reset ball
            position, then require strictly opposite sides on successive steps.
            An endpoint exactly on the plane does not trigger. No initial latch.
  contact   paper Sec. 3.2: an exchange starts at a racket contact. The
            receiver of the new ball is the attacker (its attack target aims
            its next return); the player who just hit is the defender (its
            preparation position holds until the receiver's contact). Each
            choice is in force for exactly one whole exchange. Detected the
            way the preparation mechanism detects it: the sign of the ball's
            x-velocity, with a dead-band so the contact transient does not
            fire twice (nash_skills.defend_preparation).

DecisionClock.update(env) returns the attacker (1 or 2) when a decision fires
this step, else None. Call it once per env step, after the step.
"""
from nash_skills.defend_preparation import PreparationRoleTracker
from nash_skills.skills import TABLE_SHIFT

__all__ = ["DECISION_TIMINGS", "DEFAULT_DECISION_TIMING", "DecisionClock", "live_halves",
           "validate_decision_timing"]

DECISION_TIMINGS = ("crossing", "contact")
DEFAULT_DECISION_TIMING = "crossing"


def validate_decision_timing(timing: str) -> str:
    if timing not in DECISION_TIMINGS:
        raise ValueError(f"Unknown decision timing '{timing}'. Choose from: {DECISION_TIMINGS}")
    return timing


def live_halves(attacker: int) -> dict:
    """Which half of each player's (attack, preparation) choice acts in this exchange."""
    return {attacker: "attack", 3 - attacker: "defend"}


class DecisionClock:
    def __init__(self, timing: str):
        self.timing = validate_decision_timing(timing)
        self._prep = PreparationRoleTracker()
        self._prev = None
        self._cross_x = None

    def reset(self, env=None, obs=None):
        self._prep.reset()
        self._prev = None
        self._cross_x = None
        if self.timing == "crossing" and env is not None:
            self._cross_x = float(obs[36]) if obs is not None else float(env._env.data.qpos[-7])

    def update(self, env, obs=None):
        if self.timing == "contact":
            preparing = self._prep.update(float(env._env.data.qvel[-6]))
            current = None if preparing is None else 3 - preparing
        else:
            current_x = float(obs[36]) if obs is not None else float(env._env.data.qpos[-7])
            previous_x = self._cross_x
            self._cross_x = current_x
            if previous_x is None or (previous_x - TABLE_SHIFT) * (current_x - TABLE_SHIFT) >= 0:
                return None
            return 2 if current_x > TABLE_SHIFT else 1
        fired = current is not None and current != self._prev
        self._prev = current
        return current if fired else None
