"""
When a Design B strategic decision happens, and who is the attacker at it.

  crossing  legacy: when SkillEnv's role tracker changes (the ball crosses the
            net toward the other player). Both players re-pick a full
            (attack, preparation) pair; the defender's new preparation takes
            over halfway through its preparation window.
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

    def reset(self):
        self._prep.reset()
        self._prev = None

    def update(self, env):
        if self.timing == "contact":
            preparing = self._prep.update(float(env._env.data.qvel[-6]))
            current = None if preparing is None else 3 - preparing
        else:
            current = env._role_tracker.defending_player
        fired = current is not None and current != self._prev
        self._prev = current
        return current if fired else None
