"""
Explicit rally-role state machine for the attacker/defender extension
(Design B).

Replaces a naive per-step `ball_vel_x >= 0` read (fragile: right at a
racket hit, the contact solver can produce a step or two of small/noisy
velocity before the new direction is firmly established -- confirmed in
a live PPO-driven trace, e.g. ball_vel_x=-0.0805 on exactly the step a
flip happens, settling to -3.7992 ten steps later) with two separate,
more robust rules:

  - Initial latch: who defends first is decided only once the ball's
    x-velocity clearly exceeds `vel_threshold` in magnitude. Before that
    (e.g. a step or two right after reset/serve, if the ball hasn't yet
    picked up a clear velocity), no one is latched as defender yet --
    `defending_player` is None, and callers should apply no steering to
    anyone until it resolves.
  - Subsequent flips: once a defender is latched, the only thing that
    changes it is the ball's x-position actually crossing the net plane
    (TABLE_SHIFT) -- not a change in velocity sign. A sign change alone
    (e.g. right at a hit, before the ball has traveled back across the
    net) does NOT flip the role. This trades a few steps of reaction
    time for a large hysteresis margin around the noisy moment of
    contact; live traces show the gantry reaches its target well before
    the ball arrives, so that margin is not reaction-time-critical.

Pure Python, no MuJoCo dependency -- fully unit tested
(tests/test_defend_role.py). The live wiring that feeds real ball_x/
ball_vel_x into this each step lives in nash_skills/env_wrapper.py.
"""

from nash_skills.skills import TABLE_SHIFT

__all__ = ["DefenderRoleTracker", "DEFAULT_VEL_THRESHOLD"]

# Live trace (real PPO-driven hitting) observed ~0.08 velocity magnitude
# right at a hit's contact transient, and ~3.5-4.6 at cruising speed
# between hits. 0.5 sits comfortably clear of the former and well below
# the latter.
DEFAULT_VEL_THRESHOLD = 0.5


class DefenderRoleTracker:
    """
    Stateful across a single rally: call reset() at the start of each
    rally, then update(ball_x, ball_vel_x) once per step to get the
    current defending player (1, 2, or None if not yet latched).
    """

    def __init__(self, net_x: float = TABLE_SHIFT, vel_threshold: float = DEFAULT_VEL_THRESHOLD):
        self.net_x = net_x
        self.vel_threshold = vel_threshold
        self.defending_player = None  # None, 1, or 2
        self._prev_side = None        # None, "p1", or "p2"

    def reset(self):
        """Call at the start of each new rally -- clears the latch so a
        fresh ambiguous reading can't resolve to the previous rally's
        defender."""
        self.defending_player = None
        self._prev_side = None

    def update(self, ball_x: float, ball_vel_x: float):
        side = "p2" if ball_x >= self.net_x else "p1"

        if self.defending_player is None:
            if ball_vel_x > self.vel_threshold:
                self.defending_player = 2
            elif ball_vel_x < -self.vel_threshold:
                self.defending_player = 1
            # else: still ambiguous -- stay unlatched this step
        elif self._prev_side is not None and side != self._prev_side:
            self.defending_player = 2 if side == "p2" else 1

        self._prev_side = side
        return self.defending_player
