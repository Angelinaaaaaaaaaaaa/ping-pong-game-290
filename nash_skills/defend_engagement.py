"""
Defend-position engagement latch for the attacker/defender extension
(Design B).

A live 6-combination experiment (attack={left,right} x defend={left,
center,right}, real PPO hitting, the fixed DefenderRoleTracker) came back
with the defender winning almost never in EVERY combination (opp_wr
0-1/10). A no-override control (set_defend_positions(None, None)) showed
the opposite: without any gantry override at all, the defender
essentially never loses outright (ego_wins=0/10 for both attack skills).
The defend-steering override was active for the entire defending phase --
including the moment of the actual return swing, when the low-level PPO
controller needs free gantry control to execute its learned stroke. That
is what was destroying the defender's return ability, independent of
which defend position was chosen.

Fix: defend_position is a PRE-POSITIONING action only. This module tracks,
per player, whether their defend-position steering has already reached
its target for the CURRENT defending phase:

  - While not yet reached: steer (should_steer returns True).
  - The step the target is reached (within DEFEND_REACHED_TOLERANCE):
    latch "done" and stop steering from that point on -- release full
    gantry control back to the PPO controller for the rest of this
    defending phase, even if the PPO's own arm/racket motion
    subsequently drags the gantry away from the target again while
    executing the stroke. (should_steer does NOT re-engage just because
    distance-to-target grows again within the same phase.)
  - The latch only clears when a genuinely NEW defending phase begins for
    that player -- i.e. DefenderRoleTracker reports a transition INTO
    them, which by that tracker's own construction only happens after a
    real net crossing, not a raw velocity-sign change.

DEFEND_REACHED_TOLERANCE (0.01) matches the default gantry_step
(gantry_speed_scale=1.0 * 10.0/1000.0 = 0.01). The steering law in
defend_steering._steer_toward closes any gap smaller than gantry_step
exactly on the next step once unclipped, so "within one gantry_step" is
the natural, principled definition of "arrived" -- not an arbitrary
number, and not a new ball-distance threshold.

Pure Python, no MuJoCo dependency -- fully unit tested
(tests/test_defend_engagement.py), mirroring the existing defend_role.py /
defend_steering.py pattern. The live wiring lives in
nash_skills/env_wrapper.py.
"""

from nash_skills.defend_positions import world_defend_qpos

__all__ = ["DefendEngagementTracker", "DEFEND_REACHED_TOLERANCE"]

DEFEND_REACHED_TOLERANCE = 0.01


class DefendEngagementTracker:
    """
    Stateful across a single rally: call reset() at the start of each
    rally. Call should_steer(player, defending_player, qpos, defend_name)
    once per player per step to decide whether that player's gantry
    should be overridden toward their defend position this step.
    """

    def __init__(self, tolerance: float = DEFEND_REACHED_TOLERANCE):
        self.tolerance = tolerance
        self._done = {1: False, 2: False}
        self._ever_engaged = {1: False, 2: False}
        self._prev_defending_player = None

    def reset(self):
        """Call at the start of each new rally."""
        self._done = {1: False, 2: False}
        self._ever_engaged = {1: False, 2: False}
        self._prev_defending_player = None

    def is_done(self, player: int) -> bool:
        """True if `player`'s defend-position pre-positioning has already
        completed for the current defending phase (so steering is
        currently released back to the PPO controller)."""
        return self._done.get(player, False)

    def ever_engaged(self, player: int) -> bool:
        """True if should_steer has ever returned True for `player` since
        the last reset() -- i.e. defend-steering actually activated at
        least once this rally, independent of whether it has since
        completed. Useful as a diagnostic even when defend_name was never
        set (always False then) or the player was never the defender."""
        return self._ever_engaged.get(player, False)

    def should_steer(self, player: int, defending_player, qpos, defend_name) -> bool:
        """
        Returns True if `player`'s gantry should be overridden toward
        `defend_name` this step; False otherwise (not currently
        defending, no defend position chosen, or already reached target
        this phase). Also advances this player's "new phase" bookkeeping
        and, as a side effect, latches `is_done(player)` True the step
        the target is first reached.
        """
        self._clear_done_on_new_phase(defending_player)

        if defending_player != player or defend_name is None:
            return False
        if self._done[player]:
            return False

        self._ever_engaged[player] = True

        target = world_defend_qpos(player, defend_name)
        if _distance(qpos, target) <= self.tolerance:
            self._done[player] = True
            return False

        return True

    def _clear_done_on_new_phase(self, defending_player):
        if defending_player is not None and defending_player != self._prev_defending_player:
            self._done[defending_player] = False
        self._prev_defending_player = defending_player


def _distance(qpos, target):
    (x1, y1), (x2, y2) = qpos, target
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
