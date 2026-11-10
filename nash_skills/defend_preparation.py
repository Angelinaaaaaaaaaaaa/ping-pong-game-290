"""
Paper-faithful defend mechanism for Design B: preparation by the player who
just hit (L4DC draft Sec. 3.2).

The paper's defender is the player who produced the preceding shot. It moves
toward its preparation position "during the opponent's return phase", without
seeing the attacker's new target; the attacker is the receiver. The reactive
mechanism (defend_reactive_pose.py) instead biases the RECEIVER's approach,
after the incoming ball is already observable, so it models a different role.

Window: the player the ball is moving away from, i.e. from that player's own
hit until the opponent's hit. In this window KukaTennisEnv's own pose update
returns early (ball not incoming), so the racket target is frozen at its last
interception pose and nothing tracks the ball. Commanding a preparation pose
here therefore cannot blind the low-level tracker. Static-pose redirection
failed when applied during the incoming phase ("Option B"); here it is not.
Release is automatic: once the opponent's return is incoming, the normal
interception update takes over with the full flight time left.

Baseline trace (20 rallies, 141 hits, scratchpad trace_preparation_window.py):
the direction flip lands within one step of every recorded hit (MuJoCo's soft
contact starts reversing the ball one step before the env logs the contact),
with no flips away from hits. The window lasts a median 62 steps, during which
the hitter's racket moves a median 1.1 cm.

Poses: defend_target_pose()'s calibrated left/center/right racket poses.
"""
from nash_skills.defend_role import DEFAULT_VEL_THRESHOLD

__all__ = ["PreparationRoleTracker"]


class PreparationRoleTracker:
    """Which player is preparing: the one the ball is moving away from.

    Positive world x-velocity moves the ball toward player 2, so player 1 is
    preparing. Inside the dead-band (|vel_x| <= vel_threshold, e.g. the contact
    transient) the previous answer is kept.
    """

    def __init__(self, vel_threshold: float = DEFAULT_VEL_THRESHOLD):
        self.vel_threshold = vel_threshold
        self.preparing_player = None

    def reset(self):
        self.preparing_player = None

    def update(self, ball_vel_x: float):
        if ball_vel_x > self.vel_threshold:
            self.preparing_player = 1
        elif ball_vel_x < -self.vel_threshold:
            self.preparing_player = 2
        return self.preparing_player
