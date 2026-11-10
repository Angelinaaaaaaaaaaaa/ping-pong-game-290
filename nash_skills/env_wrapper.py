"""
Thin wrapper around KukaTennisEnv that adds 5-skill support.

Usage:
    from nash_skills.env_wrapper import SkillEnv
    env = SkillEnv()
    env.set_skills("left_short", "right")   # set before each rally
    obs, info = env.reset()
    ...

The wrapper intercepts the two existing scalar attributes
(side_target, side_target_opp) and syncs them from the chosen skill,
then also passes x_target to the pose-update methods via monkey-patching
the call site that happens inside env.step().
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from mujoco_env_comp import KukaTennisEnv
from nash_skills.skills import SKILL_PROFILE_NAMES, get_skill, SKILL_NAMES, N_SKILLS, skill_index, TABLE_SHIFT
from nash_skills.defend_positions import DEFEND_POSITION_NAMES
from nash_skills.defend_steering import compute_defend_steering_action
from nash_skills.defend_role import DefenderRoleTracker
from nash_skills.defend_engagement import DefendEngagementTracker
from nash_skills.defend_target_pose import defend_target_pose, DefendPoseEngagementTracker
from nash_skills.prep_pose_tables import CALIBRATED_PREP_POSES, PREP_ORIENTATIONS, calibrated_prep_pose
from nash_skills.defend_reactive_pose import reactive_defend_pose
from nash_skills.defend_preparation import PreparationRoleTracker
from nash_skills.rally_status import rally_status


class SkillEnv:
    """
    Wraps KukaTennisEnv to support 5 named skills.

    The inner env uses:
        env.update_target_racket_pose(y_target=side_target * 0.38)
        env.update_target_racket_pose_opp(y_target=side_target_opp * 0.38)
    inside step().

    We override those calls at the step boundary to also supply x_target.
    """

    def __init__(self, proc_id: int = 1, history: int = 4, reset_mode: str = "clean", skill_profile: str = "current", gantry_speed_scale: float = 1.0, reset_targets: bool = False):
        if skill_profile not in SKILL_PROFILE_NAMES:
            raise ValueError(f"Unknown skill_profile '{skill_profile}'. Choose from: {SKILL_PROFILE_NAMES}")
        self._env = KukaTennisEnv(
            proc_id=proc_id,
            history=history,
            reset_mode=reset_mode,
            gantry_speed_scale=gantry_speed_scale,
        )
        self.skill_profile = skill_profile
        self._skill1 = "left"        # default skill for ego player
        self._skill2 = "right"       # default skill for opponent
        self._x_target1 = None
        self._x_target2 = None
        self._defend1 = None         # opt-in: defend-position name, or None (inactive)
        self._defend2 = None
        self._defend_target1 = None  # opt-in: Option B (target-pose mechanism) --
        self._defend_target2 = None  # independent of _defend1/_defend2; do not set both
        self._defend_reactive1 = None  # opt-in: velocity/trajectory-informed pre-
        self._defend_reactive2 = None  # position -- see nash_skills.defend_reactive_pose;
                                        # independent of _defend1/2 and _defend_target1/2;
                                        # do not set more than one mechanism for the same rally
        self._defend_prep1 = None   # opt-in: paper-faithful preparation by the player
        self._defend_prep2 = None   # who just hit (nash_skills.defend_preparation)
        self._prep_calibrated = False
        self._prep_orientation = "fixed"
        self._role_tracker = DefenderRoleTracker()
        self._prep_tracker = PreparationRoleTracker()
        # KukaTennisEnv.reset() never resets the racket targets (its
        # reset_target() call is commented out), so each rally starts from the
        # previous rally's last target and same-seed replays diverge. Opt-in:
        # restore the construction-time targets before every reset.
        self._reset_targets = reset_targets
        if reset_targets:
            self._home_target = np.array(self._env.curr_target, dtype=float).copy()
            self._home_target_opp = np.array(self._env.curr_target_opp, dtype=float).copy()
        self._engagement = DefendEngagementTracker()
        self._pose_engagement = DefendPoseEngagementTracker()
        self._apply_skills()

    # ------------------------------------------------------------------ #
    # Skill API                                                            #
    # ------------------------------------------------------------------ #

    def set_skills(self, skill1: str, skill2: str):
        """Call before each rally to set the skill for each player."""
        if skill1 not in SKILL_NAMES:
            raise ValueError(f"Unknown skill '{skill1}'. Choose from: {SKILL_NAMES}")
        if skill2 not in SKILL_NAMES:
            raise ValueError(f"Unknown skill '{skill2}'. Choose from: {SKILL_NAMES}")
        self._skill1 = skill1
        self._skill2 = skill2
        self._apply_skills()

    def set_defend_positions(self, pos1, pos2):
        """
        Opt-in, Design B attacker/defender extension: set each player's
        defend-position choice (used whenever the ball is heading toward
        them; their attack skill from set_skills() is used whenever it
        isn't). Pass None for a player to leave them with no defend
        position set -- if neither player ever has one set, step() applies
        no steering at all and behaves exactly as before this feature
        existed. Call per rally, alongside set_skills().
        """
        for pos in (pos1, pos2):
            if pos is not None and pos not in DEFEND_POSITION_NAMES:
                raise ValueError(
                    f"Unknown defend position '{pos}'. Choose from: {DEFEND_POSITION_NAMES}"
                )
        self._defend1 = pos1
        self._defend2 = pos2

    def set_defend_target_positions(self, pos1, pos2):
        """
        Opt-in, Option B for the attacker/defender extension: like
        set_defend_positions, but the defend position is expressed as a
        racket-POSE target (nash_skills.defend_target_pose) fed through
        the SAME already-trained target-tracking interface every attack
        skill already uses, instead of overriding the gantry ACTION
        directly. The PPO's own policy decides the action in response;
        only its input target changes. Independent of
        set_defend_positions -- the two mechanisms are alternatives for
        the same idea; do not set both for the same rally.
        """
        for pos in (pos1, pos2):
            if pos is not None and pos not in DEFEND_POSITION_NAMES:
                raise ValueError(
                    f"Unknown defend position '{pos}'. Choose from: {DEFEND_POSITION_NAMES}"
                )
        self._defend_target1 = pos1
        self._defend_target2 = pos2

    def set_defend_reactive_positions(self, pos1, pos2):
        """
        Opt-in, velocity/trajectory-informed defend-position mechanism
        (nash_skills.defend_reactive_pose) -- the fix for Option B
        (set_defend_target_positions)'s root cause. Unlike Option B's
        static, ball-independent pose, this recomputes a real ball-
        trajectory interception prediction every step (the same physics
        update_target_racket_pose_opp already uses) and adds a lateral
        bias toward the named position that fades to zero as the ball
        approaches -- so there is no frozen target and no discontinuity
        at a handoff; the target was never ball-independent. Independent
        of set_defend_positions / set_defend_target_positions -- the
        three mechanisms are alternatives for the same idea; do not set
        more than one for the same rally.
        """
        for pos in (pos1, pos2):
            if pos is not None and pos not in DEFEND_POSITION_NAMES:
                raise ValueError(
                    f"Unknown defend position '{pos}'. Choose from: {DEFEND_POSITION_NAMES}"
                )
        self._defend_reactive1 = pos1
        self._defend_reactive2 = pos2

    def set_defend_preparation_positions(self, pos1, pos2, calibrated: bool = False, orientation: str = "fixed"):
        """
        Opt-in, paper-faithful defend mechanism (nash_skills.defend_preparation):
        while the ball travels away from a player (from its own hit until the
        opponent's), that player's racket target is its calibrated defend pose;
        once the opponent's return is incoming, the normal interception update
        runs unchanged. The receiver's attack skill keeps steering its return.
        Alternative to the other defend mechanisms; do not set more than one
        for the same rally.

        calibrated=False uses defend_target_pose's legacy table; True uses this
        env's skill profile's table from nash_skills.prep_pose_tables.
        orientation="calibrated" (needs calibrated=True) also uses the calibrated
        racket orientation per position instead of the fixed face-forward one.
        """
        for pos in (pos1, pos2):
            if pos is not None and pos not in DEFEND_POSITION_NAMES:
                raise ValueError(
                    f"Unknown defend position '{pos}'. Choose from: {DEFEND_POSITION_NAMES}"
                )
        if calibrated and self.skill_profile not in CALIBRATED_PREP_POSES:
            raise ValueError(f"No calibrated preparation poses for skill profile '{self.skill_profile}'. "
                             f"Calibrated: {sorted(CALIBRATED_PREP_POSES)}")
        if orientation not in PREP_ORIENTATIONS or (orientation == "calibrated" and not calibrated):
            raise ValueError(f"orientation must be one of {PREP_ORIENTATIONS}; 'calibrated' needs calibrated=True")
        self._defend_prep1 = pos1
        self._defend_prep2 = pos2
        self._prep_calibrated = calibrated
        self._prep_orientation = orientation

    def _prep_pose(self, name):
        if self._prep_calibrated:
            return calibrated_prep_pose(name, self.skill_profile, orientation=self._prep_orientation)
        return defend_target_pose(name)

    def _reactive_pose_for_player(self, player: int, defend_name: str):
        """
        This step's reactive defend pose for `player`, or None if the
        ported physics' guard says no sensible target can be computed
        right now (caller should leave the previous target unchanged,
        same as the real update_target_racket_pose(_opp) does).

        Applies the same world-frame -> player-1-equivalent flip
        update_target_racket_pose_opp applies internally for player 2
        (negate y, mirror x as 2*TABLE_SHIFT - x, negate vel x/y) before
        calling the frame-agnostic ported physics -- the result is
        passed to set_target_pose_opp, which mirrors back for player 2's
        actual physical application, exactly like defend_target_pose()
        already does for Option B.
        """
        ball_pos = np.asarray(self._env.data.body('ball').xpos, dtype=float).copy()
        ball_vel = np.asarray(self._env.data.qvel[-6:-3], dtype=float).copy()
        if player == 2:
            ball_pos[1] = -ball_pos[1]
            ball_pos[0] = 2 * TABLE_SHIFT - ball_pos[0]
            ball_vel[0] = -ball_vel[0]
            ball_vel[1] = -ball_vel[1]
            x_target = self._x_target2
            y_target = self._env.side_target_opp * 0.38
        else:
            x_target = self._x_target1
            y_target = self._env.side_target * 0.38
        return reactive_defend_pose(ball_pos, ball_vel, defend_name, x_target, y_target)

    def _apply_skills(self):
        side1, x1 = get_skill(self._skill1, profile=self.skill_profile)
        side2, x2 = get_skill(self._skill2, profile=self.skill_profile)
        self._env.side_target     = side1
        self._env.side_target_opp = side2
        self._x_target1 = x1
        self._x_target2 = x2

    def _skill_value(self, skill: str) -> float:
        return skill_index(skill) / (N_SKILLS - 1)

    def _encode_obs_skills(self, obs):
        obs = obs.copy()
        obs[-2] = self._skill_value(self._skill1)
        obs[-1] = self._skill_value(self._skill2)
        return obs

    def _with_ball_status(self, info):
        """info plus the parts of the paper's x_t the raw observation lacks: the ball's angular
        velocity (MuJoCo free-joint qvel[-3:]) and the rally status (nash_skills.rally_status).
        Unchanged for an inner env without a simulator or rally tracker (test doubles); the v3
        state encoder then refuses the info with a KeyError."""
        data, rally = getattr(self._env, "data", None), getattr(self._env, "rally_state", None)
        if data is None or rally is None:
            return info
        return {**info, "ball_angvel": np.array(data.qvel[-3:], dtype=float), "rally_status": rally_status(rally)}

    # ------------------------------------------------------------------ #
    # Gym-style interface (delegates to inner env)                         #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None):
        if self._reset_targets:
            self._env.set_target_pose(self._home_target.copy())
            self._env.set_target_pose_opp(self._home_target_opp.copy())
        obs, info = self._env.reset(seed=seed)
        self._apply_skills()   # re-sync after reset randomises side_target
        self._role_tracker.reset()  # new rally -- clear the latched defender
        self._engagement.reset()    # new rally -- clear pre-positioning latches
        self._pose_engagement.reset()  # new rally -- clear Option B's latches too
        self._prep_tracker.reset()
        return self._encode_obs_skills(obs), self._with_ball_status(info)

    def step(self, action):
        # Sync skill targets into the inner env before it runs
        self._apply_skills()

        # Opt-in defend-position steering: override the gantry components
        # of `action` for whichever player is currently defending (ball
        # heading toward them), if they have a defend position set. No-op
        # entirely if set_defend_positions() was never called.
        if self._defend1 is not None or self._defend2 is not None:
            action = self._apply_defend_steering(action)

        # Opt-in Option B: decide, BEFORE the pose-update methods run this
        # step, whether either player should be fed a defend racket-pose
        # target instead of their normal skill-based one. No-op entirely
        # if set_defend_target_positions() was never called.
        use_defend_target1 = use_defend_target2 = False
        use_defend_reactive1 = use_defend_reactive2 = False
        use_defend_prep1 = use_defend_prep2 = False
        any_target_mode = self._defend_target1 is not None or self._defend_target2 is not None
        any_reactive_mode = self._defend_reactive1 is not None or self._defend_reactive2 is not None
        any_prep_mode = self._defend_prep1 is not None or self._defend_prep2 is not None
        if any_target_mode or any_reactive_mode or any_prep_mode:
            ball_x = float(self._env.data.body('ball').xpos[0])
            ball_vel_x = float(self._env.data.qvel[-6])
            # Still updated in preparation mode: collectors and evaluators
            # detect their decision points from this tracker's net crossings.
            defending_player = self._role_tracker.update(ball_x, ball_vel_x)

            if any_prep_mode:
                preparing_player = self._prep_tracker.update(ball_vel_x)
                use_defend_prep1 = preparing_player == 1 and self._defend_prep1 is not None
                use_defend_prep2 = preparing_player == 2 and self._defend_prep2 is not None

            if any_target_mode:
                # Racket end-effector position, not gantry qpos -- must
                # match the SAME space defend_target_pose()'s target is
                # expressed in, since that's what should_steer compares
                # it against. See defend_target_pose.py's
                # DefendPoseEngagementTracker docstring: checking qpos
                # against an unrelated gantry-qpos target meant "done"
                # almost never latched, so control never handed back to
                # ball-reactive tracking at all.
                racket_pos1 = self._env.data.body('tennis_racket').xpos
                racket_pos2 = self._env.data.body('tennis_racket_opp').xpos
                use_defend_target1 = self._pose_engagement.should_steer(
                    1, defending_player, racket_pos1, self._defend_target1)
                use_defend_target2 = self._pose_engagement.should_steer(
                    2, defending_player, racket_pos2, self._defend_target2)

            if any_reactive_mode:
                # No engagement latch needed: the target is recomputed
                # from the real ball trajectory every step (never
                # frozen), so there is nothing to "finish" and hand off
                # from -- the mechanism simply stops applying the moment
                # the role tracker says this player is no longer
                # defending, and patched_update(_opp) falls through to
                # the normal attack-skill-driven call below.
                use_defend_reactive1 = defending_player == 1 and self._defend_reactive1 is not None
                use_defend_reactive2 = defending_player == 2 and self._defend_reactive2 is not None

        # Temporarily monkey-patch the pose-update methods so x_target
        # uses the skill value instead of the hard-coded default.
        original_update     = self._env.update_target_racket_pose
        original_update_opp = self._env.update_target_racket_pose_opp

        x1 = self._x_target1
        x2 = self._x_target2

        def patched_update(**kwargs):
            if use_defend_prep1:
                self._env.set_target_pose(self._prep_pose(self._defend_prep1))
                return
            if use_defend_target1:
                self._env.set_target_pose(defend_target_pose(self._defend_target1))
                return
            if use_defend_reactive1:
                pose = self._reactive_pose_for_player(1, self._defend_reactive1)
                if pose is not None:
                    self._env.set_target_pose(pose)
                return
            kwargs.setdefault("x_target", x1)
            # y_target is supplied by step() as side_target * 0.38
            original_update(**kwargs)

        def patched_update_opp(**kwargs):
            if use_defend_prep2:
                self._env.set_target_pose_opp(self._prep_pose(self._defend_prep2))
                return
            if use_defend_target2:
                self._env.set_target_pose_opp(defend_target_pose(self._defend_target2))
                return
            if use_defend_reactive2:
                pose = self._reactive_pose_for_player(2, self._defend_reactive2)
                if pose is not None:
                    self._env.set_target_pose_opp(pose)
                return
            kwargs.setdefault("x_target", x2)
            original_update_opp(**kwargs)

        self._env.update_target_racket_pose     = patched_update
        self._env.update_target_racket_pose_opp = patched_update_opp

        try:
            result = self._env.step(action)
        finally:
            # Always restore originals, even if step() raises an exception
            self._env.update_target_racket_pose     = original_update
            self._env.update_target_racket_pose_opp = original_update_opp

        obs, reward, done, truncated, info = result
        return self._encode_obs_skills(obs), reward, done, truncated, self._with_ball_status(info)

    def _apply_defend_steering(self, action):
        """Return a copy of `action` with the defending player's gantry
        components overridden to steer toward their defend position --
        but ONLY until that target is reached. defend_position is a
        pre-positioning action, not a continuous hold: once the target is
        reached for the current defending phase, self._engagement latches
        that player "done" and gantry control releases back to the PPO
        controller for the rest of the phase (even if the PPO's own
        motion later drags the gantry away from the target again), so the
        low-level controller keeps free control for the actual return
        stroke. See nash_skills.defend_engagement for why this exists --
        without it, the override stayed active through the return swing
        itself and suppressed the defender's return ability almost
        entirely, independent of which defend position was chosen.

        Who is defending is resolved by self._role_tracker (latch on a
        clear velocity reading, flip only on an actual net crossing) --
        not by reading ball_vel_x's sign directly here. See
        nash_skills.defend_role for why."""
        gantry_step = self._env.gantry_speed_scale * 10.0 / 1000.0
        ball_x = float(self._env.data.body('ball').xpos[0])
        ball_vel_x = float(self._env.data.qvel[-6])
        qpos1 = (float(self._env.data.qpos[0]), float(self._env.data.qpos[1]))
        qpos2 = (float(self._env.data.qpos[9]), float(self._env.data.qpos[10]))

        defending_player = self._role_tracker.update(ball_x, ball_vel_x)

        steer1 = self._engagement.should_steer(1, defending_player, qpos1, self._defend1)
        steer2 = self._engagement.should_steer(2, defending_player, qpos2, self._defend2)
        effective_defend1 = self._defend1 if steer1 else None
        effective_defend2 = self._defend2 if steer2 else None

        a1, a2 = compute_defend_steering_action(
            defending_player, qpos1, qpos2, effective_defend1, effective_defend2, gantry_step,
        )

        action = action.copy()
        if a1 is not None:
            action[0], action[1] = a1
        if a2 is not None:
            action[9], action[10] = a2
        return action

    def render(self, mode="human"):
        return self._env.render(mode)

    def close(self):
        return self._env.close()

    # ------------------------------------------------------------------ #
    # Expose inner env attributes transparently                            #
    # ------------------------------------------------------------------ #

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def side_target(self):
        return self._env.side_target

    @side_target.setter
    def side_target(self, v):
        self._env.side_target = v

    @property
    def side_target_opp(self):
        return self._env.side_target_opp

    @side_target_opp.setter
    def side_target_opp(self, v):
        self._env.side_target_opp = v

    def __getattr__(self, name):
        # Fall through to the inner env for anything else
        return getattr(self._env, name)
