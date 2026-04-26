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
from nash_skills.skills import get_skill, SKILL_NAMES


class SkillEnv:
    """
    Wraps KukaTennisEnv to support 5 named skills.

    The inner env uses:
        env.update_target_racket_pose(y_target=side_target * 0.38)
        env.update_target_racket_pose_opp(y_target=side_target_opp * 0.38)
    inside step().

    We override those calls at the step boundary to also supply x_target.
    """

    def __init__(self, proc_id: int = 1, history: int = 4):
        self._env = KukaTennisEnv(proc_id=proc_id, history=history)
        self._skill1 = "left"        # default skill for ego player
        self._skill2 = "right"       # default skill for opponent
        self._x_target1 = None
        self._x_target2 = None
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

    def _apply_skills(self):
        side1, x1 = get_skill(self._skill1)
        side2, x2 = get_skill(self._skill2)
        self._env.side_target     = side1
        self._env.side_target_opp = side2
        self._x_target1 = x1
        self._x_target2 = x2

    # ------------------------------------------------------------------ #
    # Gym-style interface (delegates to inner env)                         #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None):
        obs, info = self._env.reset(seed=seed)
        self._apply_skills()   # re-sync after reset randomises side_target
        # Patch obs[-2:] so the returned observation reflects the current skill,
        # not the stale randomised side_target written by reset_ball_throw().
        obs = obs.copy()
        side1, _ = get_skill(self._skill1)
        side2, _ = get_skill(self._skill2)
        obs[-2] = side1
        obs[-1] = side2
        return obs, info

    def step(self, action):
        # Sync skill targets into the inner env before it runs
        self._apply_skills()

        # Temporarily monkey-patch the pose-update methods so x_target
        # uses the skill value instead of the hard-coded default.
        original_update     = self._env.update_target_racket_pose
        original_update_opp = self._env.update_target_racket_pose_opp

        x1 = self._x_target1
        x2 = self._x_target2

        def patched_update(**kwargs):
            kwargs.setdefault("x_target", x1)
            # y_target is supplied by step() as side_target * 0.38
            original_update(**kwargs)

        def patched_update_opp(**kwargs):
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

        return result

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
