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

from mujoco_env_comp import KukaTennisEnv
from nash_skills.skills import SKILL_PROFILE_NAMES, get_skill, SKILL_NAMES, N_SKILLS, skill_index
from nash_skills.defender import DEFENDER_NAMES, defender_target


class SkillEnv:
    """
    Wraps KukaTennisEnv to support 5 named skills.

    The inner env uses:
        env.update_target_racket_pose(y_target=side_target * 0.38)
        env.update_target_racket_pose_opp(y_target=side_target_opp * 0.38)
    inside step().

    We override those calls at the step boundary to also supply x_target.
    """

    def __init__(self, proc_id: int = 1, history: int = 4, reset_mode: str = "clean", skill_profile: str = "current", gantry_speed_scale: float = 1.0):
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
        self._apply_skills()

    # ------------------------------------------------------------------ #
    # Skill API                                                            #
    # ------------------------------------------------------------------ #

    @property
    def defender_positions(self):
        """(pos1, pos2) by name, or (None, None) before set_defenders."""
        return getattr(self, "_pos1", None), getattr(self, "_pos2", None)

    def set_skills(self, skill1: str, skill2: str):
        """Call before each rally to set the skill for each player."""
        if skill1 not in SKILL_NAMES:
            raise ValueError(f"Unknown skill '{skill1}'. Choose from: {SKILL_NAMES}")
        if skill2 not in SKILL_NAMES:
            raise ValueError(f"Unknown skill '{skill2}'. Choose from: {SKILL_NAMES}")
        self._skill1 = skill1
        self._skill2 = skill2
        self._apply_skills()

    def set_defenders(self, pos1=None, pos2=None):
        """Defender position per player, by name, or None to leave disabled.

        Separate from set_skills because the two are chosen together but used in
        different phases: the skill when you hit, the position while you wait.
        """
        for p in (pos1, pos2):
            if p is not None and p not in DEFENDER_NAMES:
                raise ValueError(f"Unknown defender position '{p}'. Choose from: {DEFENDER_NAMES}")
        # Keep the names, not just the coordinates.  The preposition is a
        # strategic choice and belongs in the high-level state, but it never
        # reaches the observation -- _encode_obs_skills writes only the two skill
        # scalars, and set_defender_targets takes coordinates -- so without this
        # the only trace of it is where the gantry has got to, which lags the
        # commitment by however long the 8.0-gain controller takes to arrive.
        self._pos1, self._pos2 = pos1, pos2
        self._env.set_defender_targets(
            target1=defender_target(pos1) if pos1 else None,
            target2=defender_target(pos2) if pos2 else None,
        )

    # Lateral sign convention for the five skills.
    #
    #   "legacy"  nash_skills.skills as written: `left` lands at -world-y.
    #   "v3"      Kexin's v3.skill_semantics convention: +y is the acting
    #             player's own physical left, so `left` lands at +world-y.
    #             This is the sign the defender positions already use, so
    #             under "v3" a shot named left and a position named def_left
    #             refer to the same half of the table.
    #
    # Only the sign changes here.  Depths come from nash_skills.skills, which is
    # now the shared upstream file (short 1.90, side 1.70, mid 2.0) -- the
    # env-var overrides this switch used to pair with are gone, so a depth sweep
    # has to edit that file or go through v3.  The action set {+-SIDE} x {short,
    # far} is symmetric, so flipping the sign relabels the five skills without
    # changing which five physical shots they are.
    # Default stays "legacy" deliberately.  This class is on the shared path --
    # nash_skills/v2's collectors and trainers all construct SkillEnv -- so
    # changing the default would silently flip the physical side every skill name
    # refers to for everyone who pulls, which is the class of divergence this
    # switch exists to make explicit rather than to introduce.  Opt in per run.
    _SKILL_SEMANTICS = os.environ.get("PINGPONG_SKILL_SEMANTICS", "legacy")

    @classmethod
    def _get_skill(cls, name, profile):
        side, depth = get_skill(name, profile=profile)
        if cls._SKILL_SEMANTICS == "v3":
            return -side, depth
        if cls._SKILL_SEMANTICS != "legacy":
            raise ValueError(
                f"PINGPONG_SKILL_SEMANTICS must be 'legacy' or 'v3', "
                f"got {cls._SKILL_SEMANTICS!r}")
        return side, depth

    def _apply_skills(self):
        side1, x1 = self._get_skill(self._skill1, self.skill_profile)
        side2, x2 = self._get_skill(self._skill2, self.skill_profile)
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

    # ------------------------------------------------------------------ #
    # Gym-style interface (delegates to inner env)                         #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None):
        obs, info = self._env.reset(seed=seed)
        self._apply_skills()   # re-sync after reset randomises side_target
        return self._encode_obs_skills(obs), info

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

        obs, reward, done, truncated, info = result
        return self._encode_obs_skills(obs), reward, done, truncated, info

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
