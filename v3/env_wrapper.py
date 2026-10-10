"""v3.0 environment wrapper with defensive gantry pre-position support.

This module imports and reuses the current five-skill `SkillEnv` so shot skill
execution, PPO observations, target-pose calculation, reward logic, collision
handling, and terminal-state inference remain unchanged.

The only v3.0 behavior added here is an action-level gantry controller for the
defending player. It adjusts the defender's gantry action toward a committed
left/center/right ready target before delegating to the existing environment
step. It never writes qpos directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from v3.planner_env import PlannerSkillEnv
from v3.constants import (
    DEFENSE_POSITIONS,
    GANTRY_X_MAX,
    GANTRY_X_MIN,
    GANTRY_Y_MAX,
    GANTRY_Y_MIN,
    GantryTarget,
    gantry_to_world_position,
    target_for_defender,
    validate_defense_position,
    world_to_gantry_target,
)
from v3.skill_semantics import get_skill as get_v3_skill


@dataclass(frozen=True)
class DefenseCommitment:
    defender: int
    position: str
    target: GantryTarget
    qpos_target: GantryTarget


class V3PrePositionEnv(PlannerSkillEnv):
    """Current five-skill env plus defender pre-position action override."""

    def __init__(
        self,
        proc_id: int = 1,
        history: int = 4,
        reset_mode: str = "ready",
        skill_profile: str = "aggressive",
        gantry_speed_scale: float = 1.0,
        position_gain: float = 8.0,
        tolerance: float = 0.01,
        planner: str = "legacy",
    ):
        super().__init__(
            proc_id=proc_id,
            history=history,
            reset_mode=reset_mode,
            skill_profile=skill_profile,
            gantry_speed_scale=gantry_speed_scale,
            planner=planner,
        )
        self.position_gain = float(position_gain)
        self.tolerance = float(tolerance)
        self._defense_commitment: Optional[DefenseCommitment] = None

    @property
    def defense_commitment(self) -> Optional[DefenseCommitment]:
        return self._defense_commitment

    def _apply_skills(self):
        """Apply V3's acting-player-relative lateral skill convention.

        This override is intentionally limited to target selection.  The
        inherited PPO controller, planner, gantry action handling, and MuJoCo
        physics remain unchanged.
        """
        side1, x1 = get_v3_skill(self._skill1, profile=self.skill_profile)
        side2, x2 = get_v3_skill(self._skill2, profile=self.skill_profile)
        self._env.side_target = side1
        self._env.side_target_opp = side2
        self._x_target1 = x1
        self._x_target2 = x2

    def set_defense_position(self, defender: int, position: str) -> DefenseCommitment:
        validate_defense_position(position)
        target = target_for_defender(defender, position)
        qpos_target = world_to_gantry_target(defender, target)
        self._defense_commitment = DefenseCommitment(defender, position, target, qpos_target)
        return self._defense_commitment

    def clear_defense_position(self) -> None:
        self._defense_commitment = None

    def defender_gantry(self, player: int) -> np.ndarray:
        if player == 1:
            return np.asarray(self._env.data.qpos[0:2], dtype=float).copy()
        if player == 2:
            return np.asarray(self._env.data.qpos[9:11], dtype=float).copy()
        raise ValueError("player must be 1 or 2")

    def step(self, action):
        adjusted = np.asarray(action, dtype=np.float32).copy()
        if self._defense_commitment is not None:
            self._apply_defensive_gantry_action(adjusted, self._defense_commitment)
        return super().step(adjusted)

    def _apply_defensive_gantry_action(self, action: np.ndarray, commitment: DefenseCommitment) -> None:
        current = self.defender_gantry(commitment.defender)
        target = np.asarray(commitment.qpos_target.as_tuple(), dtype=float)
        error = target - current
        command = np.clip(error * self.position_gain, -1.0, 1.0)

        if abs(error[0]) <= self.tolerance:
            command[0] = 0.0
        if abs(error[1]) <= self.tolerance:
            command[1] = 0.0

        if commitment.defender == 1:
            action[0] = command[0]
            action[1] = command[1]
        elif commitment.defender == 2:
            action[9] = command[0]
            action[10] = command[1]
        else:
            raise ValueError("defender must be 1 or 2")

    def defense_debug_state(self) -> dict:
        if self._defense_commitment is None:
            return {
                "defender": "",
                "defense_position": "",
                "defender_target_x": "",
                "defender_target_y": "",
                "defender_target_world_x": "",
                "defender_target_world_y": "",
                "defender_target_qpos_x": "",
                "defender_target_qpos_y": "",
                "defender_actual_x": "",
                "defender_actual_y": "",
                "defender_actual_world_x": "",
                "defender_actual_world_y": "",
            }
        c = self._defense_commitment
        actual = self.defender_gantry(c.defender)
        actual_world = gantry_to_world_position(
            c.defender,
            GantryTarget(float(actual[0]), float(actual[1])),
        )
        return {
            "defender": c.defender,
            "defense_position": c.position,
            "defender_target_x": c.target.x,
            "defender_target_y": c.target.y,
            "defender_target_world_x": c.target.x,
            "defender_target_world_y": c.target.y,
            "defender_target_qpos_x": c.qpos_target.x,
            "defender_target_qpos_y": c.qpos_target.y,
            "defender_actual_x": float(actual[0]),
            "defender_actual_y": float(actual[1]),
            "defender_actual_world_x": actual_world.x,
            "defender_actual_world_y": actual_world.y,
        }


__all__ = [
    "DEFENSE_POSITIONS",
    "DefenseCommitment",
    "V3PrePositionEnv",
]
