#!/usr/bin/env python3
"""V3 environment wrapper for legacy versus optimized racket planning."""

from __future__ import annotations

import copy
import os
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nash_skills.env_wrapper import SkillEnv  # noqa: E402
from v3.racket_target_planner import (  # noqa: E402
    NET_TOP_Z,
    TABLE_HALF_WIDTH,
    TABLE_LENGTH,
    TABLE_SHIFT,
    PlannerConfig,
    PlanResult,
    actual_landing_prediction,
    estimate_legacy,
    mirror_position_to_world,
    mirror_velocity_to_local,
    plan_optimized,
    racket_pose_p2_local_to_world,
    racket_pose_p2_world_to_local,
)


class PlannerSkillEnv(SkillEnv):
    """Existing five-skill environment with a V3 target-planner switch."""

    collected_diagnostics: list[dict[str, Any]] = []

    def __init__(
        self,
        *args,
        planner: str | None = None,
        planner_config: PlannerConfig | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if planner is not None:
            if planner_config is not None and planner_config.planner != planner:
                raise ValueError("planner and planner_config.planner disagree")
            planner_config = PlannerConfig(planner=planner)
        self.planner_config = copy.deepcopy(planner_config or PlannerConfig())
        self.planner_config.validate()
        self._planner_events: list[dict[str, Any]] = []
        self._episode_events: list[dict[str, Any]] = []
        self._episode_id = -1
        self._episode_finished = True
        self._latest_plans: dict[int, dict[str, Any]] = {}
        self._plan_cache: dict[int, dict[str, Any] | None] = {1: None, 2: None}
        self._planning_failures: dict[int, dict[str, Any] | None] = {1: None, 2: None}
        self._phase_key: str | None = None

    @classmethod
    def clear_diagnostics(cls) -> None:
        cls.collected_diagnostics = []

    @classmethod
    def diagnostics(cls) -> list[dict[str, Any]]:
        return list(cls.collected_diagnostics)

    def _world_pose(self, player: int) -> np.ndarray:
        name = "tennis_racket" if player == 1 else "tennis_racket_opp"
        body = self._env.data.body(name)
        position = np.asarray(body.xpos, dtype=float).copy()
        quaternion = np.asarray(body.xquat[[1, 2, 3, 0]], dtype=float).copy()
        return np.concatenate([position, quaternion])

    def _local_pose(self, player: int) -> np.ndarray:
        pose = self._world_pose(player)
        if player == 2:
            return racket_pose_p2_world_to_local(pose)
        return pose

    def _target_local_to_world_pose(self, player: int, target_local: np.ndarray) -> np.ndarray:
        """Convert a planner target to the raw body frame for diagnostics.

        ``set_target_pose_opp`` already expects opponent-local coordinates and
        performs its own visualization transform, so this conversion is not
        applied when sending the target to the environment.
        """
        if player == 2:
            return racket_pose_p2_local_to_world(target_local)
        return np.asarray(target_local, dtype=float).copy()

    def _local_ball(self, player: int) -> tuple[np.ndarray, np.ndarray]:
        position = np.asarray(self._env.data.body("ball").xpos, dtype=float).copy()
        velocity = np.asarray(self._env.data.qvel[-6:-3], dtype=float).copy()
        if player == 2:
            position = mirror_position_to_world(position)
            velocity = mirror_velocity_to_local(velocity)
        return position, velocity

    def _planner_result(self, player: int, kwargs: dict[str, Any]) -> PlanResult:
        ball_pos, ball_vel = self._local_ball(player)
        current_pose = self._local_pose(player)
        x_target = float(kwargs.get("x_target", TABLE_SHIFT + TABLE_LENGTH / 2.0))
        y_target = float(kwargs.get("y_target", 0.38))
        common = {
            "x_target": x_target,
            "y_target": y_target,
            "bounce_factor": float(kwargs.get("bounce_factor", 1.0)),
            "table_z": float(kwargs.get("table_z", 0.56)),
            "gravity": float(kwargs.get("g", -9.81)),
            "x_dis": float(kwargs.get("x_dis", -0.3)),
            "z_dis": float(kwargs.get("z_dis", 0.9)),
            "vx_dis": float(kwargs.get("vx_dis", 4.5)),
            "vy_dis": float(kwargs.get("vy_dis", 2.0)),
        }
        if self.planner_config.planner == "optimized":
            return plan_optimized(ball_pos, ball_vel, current_pose, config=self.planner_config, **common)
        return estimate_legacy(
            ball_pos,
            ball_vel,
            legality_filter=self.planner_config.planner == "legacy_legal",
            net_safety_margin=self.planner_config.net_safety_margin,
            table_x_safety_margin=self.planner_config.table_x_safety_margin,
            table_y_safety_margin=self.planner_config.table_y_safety_margin,
            **common,
        )

    def _current_phase_key(self) -> str | None:
        last_hitter = self._env.rally_state.get("last_hitter")
        if last_hitter in {"ego", "opp"}:
            return str(last_hitter)
        return None

    def _clear_plan_cache(self, player: int | None = None) -> None:
        if player is None:
            self._plan_cache = {1: None, 2: None}
            self._planning_failures = {1: None, 2: None}
            return
        self._plan_cache[player] = None
        self._planning_failures[player] = None

    def _sync_phase_cache(self) -> None:
        phase_key = self._current_phase_key()
        if phase_key != self._phase_key:
            if self._phase_key is not None or phase_key is not None:
                self._clear_plan_cache()
            self._phase_key = phase_key

    def _is_current_receiver(self, player: int) -> bool:
        last_hitter = self._env.rally_state.get("last_hitter")
        if last_hitter == "ego":
            return player == 2
        if last_hitter == "opp":
            return player == 1
        # Before the first recorded hit, use the same local coordinate
        # convention as _incoming_candidates to avoid planning for both sides.
        ball_pos, ball_vel = self._local_ball(player)
        return bool(ball_vel[0] < 0 and ball_pos[0] >= 0.3)

    def _cache_plan(self, player: int, result: PlanResult) -> dict[str, Any]:
        cache = {
            "result": result,
            "target_pose_local": np.asarray(result.target_pose_local, dtype=float).copy(),
            "skill": self._skill1 if player == 1 else self._skill2,
            "phase_key": self._phase_key,
            "plan_age_steps": 0,
            "plan_updates_this_return": 1,
            "planner_calls_per_return": 1,
            "analytic_solve_time": result.analytic_solve_time or 0.0,
            "average_planner_time": result.planner_time or 0.0,
            "explicit_attempted": result.solution_method in {"explicit", "grid"},
            "explicit_success": result.solution_method == "explicit",
            "grid_method_used": result.solution_method == "grid",
            "grid_fallback_used": False,
            "fallback_count": 0,
            "fallback_rate": 0.0,
            "planning_failure_reason": "",
        }
        self._plan_cache[player] = cache
        return cache

    def _cached_plan_meta(self, player: int, cache: dict[str, Any]) -> dict[str, Any]:
        result: PlanResult = cache["result"]
        target_local = np.asarray(cache["target_pose_local"], dtype=float).copy()
        return {
            "result": result,
            "target_pose_local": target_local,
            "target_pose_world": self._target_local_to_world_pose(player, target_local),
            "explicit_attempted": bool(cache["explicit_attempted"]),
            "explicit_success": bool(cache["explicit_success"]),
            "cached_plan_used": True,
            "grid_method_used": bool(cache["grid_method_used"]),
            "grid_fallback_used": bool(cache["grid_fallback_used"]),
            "legacy_fallback_used": False,
            "planning_failure_reason": cache["planning_failure_reason"],
            "plan_age_steps": int(cache["plan_age_steps"]),
            "plan_updates_this_return": int(cache["plan_updates_this_return"]),
            "planner_calls_per_return": int(cache["planner_calls_per_return"]),
            "analytic_solve_time": float(cache["analytic_solve_time"]),
            "average_planner_time": float(cache["average_planner_time"]),
            "fallback_count": int(cache["fallback_count"]),
            "fallback_rate": float(cache["fallback_rate"]),
        }

    def _cached_plan_is_valid(self, player: int, cache: dict[str, Any]) -> bool:
        target = np.asarray(cache.get("target_pose_local"), dtype=float)
        current_skill = self._skill1 if player == 1 else self._skill2
        return bool(
            cache.get("phase_key") == self._phase_key
            and cache.get("skill") == current_skill
            and target.shape == (7,)
            and np.all(np.isfinite(target))
        )

    def _install_planner_update(self, player: int, fallback):
        def update(**kwargs):
            if self.planner_config.planner == "legacy":
                result = self._planner_result(player, kwargs)
                fallback(**kwargs)
                target_local = self._env.curr_target.copy() if player == 1 else self._env.curr_target_opp.copy()
                self._latest_plans[player] = {
                    "result": result,
                    "target_pose_local": np.asarray(target_local, dtype=float).copy(),
                    "target_pose_world": (
                        np.asarray(target_local, dtype=float).copy()
                        if player == 1
                        else self._target_local_to_world_pose(player, np.asarray(target_local, dtype=float))
                    ),
                    "explicit_attempted": False,
                    "explicit_success": False,
                    "cached_plan_used": False,
                    "grid_method_used": False,
                    "grid_fallback_used": False,
                    "legacy_fallback_used": True,
                    "planning_failure_reason": "",
                    "plan_age_steps": "",
                    "plan_updates_this_return": 0,
                    "planner_calls_per_return": 0,
                    "analytic_solve_time": "",
                    "average_planner_time": "",
                    "fallback_count": 0,
                    "fallback_rate": 0.0,
                }
                return

            if self.planner_config.planner == "legacy_legal":
                # Preserve the legacy update cadence: recompute the same
                # contact point and normal candidate set on every call, but
                # install the legality-filtered normal when one is available.
                result = self._planner_result(player, kwargs)
                if result.selected and result.target_pose_local is not None:
                    target_local = np.asarray(result.target_pose_local, dtype=float).copy()
                    if player == 1:
                        self._env.set_target_pose(target_local)
                    else:
                        self._env.set_target_pose_opp(target_local)
                    fallback_used = False
                else:
                    fallback(**kwargs)
                    target_local = self._env.curr_target.copy() if player == 1 else self._env.curr_target_opp.copy()
                    fallback_used = True
                self._latest_plans[player] = {
                    "result": result,
                    "target_pose_local": np.asarray(target_local, dtype=float).copy(),
                    "target_pose_world": (
                        np.asarray(target_local, dtype=float).copy()
                        if player == 1
                        else self._target_local_to_world_pose(player, np.asarray(target_local, dtype=float))
                    ),
                    "explicit_attempted": False,
                    "explicit_success": False,
                    "cached_plan_used": False,
                    "grid_method_used": False,
                    "grid_fallback_used": False,
                    "legacy_fallback_used": fallback_used,
                    "planning_failure_reason": "" if result.selected else result.reason,
                    "plan_age_steps": 0,
                    "plan_updates_this_return": 1,
                    "planner_calls_per_return": 1,
                    "analytic_solve_time": "",
                    "average_planner_time": "",
                    "fallback_count": int(fallback_used),
                    "fallback_rate": float(fallback_used),
                }
                return

            if not self._is_current_receiver(player):
                return

            cache = self._plan_cache.get(player)
            if cache is not None and self._cached_plan_is_valid(player, cache):
                cache["plan_age_steps"] += 1
                target_local = np.asarray(cache["target_pose_local"], dtype=float)
                if player == 1:
                    self._env.set_target_pose(target_local)
                else:
                    # The original opponent planner passes local coordinates
                    # directly; set_target_pose_opp stores that local target
                    # and only transforms its visualization copy.
                    self._env.set_target_pose_opp(target_local)
                self._latest_plans[player] = self._cached_plan_meta(player, cache)
                return
            if cache is not None:
                self._clear_plan_cache(player)

            failed = self._planning_failures.get(player)
            if failed is not None and failed.get("phase_key") == self._phase_key:
                # The same incoming phase already failed and its legacy target
                # is still active. Wait for a new phase instead of replanning.
                return

            result = self._planner_result(player, kwargs)
            if result.selected and result.target_pose_local is not None:
                cache = self._cache_plan(player, result)
                target_local = cache["target_pose_local"]
                if player == 1:
                    self._env.set_target_pose(target_local)
                else:
                    # Keep the planner-local target unchanged for the
                    # opponent controller, matching the legacy call path.
                    self._env.set_target_pose_opp(target_local)
                self._latest_plans[player] = self._cached_plan_meta(player, cache)
                self._latest_plans[player]["cached_plan_used"] = False
                return

            # A failed optimized call must not replace a valid cached target.
            # No-return states (for example an illegal receiver-side bounce)
            # must also avoid creating a pre-bounce legacy target.  Legacy
            # Keep the existing no-target behavior.  A selected common-pool
            # candidate is always cached above; this branch is only for a
            # genuinely empty legal candidate pool.
            use_legacy_fallback = False
            if use_legacy_fallback:
                fallback(**kwargs)
            target_local = self._env.curr_target.copy() if player == 1 else self._env.curr_target_opp.copy()
            self._latest_plans[player] = {
                "result": result,
                "target_pose_local": np.asarray(target_local, dtype=float).copy(),
                "target_pose_world": (
                    np.asarray(target_local, dtype=float).copy()
                    if player == 1
                    else self._target_local_to_world_pose(player, np.asarray(target_local, dtype=float))
                ),
                "explicit_attempted": result.solution_method in {"explicit", "grid"},
                "explicit_success": False,
                "cached_plan_used": False,
                "grid_method_used": False,
                "grid_fallback_used": False,
                "legacy_fallback_used": use_legacy_fallback,
                "planning_failure_reason": result.reason,
                "plan_age_steps": "",
                "plan_updates_this_return": 0,
                "planner_calls_per_return": 1,
                "analytic_solve_time": result.analytic_solve_time or 0.0,
                "average_planner_time": result.planner_time or 0.0,
                "fallback_count": int(use_legacy_fallback),
                "fallback_rate": float(use_legacy_fallback),
            }
            self._planning_failures[player] = {
                "phase_key": self._phase_key,
                "reason": result.reason,
            }

        return update

    def reset(self, seed=None):
        if self._episode_events and not self._episode_finished:
            self._finish_episode("truncated", True, "reset_before_done")
        self._clear_plan_cache()
        self._latest_plans = {}
        self._phase_key = None
        obs, info = super().reset(seed=seed)
        self._episode_id += 1
        self._episode_events = []
        self._episode_finished = False
        self._latest_plans = {}
        return obs, info

    def _pose_error(self, target: np.ndarray | None, actual: np.ndarray) -> tuple[float | None, float | None]:
        if target is None:
            return None, None
        position_error = float(np.linalg.norm(np.asarray(target[:3]) - actual[:3]))
        orientation_error = float((R.from_quat(target[3:]) * R.from_quat(actual[3:]).inv()).magnitude())
        return position_error, orientation_error

    def _finish_episode(self, winner: str, truncated: bool, reason: str) -> None:
        for event in self._episode_events:
            event["rally_winner"] = winner
            if winner in {"ego", "opp"}:
                event["player_won"] = bool(
                    (event["player"] == 1 and winner == "ego")
                    or (event["player"] == 2 and winner == "opp")
                )
            else:
                event["player_won"] = ""
            event["truncated"] = bool(truncated)
            event["termination_reason"] = reason
        self._episode_finished = True

    def _record_contact(self, player: int, plan_meta: dict[str, Any] | None, info: dict[str, Any]) -> None:
        if plan_meta is None:
            plan_meta = {
                "result": PlanResult(False, "no_plan_recorded"),
                "target_pose_world": None,
                "explicit_attempted": False,
                "explicit_success": False,
                "cached_plan_used": False,
                "grid_fallback_used": False,
                "legacy_fallback_used": False,
                "planning_failure_reason": "no_plan_recorded",
                "plan_age_steps": "",
                "plan_updates_this_return": 0,
                "planner_calls_per_return": 0,
                "analytic_solve_time": "",
                "average_planner_time": "",
                "fallback_count": 0,
                "fallback_rate": 0.0,
            }
        result: PlanResult = plan_meta["result"]
        actual_world_pose = self._world_pose(player)
        target_world_pose = plan_meta.get("target_pose_world")
        actual_pos_error, actual_orientation_error = self._pose_error(target_world_pose, actual_world_pose)
        actual_local_pose = self._local_pose(player)
        target_local_pose = plan_meta.get("target_pose_local")
        local_pos_error, local_orientation_error = self._pose_error(target_local_pose, actual_local_pose)
        explicit_selected = next(
            (item for item in result.explicit_candidate_diagnostics if item.get("selected")),
            next(
                (item for item in result.candidate_pool_diagnostics if item.get("selected")),
                {},
            ),
        )

        ball_pos, ball_vel = self._local_ball(player)
        actual_landing_data = actual_landing_prediction(ball_pos, ball_vel)
        actual_landing = None
        actual_net_clearance = None
        if actual_landing_data is not None:
            actual_landing, actual_net_clearance = actual_landing_data
        predicted_landing = result.predicted_landing_local
        landing_error = None
        if predicted_landing is not None and actual_landing is not None:
            landing_error = float(np.linalg.norm(np.asarray(predicted_landing[:2]) - actual_landing[:2]))

        legal = False
        out_of_bounds = False
        net_failure = False
        if actual_landing is None or ball_vel[0] <= 0:
            out_of_bounds = True
        else:
            net_failure = actual_net_clearance is not None and actual_net_clearance < self.planner_config.net_safety_margin
            out_of_bounds = not (
                TABLE_SHIFT + self.planner_config.table_x_safety_margin
                <= actual_landing[0]
                <= TABLE_SHIFT + TABLE_LENGTH - self.planner_config.table_x_safety_margin
                and abs(actual_landing[1])
                <= TABLE_HALF_WIDTH - self.planner_config.table_y_safety_margin
            )
            legal = not net_failure and not out_of_bounds

        skill = self._skill1 if player == 1 else self._skill2
        predicted_world_landing = None
        actual_world_landing = None
        if predicted_landing is not None:
            predicted_world_landing = (
                np.asarray(predicted_landing, dtype=float)
                if player == 1
                else mirror_position_to_world(predicted_landing)
            )
        if actual_landing is not None:
            actual_world_landing = (
                np.asarray(actual_landing, dtype=float)
                if player == 1
                else mirror_position_to_world(actual_landing)
            )

        event = {
            "episode_id": int(self._episode_id),
            "player": int(player),
            "skill": skill,
            "planner": self.planner_config.planner,
            "selected_interception_point": (
                result.contact_position_local.astype(float).tolist()
                if result.contact_position_local is not None else ""
            ),
            "selected_paddle_normal": (
                result.normal_local.astype(float).tolist()
                if result.normal_local is not None else ""
            ),
            "sampled_contact_x": result.sampled_contact_x if result.sampled_contact_x is not None else "",
            "sampled_contact_height": result.sampled_contact_height if result.sampled_contact_height is not None else "",
            "contact_time_after_bounce": (
                result.contact_time_after_bounce if result.contact_time_after_bounce is not None else ""
            ),
            "predicted_bounce_position": (
                result.predicted_bounce_position_local.astype(float).tolist()
                if result.predicted_bounce_position_local is not None else ""
            ),
            "predicted_bounce_time": result.predicted_bounce_time if result.predicted_bounce_time is not None else "",
            "predicted_bounce_legal": (
                "" if result.predicted_bounce_legal is None else bool(result.predicted_bounce_legal)
            ),
            "post_bounce_velocity": (
                result.post_bounce_velocity_local.astype(float).tolist()
                if result.post_bounce_velocity_local is not None else ""
            ),
            "pre_bounce_candidate_attempted": bool(result.pre_bounce_candidate_attempted),
            "explicit_candidate_diagnostics": result.explicit_candidate_diagnostics,
            "grid_candidate_diagnostics": result.grid_candidate_diagnostics,
            "candidate_pool_diagnostics": result.candidate_pool_diagnostics,
            "best_rejected_explicit_candidate": (
                result.best_rejected_explicit_candidate
                if result.best_rejected_explicit_candidate is not None else ""
            ),
            "raw_solved_normal": explicit_selected.get("raw_solved_normal", ""),
            "flipped_solved_normal": explicit_selected.get("flipped_solved_normal", ""),
            "selected_normal": explicit_selected.get("selected_normal", ""),
            "raw_normal_change_rad": explicit_selected.get("raw_normal_change_rad", ""),
            "flipped_normal_change_rad": explicit_selected.get("flipped_normal_change_rad", ""),
            "selected_normal_change_rad": explicit_selected.get("selected_normal_change_rad", ""),
            "solved_flight_time": result.solved_flight_time if result.solved_flight_time is not None else "",
            "solved_outgoing_velocity": (
                result.solved_outgoing_velocity.astype(float).tolist()
                if result.solved_outgoing_velocity is not None else ""
            ),
            "reflection_residual": result.reflection_residual if result.reflection_residual is not None else "",
            "executability_cost": result.executability_cost if result.executability_cost is not None else "",
            "solution_method": result.solution_method,
            "fallback_grid_search": bool(result.fallback_grid_search),
            "explicit_attempted": bool(plan_meta.get("explicit_attempted", False)),
            "explicit_success": bool(plan_meta.get("explicit_success", False)),
            "cached_plan_used": bool(plan_meta.get("cached_plan_used", False)),
            "grid_fallback_used": bool(plan_meta.get("grid_fallback_used", False)),
            "grid_method_used": bool(plan_meta.get("grid_method_used", False)),
            "legacy_fallback_used": bool(plan_meta.get("legacy_fallback_used", False)),
            "planning_failure_reason": plan_meta.get("planning_failure_reason", ""),
            "plan_age_steps": plan_meta.get("plan_age_steps", ""),
            "plan_updates_this_return": plan_meta.get("plan_updates_this_return", 0),
            "planner_calls_per_return": plan_meta.get("planner_calls_per_return", 0),
            "analytic_solve_time": plan_meta.get("analytic_solve_time", ""),
            "average_planner_time": plan_meta.get("average_planner_time", ""),
            "fallback_count": plan_meta.get("fallback_count", 0),
            "fallback_rate": plan_meta.get("fallback_rate", 0.0),
            "predicted_landing_x": "" if predicted_world_landing is None else float(predicted_world_landing[0]),
            "predicted_landing_y": "" if predicted_world_landing is None else float(predicted_world_landing[1]),
            "predicted_net_clearance": result.predicted_net_clearance if result.predicted_net_clearance is not None else "",
            "target_racket_pose": (
                np.asarray(target_world_pose, dtype=float).tolist()
                if target_world_pose is not None else ""
            ),
            "actual_racket_pose": actual_world_pose.astype(float).tolist(),
            "planner_local_racket_pose": actual_local_pose.astype(float).tolist(),
            "planner_local_target_pose": (
                np.asarray(target_local_pose, dtype=float).tolist()
                if target_local_pose is not None else ""
            ),
            "planner_local_position_change": local_pos_error if local_pos_error is not None else "",
            "planner_local_orientation_change_rad": (
                local_orientation_error if local_orientation_error is not None else ""
            ),
            "actual_racket_position_error": actual_pos_error if actual_pos_error is not None else "",
            "actual_racket_orientation_error_rad": actual_orientation_error if actual_orientation_error is not None else "",
            "actual_landing_x": "" if actual_world_landing is None else float(actual_world_landing[0]),
            "actual_landing_y": "" if actual_world_landing is None else float(actual_world_landing[1]),
            "predicted_actual_landing_error": landing_error if landing_error is not None else "",
            "actual_net_clearance": actual_net_clearance if actual_net_clearance is not None else "",
            "legal_return": bool(legal),
            "out_of_bounds": bool(out_of_bounds),
            "net_failure": bool(net_failure),
            "planner_selection_success": bool(result.selected),
            "planner_reason": result.reason,
            "planner_cost": result.cost if result.cost is not None else "",
            "feasible_candidate_count": int(result.feasible_candidates),
            "explicit_candidate_count": int(result.explicit_candidate_count),
            "grid_candidate_count": int(result.grid_candidate_count),
            "rejected_candidate_count": int(result.rejected_candidate_count),
            "selected_landing_error": (
                result.selected_landing_error if result.selected_landing_error is not None else ""
            ),
            "selected_time_until_contact": (
                result.selected_time_until_contact if result.selected_time_until_contact is not None else ""
            ),
            "selected_position_change": (
                result.selected_position_change if result.selected_position_change is not None else ""
            ),
            "selected_orientation_change_rad": (
                result.selected_orientation_change_rad
                if result.selected_orientation_change_rad is not None else ""
            ),
            "selected_translation_speed": (
                result.selected_translation_speed if result.selected_translation_speed is not None else ""
            ),
            "selected_rotation_speed_rad_s": (
                result.selected_rotation_speed_rad_s
                if result.selected_rotation_speed_rad_s is not None else ""
            ),
            "legacy_normal_candidate_count": int(result.legacy_normal_candidate_count),
            "legacy_legal_after_net_count": int(result.legacy_legal_after_net_count),
            "legacy_legal_after_table_count": int(result.legacy_legal_after_table_count),
            "legacy_legality_filter_applied": bool(result.legacy_legality_filter_applied),
            "legacy_legality_filter_fallback": bool(result.legacy_legality_filter_fallback),
            "selected_predicted_net_crossing_height": (
                result.selected_predicted_net_crossing_height
                if result.selected_predicted_net_crossing_height is not None else ""
            ),
            "player_won": "",
            "rally_winner": "",
            "truncated": False,
            "termination_reason": "",
        }
        self._episode_events.append(event)
        self._planner_events.append(event)

    def step(self, action):
        self._sync_phase_cache()
        before_hitter = self._env.rally_state.get("last_hitter")

        if self.planner_config.planner in {"optimized", "legacy_legal"}:
            original_update = self._env.update_target_racket_pose
            original_update_opp = self._env.update_target_racket_pose_opp
            self._env.update_target_racket_pose = self._install_planner_update(1, original_update)
            self._env.update_target_racket_pose_opp = self._install_planner_update(2, original_update_opp)
        else:
            # Prepare a diagnostics-only estimate before the original methods
            # run; the original target update itself is untouched.
            self._latest_plans = {}
            self._latest_plans[1] = self._legacy_plan_meta(1)
            self._latest_plans[2] = self._legacy_plan_meta(2)

        try:
            obs, reward, done, truncated, info = super().step(action)
        finally:
            if self.planner_config.planner in {"optimized", "legacy_legal"}:
                self._env.update_target_racket_pose = original_update
                self._env.update_target_racket_pose_opp = original_update_opp

        after_hitter = self._env.rally_state.get("last_hitter")
        if after_hitter in {"ego", "opp"} and after_hitter != before_hitter:
            hitter = 1 if after_hitter == "ego" else 2
            self._record_contact(hitter, self._latest_plans.get(hitter), info)
            self._clear_plan_cache()

        if done:
            winner = str(info.get("winner", "unknown")) if isinstance(info, dict) else "unknown"
            reason = str(info.get("termination_reason", "done")) if isinstance(info, dict) else "done"
            self._finish_episode(winner, False, reason)
        return obs, reward, done, truncated, info

    def _legacy_plan_meta(self, player: int) -> dict[str, Any]:
        ball_pos, ball_vel = self._local_ball(player)
        result = estimate_legacy(
            ball_pos,
            ball_vel,
            x_target=(self._x_target1 if player == 1 else self._x_target2) or (TABLE_SHIFT + TABLE_LENGTH / 2.0),
            y_target=(self.side_target * 0.38 if player == 1 else self.side_target_opp * 0.38),
        )
        target_local = result.target_pose_local
        if target_local is None:
            target_local = self._env.curr_target.copy() if player == 1 else self._env.curr_target_opp.copy()
        return {
            "result": result,
            "target_pose_local": target_local,
            "target_pose_world": self._target_local_to_world_pose(player, target_local),
        }

    def close(self):
        if self._episode_events and not self._episode_finished:
            self._finish_episode("truncated", True, "environment_closed")
        type(self).collected_diagnostics.extend(self._planner_events)
        self._planner_events = []
        return super().close()
