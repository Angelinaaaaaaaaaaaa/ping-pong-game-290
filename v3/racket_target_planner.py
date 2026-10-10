#!/usr/bin/env python3
"""Legacy-compatible and constrained racket-target planning helpers.

The production environment remains untouched.  ``PlannerSkillEnv`` installs
the optimized path only for V3 ablations; ``planner='legacy'`` delegates to
the original MuJoCo method directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
import time
from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation as R


TABLE_SHIFT = 1.5
TABLE_LENGTH = 1.37
TABLE_HALF_WIDTH = 0.7625
TABLE_Z = 0.56
NET_TOP_Z = 0.67
GRAVITY = -9.81
RACKET_X_MIN = -0.7
RACKET_X_MAX = 0.4
PADDLE_OFFSET = 0.2
PADDLE_RADIUS = 0.2


@dataclass
class PlannerConfig:
    """All optimized-planner tuning parameters in physical units."""

    planner: str = "legacy"
    contact_candidates: int = 21
    normal_candidates: int = 121
    net_safety_margin: float = 0.03
    table_x_safety_margin: float = 0.03
    table_y_safety_margin: float = 0.03
    landing_weight: float = 4.0
    reach_position_weight: float = 0.4
    reach_orientation_weight: float = 0.2
    translation_speed_weight: float = 0.6
    rotation_speed_weight: float = 0.15
    net_safety_weight: float = 0.10
    paddle_center_weight: float = 0.15
    reference_weight: float = 0.05
    landing_scale: float = TABLE_LENGTH
    reach_position_scale: float = 1.0
    paddle_center_scale: float = PADDLE_RADIUS
    reference_scale: float = 0.5
    flight_time_min: float = 0.02
    flight_time_max: float = 3.0
    # Kept for CLI/config compatibility; the optimized path is analytic and
    # does not sample flight times.
    flight_time_samples: int = 96
    explicit_landing_tolerance: float = 0.01
    reflection_tolerance: float = 1e-5
    max_position_change: float = 1.75
    max_orientation_change_rad: float = 2.5
    # Calibrated from successful returns in
    # skill_eval/v3/serve_planner_diagnostic: legacy contacts reached
    # 1.706 m/s and 1.658 rad/s, while successful optimized contacts reached
    # 1.067 m/s and 0.518 rad/s.  The optimized limits use the latter PPO
    # range plus a small margin, without tightening pose bounds.
    max_translation_speed: float = 1.10
    max_rotation_speed_rad_s: float = 0.60
    # Broad hard limits only.  The empirical limits above normalize cost;
    # these prevent physically absurd candidates from entering the pool.
    hard_translation_speed: float = 3.0
    hard_rotation_speed_rad_s: float = 3.0
    net_safety_scale: float = 0.20
    contact_height_candidates: int = 7
    contact_height_min: float = TABLE_Z
    contact_height_max: float = 1.20
    normal_y_min: float = -1.0
    normal_y_max: float = 1.0
    normal_z_min: float = -0.1
    normal_z_max: float = 0.1

    def validate(self) -> None:
        if self.planner not in {"legacy", "legacy_legal", "optimized"}:
            raise ValueError("planner must be 'legacy', 'legacy_legal', or 'optimized'")
        if self.contact_candidates < 1 or self.normal_candidates < 1:
            raise ValueError("contact_candidates and normal_candidates must be positive")
        if self.net_safety_margin < 0 or self.table_x_safety_margin < 0 or self.table_y_safety_margin < 0:
            raise ValueError("safety margins must be non-negative")
        if self.table_x_safety_margin * 2 >= TABLE_LENGTH:
            raise ValueError("table_x_safety_margin leaves no legal table interval")
        if self.table_y_safety_margin >= TABLE_HALF_WIDTH:
            raise ValueError("table_y_safety_margin leaves no legal lateral interval")
        if self.flight_time_min <= 0 or self.flight_time_max <= self.flight_time_min:
            raise ValueError("flight-time bounds must be positive and ordered")
        if self.contact_height_candidates < 1 or self.contact_height_max <= self.contact_height_min:
            raise ValueError("contact height candidates and bounds must be valid")
        if self.explicit_landing_tolerance < 0 or self.reflection_tolerance < 0:
            raise ValueError("explicit solver tolerances must be non-negative")
        if self.max_position_change <= 0 or self.max_orientation_change_rad <= 0:
            raise ValueError("maximum pose changes must be positive")
        if self.max_translation_speed <= 0 or self.max_rotation_speed_rad_s <= 0:
            raise ValueError("execution speed limits must be positive")
        if self.hard_translation_speed <= 0 or self.hard_rotation_speed_rad_s <= 0:
            raise ValueError("hard execution speed limits must be positive")
        if self.net_safety_scale <= 0:
            raise ValueError("net_safety_scale must be positive")
        for name in (
            "landing_weight",
            "reach_position_weight",
            "reach_orientation_weight",
            "translation_speed_weight",
            "rotation_speed_weight",
            "net_safety_weight",
            "paddle_center_weight",
            "reference_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlanResult:
    """Selected optimized plan and predicted contact/landing diagnostics."""

    selected: bool
    reason: str
    contact_position_local: np.ndarray | None = None
    normal_local: np.ndarray | None = None
    predicted_landing_local: np.ndarray | None = None
    predicted_net_clearance: float | None = None
    target_pose_local: np.ndarray | None = None
    reference_error: float | None = None
    cost: float | None = None
    feasible_candidates: int = 0
    sampled_contact_x: float | None = None
    sampled_contact_height: float | None = None
    contact_time_after_bounce: float | None = None
    predicted_bounce_position_local: np.ndarray | None = None
    predicted_bounce_time: float | None = None
    predicted_bounce_legal: bool | None = None
    post_bounce_velocity_local: np.ndarray | None = None
    pre_bounce_candidate_attempted: bool = False
    solved_flight_time: float | None = None
    solved_outgoing_velocity: np.ndarray | None = None
    reflection_residual: float | None = None
    executability_cost: float | None = None
    solution_method: str = ""
    fallback_grid_search: bool = False
    analytic_solve_time: float | None = None
    planner_time: float | None = None
    explicit_candidate_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    best_rejected_explicit_candidate: dict[str, Any] | None = None
    grid_candidate_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    candidate_pool_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    explicit_candidate_count: int = 0
    grid_candidate_count: int = 0
    rejected_candidate_count: int = 0
    selected_landing_error: float | None = None
    selected_time_until_contact: float | None = None
    selected_position_change: float | None = None
    selected_orientation_change_rad: float | None = None
    selected_translation_speed: float | None = None
    selected_rotation_speed_rad_s: float | None = None
    legacy_normal_candidate_count: int = 0
    legacy_legal_after_net_count: int = 0
    legacy_legal_after_table_count: int = 0
    legacy_legality_filter_applied: bool = False
    legacy_legality_filter_fallback: bool = False
    selected_predicted_net_crossing_height: float | None = None

    def as_dict(self) -> dict[str, Any]:
        def value(item):
            if isinstance(item, np.ndarray):
                return item.astype(float).tolist()
            if isinstance(item, np.floating):
                return float(item)
            if isinstance(item, np.integer):
                return int(item)
            return item

        return {key: value(val) for key, val in self.__dict__.items()}


def _normal_candidates(config: PlannerConfig) -> np.ndarray:
    """Generate the same [1, y, z] normal family as the legacy planner."""
    count = int(config.normal_candidates)
    side = max(1, int(np.ceil(np.sqrt(count))))
    y_values = np.linspace(config.normal_y_min, config.normal_y_max, side)
    z_values = np.linspace(config.normal_z_min, config.normal_z_max, side)
    grid = np.asarray(np.meshgrid(y_values, z_values)).reshape(2, -1).T
    if len(grid) > count:
        indices = np.linspace(0, len(grid) - 1, count).round().astype(int)
        grid = grid[indices]
    normals = np.concatenate([np.ones((len(grid), 1)), grid], axis=1)
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


def _incoming_candidates(
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    config: PlannerConfig,
    bounce_factor: float,
    table_z: float,
    gravity: float,
    x_dis: float,
    z_dis: float,
    vx_dis: float,
    vy_dis: float,
) -> list[dict[str, np.ndarray | float]]:
    """Build multiple feasible contact candidates using legacy ball kinematics."""
    if ball_vel[0] >= 0 or ball_pos[0] < 0.3 or ball_vel[2] > 8.0 or ball_pos[2] < table_z + 0.1:
        return []

    v_table = -np.sqrt(max(ball_vel[2] ** 2 - 2 * gravity * (ball_pos[2] - table_z), 0.0))
    x_table = ball_vel[0] * (v_table - ball_vel[2]) / gravity + ball_pos[0]
    if x_table > 0:
        x_values = np.linspace(RACKET_X_MIN, min(RACKET_X_MAX, x_table), config.contact_candidates)
    else:
        x_values = np.linspace(RACKET_X_MIN, RACKET_X_MAX, config.contact_candidates)

    if x_table > 0:
        vz_bounce = -np.sqrt(
            max(-2 * gravity * max(ball_pos[2] - table_z, 0.0) + ball_vel[2] ** 2, 0.0)
        )
        t_bounce = (vz_bounce - ball_vel[2]) / gravity
    else:
        vz_bounce = None
        t_bounce = None

    candidates = []
    for x_contact in x_values:
        t_contact = (x_contact - ball_pos[0]) / ball_vel[0]
        if t_contact <= 0:
            continue
        contact_y = ball_pos[1] + ball_vel[1] * t_contact
        if x_table > 0:
            t_remaining = t_contact - float(t_bounce)
            contact_z = (
                table_z
                + 0.5 * gravity * t_remaining * t_remaining
                - float(vz_bounce) * t_remaining * bounce_factor
            )
            outgoing_velocity = np.array([
                ball_vel[0],
                ball_vel[1],
                -float(vz_bounce) + gravity * t_remaining,
            ], dtype=float)
        else:
            contact_z = ball_pos[2] + ball_vel[2] * t_contact + 0.5 * gravity * t_contact * t_contact
            outgoing_velocity = np.array([
                ball_vel[0],
                ball_vel[1],
                ball_vel[2] + gravity * t_contact,
            ], dtype=float)

        contact = np.array([x_contact, contact_y, contact_z], dtype=float)
        reference_t = (x_contact - x_dis) / vx_dis
        reference_z = z_dis + vy_dis * reference_t + 0.5 * gravity * reference_t * reference_t
        if not np.all(np.isfinite(contact)) or not np.all(np.isfinite(outgoing_velocity)):
            continue
        candidates.append({
            "contact": contact,
            "incoming_velocity": outgoing_velocity,
            "reference_error": abs(contact_z - reference_z) / max(config.reference_scale, 1e-6),
        })
    return candidates


@lru_cache(maxsize=32)
def _contact_height_set(count: int, minimum: float, maximum: float) -> tuple[float, ...]:
    """Cache the small, fixed contact-height set used by explicit planning."""
    return tuple(float(value) for value in np.linspace(minimum, maximum, count))


def _positive_quadratic_roots(a: float, b: float, c: float) -> list[float]:
    """Return finite positive real roots without iterative sampling."""
    if abs(a) <= 1e-12:
        if abs(b) <= 1e-12:
            return []
        root = -c / b
        return [float(root)] if root > 0 and np.isfinite(root) else []
    discriminant = b * b - 4.0 * a * c
    tolerance = 1e-10 * max(1.0, abs(b * b), abs(4.0 * a * c))
    if discriminant < -tolerance:
        return []
    discriminant = max(0.0, discriminant)
    if discriminant == 0.0:
        roots = [-b / (2.0 * a)]
    else:
        root_discriminant = float(np.sqrt(discriminant))
        roots = [(-b - root_discriminant) / (2.0 * a), (-b + root_discriminant) / (2.0 * a)]
    return sorted({float(root) for root in roots if root > 0 and np.isfinite(root)})


def _predict_receiver_bounce(
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    *,
    table_z: float,
    gravity: float,
) -> dict[str, Any]:
    """Predict the first future table contact in receiver-local coordinates."""
    ball_pos = np.asarray(ball_pos, dtype=float)
    ball_vel = np.asarray(ball_vel, dtype=float)
    empty: dict[str, Any] = {
        "position": None,
        "time": None,
        "velocity_before": None,
        "post_velocity": None,
        "legal": False,
        "reason": "no_future_table_bounce",
    }
    if not np.all(np.isfinite(ball_pos)) or not np.all(np.isfinite(ball_vel)):
        empty["reason"] = "nonfinite_ball_state"
        return empty

    bounce_times = _positive_quadratic_roots(
        0.5 * gravity,
        float(ball_vel[2]),
        float(ball_pos[2] - table_z),
    )
    if not bounce_times:
        return empty

    bounce_time = float(min(bounce_times))
    bounce_position = ball_pos + ball_vel * bounce_time
    bounce_position[2] = table_z
    velocity_before = ball_vel.copy()
    velocity_before[2] += gravity * bounce_time

    # This matches mujoco_env_comp.py: table contact reverses only z velocity.
    post_velocity = velocity_before.copy()
    post_velocity[2] = -velocity_before[2]
    legal = bool(
        TABLE_SHIFT - TABLE_LENGTH <= bounce_position[0] <= TABLE_SHIFT
        and abs(bounce_position[1]) <= TABLE_HALF_WIDTH
    )
    return {
        "position": np.asarray(bounce_position, dtype=float),
        "time": bounce_time,
        "velocity_before": np.asarray(velocity_before, dtype=float),
        "post_velocity": np.asarray(post_velocity, dtype=float),
        "legal": legal,
        "reason": "legal_receiver_bounce" if legal else "receiver_bounce_out_of_bounds",
    }


def _post_bounce_height_candidates(
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    config: PlannerConfig,
    table_z: float,
    gravity: float,
    x_dis: float,
    z_dis: float,
    vx_dis: float,
    vy_dis: float,
) -> tuple[dict[str, Any], list[dict[str, np.ndarray | float]]]:
    """Build contact candidates only from the legal post-bounce trajectory."""
    if ball_vel[0] >= 0 or ball_pos[0] < 0.3 or ball_vel[2] > 8.0 or ball_pos[2] < table_z + 0.1:
        return {"position": None, "time": None, "post_velocity": None, "legal": False, "reason": "invalid_incoming_state"}, []

    bounce = _predict_receiver_bounce(ball_pos, ball_vel, table_z=table_z, gravity=gravity)
    if not bounce["legal"]:
        return bounce, []

    bounce_position = np.asarray(bounce["position"], dtype=float)
    post_velocity = np.asarray(bounce["post_velocity"], dtype=float)
    if post_velocity[0] >= -1e-8:
        bounce["legal"] = False
        bounce["reason"] = "receiver_bounce_not_toward_racket"
        return bounce, []

    candidates: list[dict[str, np.ndarray | float]] = []
    for contact_height in _contact_height_set(
        int(config.contact_height_candidates),
        float(config.contact_height_min),
        float(config.contact_height_max),
    ):
        # z(s) = table_z + v_post_z*s + 0.5*g*s^2, s measured after bounce.
        contact_times = _positive_quadratic_roots(
            0.5 * gravity,
            float(post_velocity[2]),
            float(table_z - contact_height),
        )
        for contact_time in contact_times:
            if contact_time <= 1e-8:
                continue
            x_contact = bounce_position[0] + post_velocity[0] * contact_time
            if not (RACKET_X_MIN <= x_contact <= RACKET_X_MAX):
                continue
            contact_y = bounce_position[1] + post_velocity[1] * contact_time
            contact_z = (
                table_z
                + post_velocity[2] * contact_time
                + 0.5 * gravity * contact_time * contact_time
            )
            incoming_velocity = post_velocity.copy()
            incoming_velocity[2] += gravity * contact_time

            contact = np.array([x_contact, contact_y, contact_z], dtype=float)
            reference_t = (x_contact - x_dis) / vx_dis
            reference_z = z_dis + vy_dis * reference_t + 0.5 * gravity * reference_t * reference_t
            if not np.all(np.isfinite(contact)) or not np.all(np.isfinite(incoming_velocity)):
                continue
            candidates.append(
                {
                    "contact": contact,
                    "incoming_velocity": incoming_velocity,
                    "contact_height": float(contact_height),
                    "contact_time_after_bounce": float(contact_time),
                    "bounce_position": bounce_position.copy(),
                    "bounce_time": float(bounce["time"]),
                    "post_bounce_velocity": post_velocity.copy(),
                    "reference_error": abs(contact_z - reference_z) / max(config.reference_scale, 1e-6),
                }
            )
    return bounce, candidates


def _reflection_prediction(
    contact: np.ndarray,
    velocity: np.ndarray,
    normal: np.ndarray,
    table_z: float,
    gravity: float,
) -> tuple[np.ndarray, float] | None:
    """Predict net clearance and table landing with the legacy reflection model."""
    reflected = velocity - 2.0 * np.dot(velocity, normal) * normal
    if reflected[0] <= 1e-8:
        return None

    t_net = (TABLE_SHIFT - contact[0]) / reflected[0]
    if t_net <= 0:
        return None
    net_z = contact[2] + reflected[2] * t_net + 0.5 * gravity * t_net * t_net
    net_clearance = net_z - (NET_TOP_Z)

    vz_hit = -np.sqrt(max(-2 * gravity * max(contact[2] - table_z, 0.0) + reflected[2] ** 2, 0.0))
    t_hit = (vz_hit - reflected[2]) / gravity
    if t_hit <= 0:
        return None
    landing = np.array([
        contact[0] + reflected[0] * t_hit,
        contact[1] + reflected[1] * t_hit,
        table_z,
    ], dtype=float)
    if not np.all(np.isfinite(landing)):
        return None
    return landing, float(net_clearance)


def _projectile_prediction(
    contact: np.ndarray,
    outgoing_velocity: np.ndarray,
    target: np.ndarray,
    gravity: float,
) -> tuple[np.ndarray, float] | None:
    """Predict net clearance from a velocity solved for ``target``.

    The explicit solver already enforces the table landing point.  Re-solving
    the landing-time quadratic here would duplicate work and can introduce a
    small numerical discrepancy, so the target itself is the landing result.
    """
    if outgoing_velocity[0] <= 1e-8:
        return None
    t_net = (TABLE_SHIFT - contact[0]) / outgoing_velocity[0]
    if t_net <= 0:
        return None
    net_z = contact[2] + outgoing_velocity[2] * t_net + 0.5 * gravity * t_net * t_net
    net_clearance = net_z - NET_TOP_Z
    if not np.all(np.isfinite(target)) or not np.isfinite(net_clearance):
        return None
    return np.asarray(target, dtype=float).copy(), float(net_clearance)


def _target_pose(contact: np.ndarray, normal: np.ndarray) -> np.ndarray | None:
    """Derive the same racket pose convention used by the legacy method."""
    theta_z = np.arctan2(contact[2], contact[1])
    x_axis = np.array([0.0, np.cos(theta_z), np.sin(theta_z)], dtype=float)
    if abs(normal[0]) < 1e-8:
        return None
    x_axis[0] = (-x_axis[1] * normal[1] - x_axis[2] * normal[2]) / normal[0]
    x_norm = np.linalg.norm(x_axis)
    if x_norm <= 1e-8:
        return None
    x_axis /= x_norm
    y_axis = np.cross(normal, x_axis)
    rotation = R.from_matrix(np.array([x_axis, y_axis, normal]).T)
    pose_position = contact.copy()
    pose_position[2] -= PADDLE_OFFSET * np.sin(theta_z)
    pose_position[1] -= PADDLE_OFFSET * np.cos(theta_z)
    pose_position[0] -= 0.1
    return np.concatenate([pose_position, rotation.as_quat()])


def _minimum_normal_alignment_pose(
    position: np.ndarray,
    current_pose: np.ndarray,
    desired_normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Keep the solved normal and minimize free twist from the current pose."""
    current_rotation = R.from_quat(current_pose[3:])
    current_matrix = current_rotation.as_matrix()
    current_normal = current_matrix[:, 2]
    desired_normal = np.asarray(desired_normal, dtype=float)
    desired_length = float(np.linalg.norm(desired_normal))
    if desired_length <= 1e-8 or not np.all(np.isfinite(desired_normal)):
        return None
    desired_normal = desired_normal / desired_length

    dot = float(np.clip(np.dot(current_normal, desired_normal), -1.0, 1.0))
    cross = np.cross(current_normal, desired_normal)
    cross_length = float(np.linalg.norm(cross))
    if cross_length <= 1e-10:
        if dot >= 0.0:
            alignment = R.identity()
        else:
            # Pick a deterministic axis perpendicular to the current normal
            # for the 180-degree anti-parallel case.
            basis_index = int(np.argmin(np.abs(current_normal)))
            basis = np.zeros(3, dtype=float)
            basis[basis_index] = 1.0
            axis = np.cross(current_normal, basis)
            axis /= np.linalg.norm(axis)
            alignment = R.from_rotvec(np.pi * axis)
    else:
        angle = float(np.arctan2(cross_length, dot))
        alignment = R.from_rotvec((cross / cross_length) * angle)

    target_matrix = alignment.as_matrix() @ current_matrix
    if np.linalg.norm(target_matrix[:, 2] - desired_normal) > 1e-7:
        return None
    target_pose = np.concatenate([
        np.asarray(position, dtype=float),
        R.from_matrix(target_matrix).as_quat(),
    ])
    return target_pose, current_normal, float(alignment.magnitude())


def _target_poses(contact: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Vectorized form of ``_target_pose`` for the optimized normal search."""
    theta_z = np.arctan2(contact[2], contact[1])
    x_axis = np.tile(np.array([0.0, np.cos(theta_z), np.sin(theta_z)]), (len(normals), 1))
    x_axis[:, 0] = (-x_axis[:, 1] * normals[:, 1] - x_axis[:, 2] * normals[:, 2]) / normals[:, 0]
    x_axis /= np.linalg.norm(x_axis, axis=1, keepdims=True)
    y_axis = np.cross(normals, x_axis)
    rotations = np.stack([x_axis, y_axis, normals], axis=2)
    quaternions = R.from_matrix(rotations).as_quat()
    positions = np.tile(contact, (len(normals), 1))
    positions[:, 2] -= PADDLE_OFFSET * np.sin(theta_z)
    positions[:, 1] -= PADDLE_OFFSET * np.cos(theta_z)
    positions[:, 0] -= 0.1
    return np.concatenate([positions, quaternions], axis=1)


def _orientation_error(target_quat: np.ndarray, current_quat: np.ndarray) -> float:
    target = R.from_quat(target_quat)
    current = R.from_quat(current_quat)
    return float((target * current.inv()).magnitude() / np.pi)


def _cost(
    config: PlannerConfig,
    contact: np.ndarray,
    target_pose: np.ndarray,
    current_pose: np.ndarray,
    landing: np.ndarray,
    x_target: float,
    y_target: float,
    reference_error: float,
) -> float:
    landing_error = np.linalg.norm(landing[:2] - np.array([x_target, y_target])) / max(config.landing_scale, 1e-6)
    reach_position = np.linalg.norm(target_pose[:3] - current_pose[:3]) / max(config.reach_position_scale, 1e-6)
    reach_orientation = _orientation_error(target_pose[3:], current_pose[3:])
    center_error = np.linalg.norm(contact - target_pose[:3]) / max(config.paddle_center_scale, 1e-6)
    return float(
        config.landing_weight * landing_error
        + config.reach_position_weight * reach_position
        + config.reach_orientation_weight * reach_orientation
        + config.paddle_center_weight * center_error
        + config.reference_weight * reference_error
    )


def _explicit_outgoing_velocity(
    contact: np.ndarray,
    target: np.ndarray,
    flight_time: float,
    gravity: float,
) -> np.ndarray:
    """Return the projectile velocity that reaches ``target`` at ``T``."""
    displacement = target - contact
    return np.array(
        [
            displacement[0] / flight_time,
            displacement[1] / flight_time,
            (displacement[2] - 0.5 * gravity * flight_time * flight_time) / flight_time,
        ],
        dtype=float,
    )


def _solve_flight_times(
    contact: np.ndarray,
    incoming_velocity: np.ndarray,
    target: np.ndarray,
    config: PlannerConfig,
    gravity: float,
) -> list[float]:
    """Solve the stationary-paddle speed equation analytically in ``T**2``."""
    incoming_speed_sq = float(np.dot(incoming_velocity, incoming_velocity))
    if incoming_speed_sq <= 1e-16 or not np.isfinite(incoming_speed_sq):
        return []

    dz = float(target[2] - contact[2])
    displacement_sq = float(np.dot(target - contact, target - contact))
    u_roots = _positive_quadratic_roots(
        0.25 * gravity * gravity,
        -(dz * gravity + incoming_speed_sq),
        displacement_sq,
    )
    flight_times = []
    for u in u_roots:
        flight_time = float(np.sqrt(u))
        if config.flight_time_min <= flight_time <= config.flight_time_max:
            flight_times.append(flight_time)
    return flight_times


def _executability_cost(
    config: PlannerConfig,
    contact: np.ndarray,
    target_pose: np.ndarray,
    current_pose: np.ndarray,
    reference_error: float,
) -> tuple[float, float, float, float]:
    """Return the legacy executability terms.

    Speed-aware terms are added in ``_plan_explicit`` because contact time is
    part of each analytic candidate, not part of a pose alone.
    """
    reach_position = float(np.linalg.norm(target_pose[:3] - current_pose[:3])) / max(config.reach_position_scale, 1e-6)
    reach_orientation = _orientation_error(target_pose[3:], current_pose[3:])
    center_error = float(np.linalg.norm(contact - target_pose[:3])) / max(config.paddle_center_scale, 1e-6)
    total = (
        config.reach_position_weight * reach_position
        + config.reach_orientation_weight * reach_orientation
        + config.paddle_center_weight * center_error
        + config.reference_weight * float(reference_error)
    )
    return float(total), reach_position, reach_orientation, center_error


def _plan_explicit(
    candidates: list[dict[str, np.ndarray | float]],
    current_pose: np.ndarray,
    *,
    x_target: float,
    y_target: float,
    config: PlannerConfig,
    table_z: float,
    gravity: float,
) -> PlanResult:
    target = np.array([x_target, y_target, table_z], dtype=float)
    feasible: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    rejection_reasons: set[str] = set()
    candidate_diagnostics: list[dict[str, Any]] = []
    best_rejected: dict[str, Any] | None = None
    current_rotation = R.from_quat(current_pose[3:])
    current_normal = current_rotation.as_matrix()[:, 2]

    def vector_value(value: np.ndarray | None) -> list[float] | None:
        if value is None:
            return None
        return np.asarray(value, dtype=float).tolist()

    def reject(diagnostic: dict[str, Any], reason: str) -> None:
        nonlocal best_rejected
        diagnostic["rejection_reason"] = reason
        diagnostic["selected"] = False
        candidate_diagnostics.append(diagnostic)
        rejection_reasons.add(reason)
        score = diagnostic.get("candidate_cost")
        if score is None:
            score = float("inf")
        best_score = best_rejected.get("candidate_cost") if best_rejected is not None else None
        if best_score is None:
            best_score = float("inf")
        if best_rejected is None or float(score) < float(best_score):
            best_rejected = diagnostic

    for candidate in candidates:
        contact = np.asarray(candidate["contact"], dtype=float)
        incoming = np.asarray(candidate["incoming_velocity"], dtype=float)
        flight_times = _solve_flight_times(contact, incoming, target, config, gravity)
        if not flight_times:
            reject(
                {
                    "contact_position": vector_value(contact),
                    "contact_height": float(candidate["contact_height"]),
                    "contact_time_after_bounce": float(candidate["contact_time_after_bounce"]),
                    "time_until_contact": float(candidate["bounce_time"])
                    + float(candidate["contact_time_after_bounce"]),
                    "flight_time": None,
                    "current_paddle_normal": vector_value(current_normal),
                    "required_paddle_normal": None,
                    "raw_solved_normal": None,
                    "flipped_solved_normal": None,
                    "selected_normal": None,
                    "raw_normal_change_rad": None,
                    "flipped_normal_change_rad": None,
                    "selected_normal_change_rad": None,
                    "normal_change_rad": None,
                    "theta_orientation_change_rad": None,
                    "minimum_orientation_change_rad": None,
                    "position_change": None,
                    "required_translation_speed": None,
                    "required_rotation_speed_rad_s": None,
                    "translation_speed_cost": None,
                    "rotation_speed_cost": None,
                    "candidate_cost": None,
                },
                "no_valid_flight_time_root",
            )
            continue

        for flight_time in flight_times:
            outgoing = _explicit_outgoing_velocity(contact, target, flight_time, gravity)
            normal_delta = incoming - outgoing
            normal_length = float(np.linalg.norm(normal_delta))
            diagnostic = {
                "contact_position": vector_value(contact),
                "contact_height": float(candidate["contact_height"]),
                "contact_time_after_bounce": float(candidate["contact_time_after_bounce"]),
                "time_until_contact": float(candidate["bounce_time"])
                + float(candidate["contact_time_after_bounce"]),
                "flight_time": float(flight_time),
                "current_paddle_normal": vector_value(current_normal),
                "required_paddle_normal": None,
                "raw_solved_normal": None,
                "flipped_solved_normal": None,
                "selected_normal": None,
                "raw_normal_change_rad": None,
                "flipped_normal_change_rad": None,
                "selected_normal_change_rad": None,
                "normal_change_rad": None,
                "theta_orientation_change_rad": None,
                "minimum_orientation_change_rad": None,
                "position_change": None,
                "required_translation_speed": None,
                "required_rotation_speed_rad_s": None,
                "translation_speed_cost": None,
                "rotation_speed_cost": None,
                "candidate_cost": None,
            }
            if normal_length <= 1e-8:
                reject(diagnostic, "zero_normal")
                continue
            raw_normal = normal_delta / normal_length
            flipped_normal = -raw_normal
            raw_normal_change = float(
                np.arccos(np.clip(np.dot(current_normal, raw_normal), -1.0, 1.0))
            )
            flipped_normal_change = float(
                np.arccos(np.clip(np.dot(current_normal, flipped_normal), -1.0, 1.0))
            )
            if flipped_normal_change < raw_normal_change:
                normal = flipped_normal
                normal_change = flipped_normal_change
            else:
                normal = raw_normal
                normal_change = raw_normal_change
            diagnostic["required_paddle_normal"] = vector_value(normal)
            diagnostic["raw_solved_normal"] = vector_value(raw_normal)
            diagnostic["flipped_solved_normal"] = vector_value(flipped_normal)
            diagnostic["selected_normal"] = vector_value(normal)
            diagnostic["raw_normal_change_rad"] = raw_normal_change
            diagnostic["flipped_normal_change_rad"] = flipped_normal_change
            diagnostic["selected_normal_change_rad"] = normal_change
            diagnostic["normal_change_rad"] = normal_change

            theta_pose = _target_pose(contact, normal)
            if theta_pose is not None and np.all(np.isfinite(theta_pose)):
                position_change = float(np.linalg.norm(theta_pose[:3] - current_pose[:3]))
                theta_orientation_change = _orientation_error(theta_pose[3:], current_pose[3:]) * np.pi
                diagnostic["theta_orientation_change_rad"] = theta_orientation_change
                diagnostic["position_change"] = position_change
                minimum_pose_data = _minimum_normal_alignment_pose(
                    theta_pose[:3], current_pose, normal
                )
            else:
                position_change = None
                minimum_pose_data = None

            if minimum_pose_data is None:
                reject(diagnostic, "invalid_minimum_orientation_pose")
                continue
            pose, _verified_current_normal, minimum_orientation_change = minimum_pose_data
            diagnostic["minimum_orientation_change_rad"] = minimum_orientation_change
            time_until_contact = float(candidate["bounce_time"]) + float(
                candidate["contact_time_after_bounce"]
            )
            if time_until_contact <= 0 or not np.isfinite(time_until_contact):
                reject(diagnostic, "invalid_contact_timing")
                continue

            position_change = diagnostic["position_change"]
            required_translation_speed = float(position_change) / time_until_contact
            required_rotation_speed = float(minimum_orientation_change) / time_until_contact
            diagnostic["required_translation_speed"] = required_translation_speed
            diagnostic["required_rotation_speed_rad_s"] = required_rotation_speed
            diagnostic["translation_speed_cost"] = (
                required_translation_speed / max(config.max_translation_speed, 1e-6)
            )
            diagnostic["rotation_speed_cost"] = (
                required_rotation_speed / max(config.max_rotation_speed_rad_s, 1e-6)
            )

            if diagnostic["position_change"] is not None:
                candidate_cost, _, _, _ = _executability_cost(
                    config,
                    contact,
                    pose,
                    current_pose,
                    float(candidate["reference_error"]),
                )
                diagnostic["candidate_cost"] = (
                    candidate_cost
                    + config.translation_speed_weight * diagnostic["translation_speed_cost"]
                    + config.rotation_speed_weight * diagnostic["rotation_speed_cost"]
                )

            if normal_change > config.max_orientation_change_rad:
                reject(diagnostic, "excessive_normal_change")
                continue

            reflected = incoming - 2.0 * float(np.dot(incoming, normal)) * normal
            reflection_residual = float(np.linalg.norm(reflected - outgoing))
            if reflection_residual > config.reflection_tolerance:
                reject(diagnostic, "reflection_residual")
                continue

            prediction = _projectile_prediction(contact, outgoing, target, gravity)
            if prediction is None:
                reject(diagnostic, "wrong_direction_or_flight")
                continue
            landing, net_clearance = prediction
            landing_error = float(np.linalg.norm(landing[:2] - target[:2]))
            if landing_error > config.explicit_landing_tolerance:
                reject(diagnostic, "landing_mismatch")
                continue
            if net_clearance < config.net_safety_margin:
                reject(diagnostic, "net_clearance")
                continue
            if not (
                TABLE_SHIFT + config.table_x_safety_margin
                <= landing[0]
                <= TABLE_SHIFT + TABLE_LENGTH - config.table_x_safety_margin
                and abs(landing[1]) <= TABLE_HALF_WIDTH - config.table_y_safety_margin
            ):
                reject(diagnostic, "table_bounds")
                continue

            if position_change is None or not np.all(np.isfinite(pose)):
                reject(diagnostic, "invalid_pose")
                continue
            if position_change > config.max_position_change:
                reject(diagnostic, "static_position_limit")
                continue
            if minimum_orientation_change > config.max_orientation_change_rad:
                reject(diagnostic, "static_orientation_limit")
                continue
            if required_translation_speed > config.max_translation_speed:
                reject(diagnostic, "translation_speed_limit")
                continue
            if required_rotation_speed > config.max_rotation_speed_rad_s:
                reject(diagnostic, "rotation_speed_limit")
                continue

            cost = float(diagnostic["candidate_cost"])
            diagnostic["rejection_reason"] = ""
            diagnostic["selected"] = False
            candidate_diagnostics.append(diagnostic)
            feasible.append(
                (
                    cost,
                    {
                        "contact": contact,
                        "normal": normal,
                        "landing": landing,
                        "net_clearance": net_clearance,
                        "pose": pose,
                        "reference_error": float(candidate["reference_error"]),
                        "contact_height": float(candidate["contact_height"]),
                        "contact_time_after_bounce": float(candidate["contact_time_after_bounce"]),
                        "bounce_position": np.asarray(candidate["bounce_position"], dtype=float),
                        "bounce_time": float(candidate["bounce_time"]),
                        "post_bounce_velocity": np.asarray(candidate["post_bounce_velocity"], dtype=float),
                        "flight_time": float(flight_time),
                        "outgoing": outgoing,
                        "reflection_residual": reflection_residual,
                    },
                    diagnostic,
                )
            )

    if not feasible:
        detail = ",".join(sorted(rejection_reasons)) or "no_valid_flight_time_root"
        return PlanResult(
            False,
            f"explicit_solver_failed:{detail}",
            solution_method="explicit",
            explicit_candidate_diagnostics=candidate_diagnostics,
            best_rejected_explicit_candidate=best_rejected,
        )

    cost, selected, selected_diagnostic = min(feasible, key=lambda item: item[0])
    selected_diagnostic["selected"] = True
    selected_diagnostic["rejection_reason"] = ""
    return PlanResult(
        selected=True,
        reason="optimized_explicit",
        contact_position_local=np.asarray(selected["contact"]),
        normal_local=np.asarray(selected["normal"]),
        predicted_landing_local=np.asarray(selected["landing"]),
        predicted_net_clearance=float(selected["net_clearance"]),
        target_pose_local=np.asarray(selected["pose"]),
        reference_error=float(selected["reference_error"]),
        cost=float(cost),
        feasible_candidates=len(feasible),
        sampled_contact_x=float(selected["contact"][0]),
        sampled_contact_height=float(selected["contact_height"]),
        contact_time_after_bounce=float(selected["contact_time_after_bounce"]),
        predicted_bounce_position_local=np.asarray(selected["bounce_position"]),
        predicted_bounce_time=float(selected["bounce_time"]),
        predicted_bounce_legal=True,
        post_bounce_velocity_local=np.asarray(selected["post_bounce_velocity"]),
        solved_flight_time=float(selected["flight_time"]),
        solved_outgoing_velocity=np.asarray(selected["outgoing"]),
        reflection_residual=float(selected["reflection_residual"]),
        executability_cost=float(cost),
        solution_method="explicit",
        explicit_candidate_diagnostics=candidate_diagnostics,
        best_rejected_explicit_candidate=best_rejected,
    )


def _plan_grid_optimized(
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    current_pose: np.ndarray,
    *,
    x_target: float,
    y_target: float,
    config: PlannerConfig,
    bounce_factor: float = 1.0,
    table_z: float = TABLE_Z,
    gravity: float = GRAVITY,
    x_dis: float = -0.3,
    z_dis: float = 0.9,
    vx_dis: float = 4.5,
    vy_dis: float = 2.0,
    candidates: list[dict[str, np.ndarray | float]] | None = None,
) -> PlanResult:
    """Grid-search normals for already-valid post-bounce contacts."""
    config.validate()
    if candidates is None:
        return PlanResult(False, "no_post_bounce_contact_candidates")
    if not candidates:
        return PlanResult(False, "no_post_bounce_contact_candidates")

    normals = _normal_candidates(config)
    current_pose = np.asarray(current_pose, dtype=float)
    feasible = []
    for candidate in candidates:
        contact = candidate["contact"]
        velocity = candidate["incoming_velocity"]
        reflected = velocity[None, :] - 2.0 * np.sum(velocity[None, :] * normals, axis=1)[:, None] * normals
        valid_direction = reflected[:, 0] > 1e-8
        with np.errstate(divide="ignore", invalid="ignore"):
            t_net = (TABLE_SHIFT - contact[0]) / reflected[:, 0]
            net_z = contact[2] + reflected[:, 2] * t_net + 0.5 * gravity * t_net * t_net
            vz_hit = -np.sqrt(np.maximum(-2 * gravity * max(contact[2] - table_z, 0.0) + reflected[:, 2] ** 2, 0.0))
            t_hit = (vz_hit - reflected[:, 2]) / gravity
            landing = np.column_stack([
                contact[0] + reflected[:, 0] * t_hit,
                contact[1] + reflected[:, 1] * t_hit,
                np.full(len(normals), table_z),
            ])
        net_clearance = net_z - NET_TOP_Z
        safe = (
            valid_direction
            & (t_net > 0)
            & (t_hit > 0)
            & np.isfinite(landing).all(axis=1)
            & (net_clearance >= config.net_safety_margin)
            & (landing[:, 0] >= TABLE_SHIFT + config.table_x_safety_margin)
            & (landing[:, 0] <= TABLE_SHIFT + TABLE_LENGTH - config.table_x_safety_margin)
            & (np.abs(landing[:, 1]) <= TABLE_HALF_WIDTH - config.table_y_safety_margin)
        )
        if not np.any(safe):
            continue

        poses = _target_poses(contact, normals)
        current_rotation = R.from_quat(current_pose[3:])
        orientation_cost = np.asarray(
            (R.from_quat(poses[:, 3:]) * current_rotation.inv()).magnitude()
        ) / np.pi
        landing_cost = np.linalg.norm(
            landing[:, :2] - np.array([x_target, y_target]), axis=1
        ) / max(config.landing_scale, 1e-6)
        reach_position_cost = np.linalg.norm(
            poses[:, :3] - current_pose[:3], axis=1
        ) / max(config.reach_position_scale, 1e-6)
        center_cost = np.linalg.norm(
            contact[None, :] - poses[:, :3], axis=1
        ) / max(config.paddle_center_scale, 1e-6)
        costs = (
            config.landing_weight * landing_cost
            + config.reach_position_weight * reach_position_cost
            + config.reach_orientation_weight * orientation_cost
            + config.paddle_center_weight * center_cost
            + config.reference_weight * float(candidate["reference_error"])
        )
        for index in np.flatnonzero(safe):
            feasible.append((
                float(costs[index]),
                contact,
                normals[index],
                landing[index],
                net_clearance[index],
                poses[index],
                candidate["reference_error"],
                candidate,
            ))

    if not feasible:
        return PlanResult(False, "no_safe_reflection_candidates")

    selected = min(feasible, key=lambda item: item[0])
    return PlanResult(
        selected=True,
        reason="optimized",
        contact_position_local=np.asarray(selected[1]),
        normal_local=np.asarray(selected[2]),
        predicted_landing_local=np.asarray(selected[3]),
        predicted_net_clearance=float(selected[4]),
        target_pose_local=np.asarray(selected[5]),
        reference_error=float(selected[6]),
        cost=float(selected[0]),
        feasible_candidates=len(feasible),
        sampled_contact_x=float(selected[1][0]),
        sampled_contact_height=float(selected[7]["contact_height"]),
        contact_time_after_bounce=float(selected[7]["contact_time_after_bounce"]),
        predicted_bounce_position_local=np.asarray(selected[7]["bounce_position"]),
        predicted_bounce_time=float(selected[7]["bounce_time"]),
        predicted_bounce_legal=True,
        post_bounce_velocity_local=np.asarray(selected[7]["post_bounce_velocity"]),
    )


def _common_candidate_metrics(
    contact: np.ndarray,
    pose: np.ndarray,
    landing: np.ndarray,
    net_clearance: float,
    candidate: dict[str, Any],
    current_pose: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    """Compute method-independent landing and executability metrics."""
    time_until_contact = float(candidate["bounce_time"]) + float(
        candidate["contact_time_after_bounce"]
    )
    position_change = float(np.linalg.norm(np.asarray(pose[:3]) - current_pose[:3]))
    orientation_change = float(
        (R.from_quat(np.asarray(pose[3:])) * R.from_quat(current_pose[3:]).inv()).magnitude()
    )
    landing_error = float(np.linalg.norm(np.asarray(landing[:2]) - target[:2]))
    return {
        "time_until_contact": time_until_contact,
        "position_change": position_change,
        "orientation_change_rad": orientation_change,
        "required_translation_speed": position_change / max(time_until_contact, 1e-8),
        "required_rotation_speed_rad_s": orientation_change / max(time_until_contact, 1e-8),
        "landing_error": landing_error,
        "net_clearance": float(net_clearance),
    }


def _common_candidate_cost(
    metrics: dict[str, Any],
    candidate: dict[str, Any],
    config: PlannerConfig,
) -> float:
    """Rank both candidate sources with one landing-first normalized cost."""
    landing_cost = metrics["landing_error"] / max(config.landing_scale, 1e-6)
    position_cost = metrics["position_change"] / max(config.reach_position_scale, 1e-6)
    orientation_cost = metrics["orientation_change_rad"] / np.pi
    translation_speed_cost = metrics["required_translation_speed"] / max(
        config.max_translation_speed, 1e-6
    )
    rotation_speed_cost = metrics["required_rotation_speed_rad_s"] / max(
        config.max_rotation_speed_rad_s, 1e-6
    )
    net_cost = max(
        0.0,
        1.0 - metrics["net_clearance"] / max(config.net_safety_scale, 1e-6),
    )
    return float(
        config.landing_weight * landing_cost
        + config.reach_position_weight * position_cost
        + config.reach_orientation_weight * orientation_cost
        + config.translation_speed_weight * translation_speed_cost
        + config.rotation_speed_weight * rotation_speed_cost
        + config.net_safety_weight * net_cost
        + config.paddle_center_weight * float(
            np.linalg.norm(np.asarray(candidate["contact"]) - np.asarray(candidate["pose"][:3]))
        ) / max(config.paddle_center_scale, 1e-6)
        + config.reference_weight * float(candidate["reference_error"])
    )


def _pool_rejection(
    diagnostics: list[dict[str, Any]],
    rejection_counts: dict[str, int],
    diagnostic: dict[str, Any],
    reason: str,
) -> None:
    diagnostic["selected"] = False
    diagnostic["rejection_reason"] = reason
    diagnostics.append(diagnostic)
    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1


def _generate_explicit_candidate_pool(
    candidates: list[dict[str, np.ndarray | float]],
    current_pose: np.ndarray,
    *,
    x_target: float,
    y_target: float,
    config: PlannerConfig,
    table_z: float,
    gravity: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int], dict[str, Any] | None]:
    """Generate valid analytic candidates without selecting one."""
    target = np.array([x_target, y_target, table_z], dtype=float)
    current_rotation = R.from_quat(current_pose[3:])
    current_normal = current_rotation.as_matrix()[:, 2]
    pool: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    best_rejected: dict[str, Any] | None = None

    def vector_value(value: np.ndarray | None) -> list[float] | None:
        return None if value is None else np.asarray(value, dtype=float).tolist()

    for candidate in candidates:
        contact = np.asarray(candidate["contact"], dtype=float)
        incoming = np.asarray(candidate["incoming_velocity"], dtype=float)
        flight_times = _solve_flight_times(contact, incoming, target, config, gravity)
        if not flight_times:
            _pool_rejection(
                diagnostics,
                rejection_counts,
                {
                    "source_method": "explicit",
                    "contact_position": vector_value(contact),
                    "contact_height": float(candidate["contact_height"]),
                    "time_until_contact": float(candidate["bounce_time"])
                    + float(candidate["contact_time_after_bounce"]),
                    "landing_error": None,
                    "candidate_cost": None,
                },
                "no_valid_flight_time_root",
            )
            continue

        for flight_time in flight_times:
            outgoing = _explicit_outgoing_velocity(contact, target, flight_time, gravity)
            normal_delta = incoming - outgoing
            normal_length = float(np.linalg.norm(normal_delta))
            diagnostic: dict[str, Any] = {
                "source_method": "explicit",
                "contact_position": vector_value(contact),
                "contact_height": float(candidate["contact_height"]),
                "time_until_contact": float(candidate["bounce_time"])
                + float(candidate["contact_time_after_bounce"]),
                "flight_time": float(flight_time),
                "current_paddle_normal": vector_value(current_normal),
                "required_paddle_normal": None,
                "landing_error": None,
                "net_clearance": None,
                "position_change": None,
                "orientation_change_rad": None,
                "required_translation_speed": None,
                "required_rotation_speed_rad_s": None,
                "candidate_cost": None,
                "selected": False,
            }
            if normal_length <= 1e-8:
                _pool_rejection(diagnostics, rejection_counts, diagnostic, "zero_normal")
                continue

            raw_normal = normal_delta / normal_length
            flipped_normal = -raw_normal
            raw_change = float(np.arccos(np.clip(np.dot(current_normal, raw_normal), -1.0, 1.0)))
            flipped_change = float(np.arccos(np.clip(np.dot(current_normal, flipped_normal), -1.0, 1.0)))
            normal = flipped_normal if flipped_change < raw_change else raw_normal
            normal_change = min(raw_change, flipped_change)
            diagnostic.update({
                "raw_solved_normal": vector_value(raw_normal),
                "flipped_solved_normal": vector_value(flipped_normal),
                "selected_normal": vector_value(normal),
                "required_paddle_normal": vector_value(normal),
                "raw_normal_change_rad": raw_change,
                "flipped_normal_change_rad": flipped_change,
                "selected_normal_change_rad": normal_change,
            })

            theta_pose = _target_pose(contact, normal)
            if theta_pose is None or not np.all(np.isfinite(theta_pose)):
                _pool_rejection(diagnostics, rejection_counts, diagnostic, "invalid_pose")
                continue
            minimum_pose_data = _minimum_normal_alignment_pose(theta_pose[:3], current_pose, normal)
            if minimum_pose_data is None:
                _pool_rejection(diagnostics, rejection_counts, diagnostic, "invalid_minimum_orientation_pose")
                continue
            pose, _, _ = minimum_pose_data
            reflected = incoming - 2.0 * float(np.dot(incoming, normal)) * normal
            reflection_residual = float(np.linalg.norm(reflected - outgoing))
            if reflection_residual > config.reflection_tolerance:
                _pool_rejection(diagnostics, rejection_counts, diagnostic, "reflection_residual")
                continue
            prediction = _projectile_prediction(contact, outgoing, target, gravity)
            if prediction is None:
                _pool_rejection(diagnostics, rejection_counts, diagnostic, "wrong_direction_or_flight")
                continue
            landing, net_clearance = prediction
            metrics = _common_candidate_metrics(
                contact, pose, landing, net_clearance, candidate, current_pose, target
            )
            diagnostic.update(metrics)
            diagnostic["predicted_landing"] = vector_value(landing)
            diagnostic["target_pose"] = vector_value(pose)
            diagnostic["selected_paddle_normal"] = vector_value(normal)
            diagnostic["reflection_residual"] = reflection_residual
            if metrics["net_clearance"] < config.net_safety_margin:
                _pool_rejection(diagnostics, rejection_counts, diagnostic, "net_clearance")
                continue
            if not (
                TABLE_SHIFT + config.table_x_safety_margin
                <= landing[0]
                <= TABLE_SHIFT + TABLE_LENGTH - config.table_x_safety_margin
                and abs(landing[1]) <= TABLE_HALF_WIDTH - config.table_y_safety_margin
            ):
                _pool_rejection(diagnostics, rejection_counts, diagnostic, "table_bounds")
                continue
            if metrics["position_change"] > config.max_position_change:
                _pool_rejection(diagnostics, rejection_counts, diagnostic, "static_position_limit")
                continue
            if metrics["orientation_change_rad"] > config.max_orientation_change_rad:
                _pool_rejection(diagnostics, rejection_counts, diagnostic, "static_orientation_limit")
                continue
            if metrics["required_translation_speed"] > config.hard_translation_speed:
                _pool_rejection(diagnostics, rejection_counts, diagnostic, "translation_speed_limit")
                continue
            if metrics["required_rotation_speed_rad_s"] > config.hard_rotation_speed_rad_s:
                _pool_rejection(diagnostics, rejection_counts, diagnostic, "rotation_speed_limit")
                continue

            record = {
                **metrics,
                "source_method": "explicit",
                "contact": contact.copy(),
                "normal": normal.copy(),
                "landing": np.asarray(landing, dtype=float).copy(),
                "pose": np.asarray(pose, dtype=float).copy(),
                "reference_error": float(candidate["reference_error"]),
                "contact_height": float(candidate["contact_height"]),
                "contact_time_after_bounce": float(candidate["contact_time_after_bounce"]),
                "bounce_position": np.asarray(candidate["bounce_position"], dtype=float).copy(),
                "bounce_time": float(candidate["bounce_time"]),
                "post_bounce_velocity": np.asarray(candidate["post_bounce_velocity"], dtype=float).copy(),
                "flight_time": float(flight_time),
                "outgoing": outgoing.copy(),
                "reflection_residual": reflection_residual,
                "diagnostic": diagnostic,
            }
            diagnostic["candidate_cost"] = _common_candidate_cost(record, record, config)
            pool.append(record)

    for diagnostic in diagnostics:
        if diagnostic.get("rejection_reason") and diagnostic.get("candidate_cost") is not None:
            if best_rejected is None or diagnostic["candidate_cost"] < best_rejected.get("candidate_cost", float("inf")):
                best_rejected = diagnostic
    return pool, diagnostics, rejection_counts, best_rejected


def _generate_grid_candidate_pool(
    candidates: list[dict[str, np.ndarray | float]],
    current_pose: np.ndarray,
    *,
    x_target: float,
    y_target: float,
    config: PlannerConfig,
    table_z: float,
    gravity: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Generate valid grid-reflection candidates without selecting one."""
    target = np.array([x_target, y_target, table_z], dtype=float)
    normals = _normal_candidates(config)
    current_pose = np.asarray(current_pose, dtype=float)
    pool: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    for candidate in candidates:
        contact = np.asarray(candidate["contact"], dtype=float)
        incoming = np.asarray(candidate["incoming_velocity"], dtype=float)
        reflected = incoming[None, :] - 2.0 * np.sum(incoming[None, :] * normals, axis=1)[:, None] * normals
        poses = _target_poses(contact, normals)
        for index in range(len(normals)):
            prediction = _reflection_prediction(contact, incoming, normals[index], table_z, gravity)
            diagnostic: dict[str, Any] = {
                "source_method": "grid",
                "contact_position": contact.tolist(),
                "contact_height": float(candidate["contact_height"]),
                "time_until_contact": float(candidate["bounce_time"])
                + float(candidate["contact_time_after_bounce"]),
                "normal": normals[index].astype(float).tolist(),
                "selected": False,
            }
            if prediction is None:
                _pool_rejection(diagnostics, rejection_counts, diagnostic, "wrong_direction_or_flight")
                continue
            landing, net_clearance = prediction
            metrics = _common_candidate_metrics(
                contact, poses[index], landing, net_clearance, candidate, current_pose, target
            )
            diagnostic.update(metrics)
            diagnostic["predicted_landing"] = landing.astype(float).tolist()
            diagnostic["target_pose"] = poses[index].astype(float).tolist()
            if metrics["net_clearance"] < config.net_safety_margin:
                _pool_rejection(diagnostics, rejection_counts, diagnostic, "net_clearance")
                continue
            if not (
                TABLE_SHIFT + config.table_x_safety_margin
                <= landing[0]
                <= TABLE_SHIFT + TABLE_LENGTH - config.table_x_safety_margin
                and abs(landing[1]) <= TABLE_HALF_WIDTH - config.table_y_safety_margin
            ):
                _pool_rejection(diagnostics, rejection_counts, diagnostic, "table_bounds")
                continue
            if metrics["position_change"] > config.max_position_change:
                _pool_rejection(diagnostics, rejection_counts, diagnostic, "static_position_limit")
                continue
            if metrics["orientation_change_rad"] > config.max_orientation_change_rad:
                _pool_rejection(diagnostics, rejection_counts, diagnostic, "static_orientation_limit")
                continue
            if metrics["required_translation_speed"] > config.hard_translation_speed:
                _pool_rejection(diagnostics, rejection_counts, diagnostic, "translation_speed_limit")
                continue
            if metrics["required_rotation_speed_rad_s"] > config.hard_rotation_speed_rad_s:
                _pool_rejection(diagnostics, rejection_counts, diagnostic, "rotation_speed_limit")
                continue
            record = {
                **metrics,
                "source_method": "grid",
                "contact": contact.copy(),
                "normal": normals[index].copy(),
                "landing": landing.copy(),
                "pose": poses[index].copy(),
                "reference_error": float(candidate["reference_error"]),
                "contact_height": float(candidate["contact_height"]),
                "contact_time_after_bounce": float(candidate["contact_time_after_bounce"]),
                "bounce_position": np.asarray(candidate["bounce_position"], dtype=float).copy(),
                "bounce_time": float(candidate["bounce_time"]),
                "post_bounce_velocity": np.asarray(candidate["post_bounce_velocity"], dtype=float).copy(),
                "flight_time": None,
                "outgoing": reflected[index].copy(),
                "reflection_residual": 0.0,
                "diagnostic": diagnostic,
            }
            diagnostic["candidate_cost"] = _common_candidate_cost(record, record, config)
            pool.append(record)
    return pool, diagnostics, rejection_counts


def _select_common_candidate_pool(
    explicit_pool: list[dict[str, Any]],
    grid_pool: list[dict[str, Any]],
    explicit_diagnostics: list[dict[str, Any]],
    grid_diagnostics: list[dict[str, Any]],
    explicit_rejections: dict[str, int],
    grid_rejections: dict[str, int],
    best_rejected: dict[str, Any] | None,
    bounce: dict[str, Any],
) -> PlanResult:
    """Select the best executable record irrespective of generation method."""
    pool = explicit_pool + grid_pool
    pool_diagnostics = [record["diagnostic"] for record in pool]
    for diagnostic in pool_diagnostics:
        diagnostic["selected"] = False
    if not pool:
        reasons = sorted(set(explicit_rejections) | set(grid_rejections))
        reason = "common_candidate_pool_empty"
        if reasons:
            reason += ":" + ",".join(reasons)
        return PlanResult(
            False,
            reason,
            solution_method="",
            explicit_candidate_diagnostics=explicit_diagnostics,
            grid_candidate_diagnostics=grid_diagnostics,
            candidate_pool_diagnostics=pool_diagnostics,
            explicit_candidate_count=len(explicit_pool),
            grid_candidate_count=len(grid_pool),
            rejected_candidate_count=sum(explicit_rejections.values()) + sum(grid_rejections.values()),
            best_rejected_explicit_candidate=best_rejected,
            predicted_bounce_position_local=np.asarray(bounce["position"], dtype=float),
            predicted_bounce_time=float(bounce["time"]),
            predicted_bounce_legal=True,
            post_bounce_velocity_local=np.asarray(bounce["post_velocity"], dtype=float),
        )

    selected = min(pool, key=lambda record: float(record["diagnostic"]["candidate_cost"]))
    selected["diagnostic"]["selected"] = True
    method = selected["source_method"]
    return PlanResult(
        True,
        "optimized_common_pool",
        contact_position_local=selected["contact"],
        normal_local=selected["normal"],
        predicted_landing_local=selected["landing"],
        predicted_net_clearance=float(selected["net_clearance"]),
        target_pose_local=selected["pose"],
        reference_error=float(selected["reference_error"]),
        cost=float(selected["diagnostic"]["candidate_cost"]),
        feasible_candidates=len(pool),
        sampled_contact_x=float(selected["contact"][0]),
        sampled_contact_height=float(selected["contact_height"]),
        contact_time_after_bounce=float(selected["contact_time_after_bounce"]),
        predicted_bounce_position_local=np.asarray(bounce["position"], dtype=float),
        predicted_bounce_time=float(bounce["time"]),
        predicted_bounce_legal=True,
        post_bounce_velocity_local=np.asarray(bounce["post_velocity"], dtype=float),
        solved_flight_time=selected["flight_time"],
        solved_outgoing_velocity=selected["outgoing"],
        reflection_residual=float(selected["reflection_residual"]),
        executability_cost=float(selected["diagnostic"]["candidate_cost"]),
        solution_method=method,
        fallback_grid_search=False,
        explicit_candidate_diagnostics=explicit_diagnostics,
        grid_candidate_diagnostics=grid_diagnostics,
        candidate_pool_diagnostics=pool_diagnostics,
        explicit_candidate_count=len(explicit_pool),
        grid_candidate_count=len(grid_pool),
        rejected_candidate_count=sum(explicit_rejections.values()) + sum(grid_rejections.values()),
        selected_landing_error=float(selected["landing_error"]),
        selected_time_until_contact=float(selected["time_until_contact"]),
        selected_position_change=float(selected["position_change"]),
        selected_orientation_change_rad=float(selected["orientation_change_rad"]),
        selected_translation_speed=float(selected["required_translation_speed"]),
        selected_rotation_speed_rad_s=float(selected["required_rotation_speed_rad_s"]),
        best_rejected_explicit_candidate=best_rejected,
    )


def plan_optimized(
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    current_pose: np.ndarray,
    *,
    x_target: float,
    y_target: float,
    config: PlannerConfig,
    bounce_factor: float = 1.0,
    table_z: float = TABLE_Z,
    gravity: float = GRAVITY,
    x_dis: float = -0.3,
    z_dis: float = 0.9,
    vx_dis: float = 4.5,
    vy_dis: float = 2.0,
) -> PlanResult:
    """Generate explicit and grid candidates, then select from one pool."""
    config.validate()
    planner_start = time.perf_counter()
    ball_pos = np.asarray(ball_pos, dtype=float)
    ball_vel = np.asarray(ball_vel, dtype=float)
    current_pose = np.asarray(current_pose, dtype=float)
    # ``bounce_factor`` remains in the public interface, but the simulator's
    # table collision is an elastic z-velocity flip, so optimized planning
    # follows that physics exactly.
    bounce, candidates = _post_bounce_height_candidates(
        ball_pos,
        ball_vel,
        config,
        table_z,
        gravity,
        x_dis,
        z_dis,
        vx_dis,
        vy_dis,
    )
    bounce_position = bounce.get("position")
    post_bounce_velocity = bounce.get("post_velocity")
    bounce_kwargs = {
        "predicted_bounce_position_local": (
            None if bounce_position is None else np.asarray(bounce_position, dtype=float)
        ),
        "predicted_bounce_time": bounce.get("time"),
        "predicted_bounce_legal": bool(bounce.get("legal", False)),
        "post_bounce_velocity_local": (
            None if post_bounce_velocity is None else np.asarray(post_bounce_velocity, dtype=float)
        ),
        "pre_bounce_candidate_attempted": False,
    }
    if bounce.get("reason") == "receiver_bounce_out_of_bounds":
        return PlanResult(
            False,
            "predicted_receiver_bounce_out_of_bounds",
            planner_time=time.perf_counter() - planner_start,
            **bounce_kwargs,
        )
    if not bounce.get("legal", False):
        return PlanResult(
            False,
            str(bounce.get("reason", "no_predicted_receiver_bounce")),
            planner_time=time.perf_counter() - planner_start,
            **bounce_kwargs,
        )
    if not candidates:
        return PlanResult(
            False,
            "no_post_bounce_contact_candidates",
            analytic_solve_time=0.0,
            planner_time=time.perf_counter() - planner_start,
            **bounce_kwargs,
        )
    analytic_start = time.perf_counter()
    explicit_pool, explicit_diagnostics, explicit_rejections, best_rejected = (
        _generate_explicit_candidate_pool(
            candidates,
            current_pose,
            x_target=x_target,
            y_target=y_target,
            config=config,
            table_z=table_z,
            gravity=gravity,
        )
    )
    analytic_elapsed = time.perf_counter() - analytic_start
    grid_pool, grid_diagnostics, grid_rejections = _generate_grid_candidate_pool(
        candidates,
        current_pose,
        x_target=x_target,
        y_target=y_target,
        config=config,
        table_z=table_z,
        gravity=gravity,
    )
    result = _select_common_candidate_pool(
        explicit_pool,
        grid_pool,
        explicit_diagnostics,
        grid_diagnostics,
        explicit_rejections,
        grid_rejections,
        best_rejected,
        bounce,
    )
    result.analytic_solve_time = analytic_elapsed
    result.planner_time = time.perf_counter() - planner_start
    result.pre_bounce_candidate_attempted = False
    return result


def estimate_legacy(
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    *,
    x_target: float,
    y_target: float,
    bounce_factor: float = 1.0,
    table_z: float = TABLE_Z,
    gravity: float = GRAVITY,
    x_dis: float = -0.3,
    z_dis: float = 0.9,
    vx_dis: float = 4.5,
    vy_dis: float = 2.0,
    legality_filter: bool = False,
    net_safety_margin: float = 0.03,
    table_x_safety_margin: float = 0.03,
    table_y_safety_margin: float = 0.03,
) -> PlanResult:
    """Reconstruct legacy's selected point/normal for diagnostics only.

    This function never writes an environment target.  It mirrors the old
    100-point contact and 100x100-normal argmins so legacy execution itself
    remains delegated to the original method.
    """
    config = PlannerConfig(
        planner="legacy_legal" if legality_filter else "legacy",
        contact_candidates=100,
        normal_candidates=10000,
        reference_scale=1.0,
        net_safety_margin=net_safety_margin,
        table_x_safety_margin=table_x_safety_margin,
        table_y_safety_margin=table_y_safety_margin,
    )
    candidates = _incoming_candidates(
        np.asarray(ball_pos, dtype=float),
        np.asarray(ball_vel, dtype=float),
        config,
        bounce_factor,
        table_z,
        gravity,
        x_dis,
        z_dis,
        vx_dis,
        vy_dis,
    )
    if not candidates:
        return PlanResult(
            False,
            "no_legacy_contact_candidate",
            legacy_legality_filter_applied=bool(legality_filter),
        )
    contact_candidate = min(candidates, key=lambda item: float(item["reference_error"]))
    contact = np.asarray(contact_candidate["contact"])
    velocity = np.asarray(contact_candidate["incoming_velocity"])
    normals = _normal_candidates(config)
    reflected = velocity[None, :] - 2.0 * np.sum(velocity[None, :] * normals, axis=1)[:, None] * normals
    valid_direction = reflected[:, 0] > 1e-8
    t_net = np.divide(
        TABLE_SHIFT - contact[0],
        reflected[:, 0],
        out=np.full(len(normals), np.nan),
        where=valid_direction,
    )
    net_z = contact[2] + reflected[:, 2] * t_net + 0.5 * gravity * t_net * t_net
    vz_hit = -np.sqrt(np.maximum(-2 * gravity * max(contact[2] - table_z, 0.0) + reflected[:, 2] ** 2, 0.0))
    t_hit = (vz_hit - reflected[:, 2]) / gravity
    valid = valid_direction & (t_net > 0) & (t_hit > 0) & np.isfinite(t_hit)
    landings = np.column_stack([
        contact[0] + reflected[:, 0] * t_hit,
        contact[1] + reflected[:, 1] * t_hit,
        np.full(len(normals), table_z),
    ])
    net_legal = valid & (net_z >= NET_TOP_Z + config.net_safety_margin)
    table_legal = net_legal & (
        (landings[:, 0] >= TABLE_SHIFT + config.table_x_safety_margin)
        & (landings[:, 0] <= TABLE_SHIFT + TABLE_LENGTH - config.table_x_safety_margin)
        & (np.abs(landings[:, 1]) <= TABLE_HALF_WIDTH - config.table_y_safety_margin)
    )
    costs = (landings[:, 0] - x_target) ** 2 + (landings[:, 1] - y_target) ** 2
    original_costs = costs.copy()
    original_costs[~valid] = np.inf
    selection_costs = original_costs.copy()
    fallback_to_original = False
    if legality_filter:
        if np.any(table_legal):
            selection_costs[~table_legal] = np.inf
        else:
            # Preserve the original legacy result when legality filtering
            # would otherwise remove every normal candidate.
            fallback_to_original = True
    best_index = int(np.argmin(selection_costs))
    if not np.isfinite(selection_costs[best_index]):
        return PlanResult(
            False,
            "no_legacy_normal_candidate",
            legacy_normal_candidate_count=len(normals),
            legacy_legal_after_net_count=int(np.count_nonzero(net_legal)),
            legacy_legal_after_table_count=int(np.count_nonzero(table_legal)),
            legacy_legality_filter_applied=bool(legality_filter),
            legacy_legality_filter_fallback=bool(fallback_to_original),
        )
    normal = normals[best_index]
    pose = _target_pose(contact, normal)
    if pose is None:
        return PlanResult(
            False,
            "invalid_legacy_pose",
            legacy_normal_candidate_count=len(normals),
            legacy_legal_after_net_count=int(np.count_nonzero(net_legal)),
            legacy_legal_after_table_count=int(np.count_nonzero(table_legal)),
            legacy_legality_filter_applied=bool(legality_filter),
            legacy_legality_filter_fallback=bool(fallback_to_original),
        )
    return PlanResult(
        selected=True,
        reason=(
            "legacy_legal_fallback_original"
            if fallback_to_original
            else ("legacy_legal" if legality_filter else "legacy_estimate")
        ),
        contact_position_local=contact,
        normal_local=np.asarray(normal),
        predicted_landing_local=np.asarray(landings[best_index]),
        predicted_net_clearance=float(net_z[best_index] - NET_TOP_Z),
        target_pose_local=np.asarray(pose),
        reference_error=float(contact_candidate["reference_error"]),
        cost=float(selection_costs[best_index]),
        feasible_candidates=len(candidates),
        legacy_normal_candidate_count=len(normals),
        legacy_legal_after_net_count=int(np.count_nonzero(net_legal)),
        legacy_legal_after_table_count=int(np.count_nonzero(table_legal)),
        legacy_legality_filter_applied=bool(legality_filter),
        legacy_legality_filter_fallback=bool(fallback_to_original),
        selected_predicted_net_crossing_height=float(net_z[best_index]),
        selected_landing_error=float(
            np.hypot(landings[best_index, 0] - x_target, landings[best_index, 1] - y_target)
        ),
    )


def mirror_position_to_world(local_position: np.ndarray) -> np.ndarray:
    position = np.asarray(local_position, dtype=float).copy()
    position[0] = 2 * TABLE_SHIFT - position[0]
    position[1] = -position[1]
    return position


def mirror_pose_to_world(local_pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(local_pose, dtype=float).copy()
    pose[:3] = mirror_position_to_world(pose[:3])
    mirror = np.diag([-1.0, -1.0, 1.0])
    pose[3:] = R.from_matrix(mirror @ R.from_quat(pose[3:]).as_matrix()).as_quat()
    return pose


_P2_RACKET_ORIENTATION_MIRROR = np.diag([-1.0, -1.0, 1.0])


def racket_pose_p2_world_to_local(world_pose: np.ndarray) -> np.ndarray:
    """Map the raw P2 racket body pose into the P1-like planner frame.

    The opponent robot chain is already positioned in the same x/y racket
    frame as P1 in the MuJoCo model.  Only its orientation is represented in
    the opponent body convention and needs the established 180-degree x/y
    rotation.  This is intentionally separate from the ball/table transform.
    """
    pose = np.asarray(world_pose, dtype=float).copy()
    pose[3:] = R.from_matrix(
        _P2_RACKET_ORIENTATION_MIRROR @ R.from_quat(pose[3:]).as_matrix()
    ).as_quat()
    return pose


def racket_pose_p2_local_to_world(local_pose: np.ndarray) -> np.ndarray:
    """Inverse of :func:`racket_pose_p2_world_to_local` for raw body diagnostics."""
    pose = np.asarray(local_pose, dtype=float).copy()
    pose[3:] = R.from_matrix(
        _P2_RACKET_ORIENTATION_MIRROR.T @ R.from_quat(pose[3:]).as_matrix()
    ).as_quat()
    return pose


def debug_racket_pose_symmetry(*, verbose: bool = True) -> list[dict[str, Any]]:
    """Verify P1/P2 racket executability in the model's pose conventions."""
    p1_rotation = R.from_euler("xyz", [0.12, -0.18, 0.27])
    target_rotation = p1_rotation * R.from_euler("xyz", [0.05, 0.02, -0.03])
    p1_world = np.array(
        [-0.24, -0.01, 1.22, *p1_rotation.as_quat()],
        dtype=float,
    )
    p1_target_local = np.array(
        [-0.08, 0.08, 1.16, *target_rotation.as_quat()],
        dtype=float,
    )

    # This is the mirrored robot state represented by the current MuJoCo
    # opponent chain: position is already in the P1-like frame, orientation
    # uses the opponent-body convention.
    p2_world = p1_world.copy()
    p2_world[3:] = R.from_matrix(
        _P2_RACKET_ORIENTATION_MIRROR @ p1_rotation.as_matrix()
    ).as_quat()
    p2_target_local = p1_target_local.copy()

    rows: list[dict[str, Any]] = []
    for player, raw_world, target_local, to_local, to_world in (
        (1, p1_world, p1_target_local, lambda pose: np.asarray(pose, dtype=float).copy(), lambda pose: np.asarray(pose, dtype=float).copy()),
        (2, p2_world, p2_target_local, racket_pose_p2_world_to_local, racket_pose_p2_local_to_world),
    ):
        local_pose = to_local(raw_world)
        target_world = to_world(target_local)
        position_change = float(np.linalg.norm(target_local[:3] - local_pose[:3]))
        orientation_change = float(
            (R.from_quat(target_local[3:]) * R.from_quat(local_pose[3:]).inv()).magnitude()
        )
        round_trip = to_local(target_world)
        if not np.allclose(round_trip, target_local, atol=1e-10):
            raise AssertionError(f"P{player} racket pose transform is not invertible")
        row = {
            "player": player,
            "raw_world_pose": raw_world.tolist(),
            "planner_local_pose": local_pose.tolist(),
            "planner_local_target_pose": target_local.tolist(),
            "position_change": position_change,
            "orientation_change_rad": orientation_change,
        }
        rows.append(row)
        if verbose:
            print(f"P{player} raw world pose: {np.asarray(row['raw_world_pose'])}")
            print(f"P{player} planner-local pose: {np.asarray(row['planner_local_pose'])}")
            print(f"P{player} planner-local target: {np.asarray(row['planner_local_target_pose'])}")
            print(
                f"P{player} position_change={position_change:.6f} "
                f"orientation_change={orientation_change:.6f} rad"
            )

    if not np.isclose(rows[0]["position_change"], rows[1]["position_change"], atol=1e-10):
        raise AssertionError("P1/P2 position executability is asymmetric")
    if not np.isclose(rows[0]["orientation_change_rad"], rows[1]["orientation_change_rad"], atol=1e-10):
        raise AssertionError("P1/P2 orientation executability is asymmetric")
    return rows


def mirror_velocity_to_local(world_velocity: np.ndarray) -> np.ndarray:
    velocity = np.asarray(world_velocity, dtype=float).copy()
    velocity[0] = -velocity[0]
    velocity[1] = -velocity[1]
    return velocity


def actual_landing_prediction(
    ball_pos_local: np.ndarray,
    ball_vel_local: np.ndarray,
    *,
    table_z: float = TABLE_Z,
    gravity: float = GRAVITY,
) -> tuple[np.ndarray, float] | None:
    """Apply the same landing calculation to the realized post-contact state."""
    if ball_vel_local[0] <= 1e-8:
        return None
    vz_hit = -np.sqrt(max(-2 * gravity * max(ball_pos_local[2] - table_z, 0.0) + ball_vel_local[2] ** 2, 0.0))
    t_hit = (vz_hit - ball_vel_local[2]) / gravity
    if t_hit <= 0:
        return None
    landing = np.array([
        ball_pos_local[0] + ball_vel_local[0] * t_hit,
        ball_pos_local[1] + ball_vel_local[1] * t_hit,
        table_z,
    ], dtype=float)
    t_net = (TABLE_SHIFT - ball_pos_local[0]) / ball_vel_local[0]
    if t_net <= 0:
        return landing, float("-inf")
    net_z = ball_pos_local[2] + ball_vel_local[2] * t_net + 0.5 * gravity * t_net * t_net
    return landing, float(net_z - NET_TOP_Z)


def debug_explicit_solver(*, verbose: bool = True) -> list[dict[str, Any]]:
    """Exercise explicit planning against representative current-profile skills."""
    # The harness focuses on ballistics/reflection.  Use generous reach
    # limits here so this synthetic pose does not turn the physics check into
    # a PPO executability test; normal PlannerConfig defaults remain empirical.
    config = PlannerConfig(
        planner="optimized",
        contact_candidates=21,
        normal_candidates=25,
        max_orientation_change_rad=2.0,
        max_translation_speed=10.0,
        max_rotation_speed_rad_s=10.0,
        hard_translation_speed=10.0,
        hard_rotation_speed_rad_s=10.0,
    )
    current_pose = np.array([-0.4, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0], dtype=float)
    skill_targets = {
        "left": (2.19, -0.38),
        "left_short": (1.95, -0.38),
        "center_safe": (2.0, 0.0),
        "right_short": (1.95, 0.38),
        "right": (2.19, 0.38),
    }
    cases = [
        (np.array([1.59225, -0.18511, 1.12778]), np.array([-2.94529, -0.20961, 0.80126]), "left"),
        (np.array([2.10265, -0.14711, 1.08589]), np.array([-3.9, 0.74297, 1.30662]), "center_safe"),
        (np.array([1.98246, -0.24387, 1.06869]), np.array([-3.41455, -0.10272, 1.03062]), "right"),
    ]
    results: list[dict[str, Any]] = []
    for ball_pos, ball_vel, skill in cases:
        x_target, y_target = skill_targets[skill]
        result = plan_optimized(
            ball_pos,
            ball_vel,
            current_pose,
            x_target=x_target,
            y_target=y_target,
            config=config,
        )
        if not result.selected or result.solution_method != "explicit":
            raise AssertionError(f"explicit solver did not select {skill}: {result.reason}")
        if result.predicted_bounce_legal is not True:
            raise AssertionError(f"predicted receiver bounce was not legal for {skill}")
        if result.pre_bounce_candidate_attempted:
            raise AssertionError("optimized planner attempted a pre-bounce contact")
        if result.contact_time_after_bounce is None or result.contact_time_after_bounce <= 0:
            raise AssertionError(f"invalid post-bounce contact time for {skill}")
        landing_error = float(
            np.linalg.norm(result.predicted_landing_local[:2] - np.array([x_target, y_target]))
        )
        if landing_error > config.explicit_landing_tolerance:
            raise AssertionError(f"landing mismatch for {skill}: {landing_error}")
        if result.reflection_residual is None or result.reflection_residual > config.reflection_tolerance:
            raise AssertionError(f"reflection residual too large for {skill}: {result.reflection_residual}")
        if result.predicted_net_clearance is None or result.predicted_net_clearance < config.net_safety_margin:
            raise AssertionError(f"net clearance failed for {skill}: {result.predicted_net_clearance}")
        selected_diagnostics = [
            item for item in result.candidate_pool_diagnostics if item.get("selected")
        ]
        if len(selected_diagnostics) != 1:
            raise AssertionError(f"expected one selected explicit diagnostic for {skill}")
        selected_diagnostic = selected_diagnostics[0]
        results.append(
            {
                "skill": skill,
                "landing_error": landing_error,
                "reflection_residual": result.reflection_residual,
                "net_clearance": result.predicted_net_clearance,
                "flight_time": result.solved_flight_time,
                "bounce_position": result.predicted_bounce_position_local.tolist(),
                "contact_time_after_bounce": result.contact_time_after_bounce,
                "normal_change_rad": selected_diagnostic.get("normal_change_rad", ""),
                "theta_orientation_change_rad": selected_diagnostic.get("theta_orientation_change_rad", ""),
                "minimum_orientation_change_rad": selected_diagnostic.get("minimum_orientation_change_rad", ""),
            }
        )
        if verbose:
            print(
                f"{skill}: landing_error={landing_error:.3e} "
                f"reflection_residual={result.reflection_residual:.3e} "
                f"net_clearance={result.predicted_net_clearance:.3f} "
                f"T={result.solved_flight_time:.3f} "
                f"bounce_x={result.predicted_bounce_position_local[0]:.3f} "
                f"contact_after_bounce={result.contact_time_after_bounce:.3f} "
                f"method={result.solution_method} "
                f"landing_error={result.selected_landing_error:.3e}"
            )
    no_return = plan_optimized(
        np.array([1.5, 0.0, 1.0]),
        np.array([-6.0, 0.0, 0.0]),
        current_pose,
        x_target=2.0,
        y_target=0.0,
        config=config,
    )
    if no_return.reason != "predicted_receiver_bounce_out_of_bounds":
        raise AssertionError(f"illegal bounce was not rejected: {no_return.reason}")
    if no_return.pre_bounce_candidate_attempted or no_return.predicted_bounce_legal:
        raise AssertionError("illegal bounce produced a pre-bounce return candidate")
    return results


if __name__ == "__main__":
    debug_explicit_solver()
    debug_racket_pose_symmetry()
