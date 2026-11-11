"""Conservative extraction from the published LATTE-MV reconstruction frame.

Geometry/indices are established in the official reconstruction and loader
source, not estimated from contact flags.  Classification is a provisional
nearest-V3-target proxy, not a recognized human action annotation.
"""
from dataclasses import dataclass
import numpy as np
from v3.skill_semantics import skill_target_xy
from nash_skills.skills import SKILL_NAMES, TABLE_SHIFT

UNIT_M = .003048
HALF_LENGTH = 450 * UNIT_M
HALF_WIDTH = 250 * UNIT_M
TABLE_Z = 250 * UNIT_M
NET_TOP = TABLE_Z + .1524  # Nominal 15.24 cm net; source visualization has inconsistent net drawing.
ROBOT_HALF_LENGTH = 1.37  # mujoco_env_comp table_top geom size
ROBOT_HALF_WIDTH = .7625


@dataclass
class Config:
    center_normalized: float = .05 / ROBOT_HALF_WIDTH
    position_time: str = 'crossing'
    landmark: str = 'feet_midpoint'
    robot_profile: str = 'aggressive'
    edge_margin_m: float = .005
    min_net_clearance_m: float = 0.0
    net_height_m: float = .1524


def starts(flags):
    active = np.asarray(flags) > .5
    return np.flatnonzero(active & ~np.r_[False, active[:-1]])


def orientation(player):
    if player not in (1, 2):
        raise ValueError('Player must be 1 or 2')
    return 1 if player == 1 else -1


def position_bin(world_y_m, player, center_normalized):
    lateral = orientation(player) * world_y_m / HALF_WIDTH
    return 'center' if abs(lateral) <= center_normalized else 'left' if lateral > 0 else 'right'


def normalized_landing(xy_m, player):
    return orientation(player) * np.asarray(xy_m) / np.array([HALF_LENGTH, HALF_WIDTH])


def target_prototypes(profile):
    return np.array([[(skill_target_xy(s, profile)[0] - TABLE_SHIFT) / ROBOT_HALF_LENGTH,
                      skill_target_xy(s, profile)[1] / ROBOT_HALF_WIDTH] for s in SKILL_NAMES])


def landing_category(depth_lateral, profile='aggressive'):
    xy = np.asarray(depth_lateral)
    if not np.isfinite(xy).all() or not (0 < xy[0] < 1 and abs(xy[1]) < 1):
        return None
    return SKILL_NAMES[int(np.argmin(np.sum((target_prototypes(profile) - xy) ** 2, axis=1)))]


def body_positions(scene, landmark):
    positions = {}
    for player, offset in ((1, 2), (2, 46)):
        positions[player] = (scene[:, offset + 8, :] if landmark == 'pelvis'
                             else np.mean(scene[:, [offset + 20, offset + 23], :], axis=1)) * UNIT_M
    return positions


def extract(scene, rally_id, config):
    if scene.ndim != 3 or scene.shape[1:] != (91, 3):
        raise ValueError(f'Unexpected scene shape: {scene.shape}')
    if not np.isin(scene[:, 1], [0, 1]).all():
        raise ValueError('Nonbinary contact indicators')
    ball = scene[:, 90] * UNIT_M
    positions = body_positions(scene, config.landmark)
    for player in (1, 2):
        if not np.isfinite(positions[player]).all() or orientation(player) * np.median(positions[player][:, 0]) >= 0:
            raise ValueError('Player side/pose inconsistent with official P1-negative-x/P2-positive-x frame')
    hits = sorted((int(t), p) for p in (1, 2) for t in starts(scene[:, 1, p-1]))
    bounces = starts(scene[:, 1, 2])
    rows = []
    for i, (contact, player) in enumerate(hits):
        end = hits[i+1][0] if i+1 < len(hits) else len(scene)-1
        row = dict(rally_id=rally_id, shot_index=i, attacker_id=player, defender_id=3-player,
                   contact_frame=contact, contact_seconds=contact/scene[0, 0, 0], interval_end_frame=end,
                   crossing_frame=np.nan, bounce_frame=np.nan, valid_net_crossing=False,
                   valid_opponent_landing=False, classified=False, human_skill_proxy='',
                   classification_status='provisional_nearest_v3_target', exclusion_reason='')
        def reject(reason):
            row['exclusion_reason'] = reason
            rows.append(row)
        for p in (1,2):
            if contact > 0 and np.isfinite(positions[p][contact-1]).all():
                for k, axis in enumerate('xyz'):
                    row[f'p{p}_pre_contact_{axis}_m'] = float(positions[p][contact-1,k])
        if contact < 1 or contact+1 >= len(ball) or end <= contact+1:
            reject('contact_context_unavailable'); continue
        if np.sum(scene[contact,1,:2]) != 1:
            reject('simultaneous_racket_contacts'); continue
        if i+1 < len(hits) and hits[i+1][1] == player:
            reject('consecutive_same_player_contacts_ambiguous'); continue
        segment = ball[contact:end+1]
        if not np.isfinite(ball[contact-1:contact+2]).all():
            reject('missing_ball_at_contact'); continue
        incoming = (ball[contact]-ball[contact-1])*scene[0,0,0]
        outgoing = (ball[contact+1]-ball[contact])*scene[0,0,0]
        row.update(incoming_vx_mps=float(incoming[0]), outgoing_vx_mps=float(outgoing[0]),
                   contact_velocity_change_mps=float(np.linalg.norm(outgoing-incoming)))
        sign = orientation(player)
        if sign*ball[contact,0] >= 0:
            reject('contact_on_wrong_table_side'); continue
        # Serve intervals can contain a self-half bounce and differ from robot returns.
        bs = [int(t) for t in bounces if contact < t < end]
        if any(np.isfinite(ball[t]).all() and sign*ball[t,0] <= 0 for t in bs):
            reject('self_half_bounce_or_serve'); continue
        if sign*incoming[0] >= 0 or sign*outgoing[0] <= 0:
            reject('no_plausible_longitudinal_contact_reversal'); continue
        crossings = []
        for t in range(contact, end):
            if np.isfinite(ball[t:t+2]).all() and sign*ball[t,0] <= 0 < sign*ball[t+1,0]:
                fraction = -ball[t,0] / (ball[t+1,0]-ball[t,0])
                crossings.append((t, fraction, ball[t] + fraction*(ball[t+1]-ball[t])))
        if len(crossings) != 1:
            reject('missing_or_ambiguous_directed_net_crossing'); continue
        t, fraction, cross = crossings[0]
        row.update(crossing_frame=float(t+fraction), crossing_seconds=float((t+fraction)/scene[0,0,0]),
                   crossing_y_m=float(cross[1]), crossing_z_m=float(cross[2]))
        for p in (1,2):
            pos = positions[p][t] + fraction*(positions[p][t+1]-positions[p][t])
            for k, axis in enumerate('xyz'):
                row[f'p{p}_crossing_{axis}_m'] = float(pos[k])
        if abs(cross[1]) > HALF_WIDTH or cross[2] < TABLE_Z + config.net_height_m + config.min_net_clearance_m:
            reject('crossing_outside_net_span_or_below_net'); continue
        row['valid_net_crossing'] = True
        after = [b for b in bs if b > t+fraction]
        if len(after) != 1:
            reject('missing_or_multiple_post_crossing_bounces'); continue
        bounce = after[0]
        row['bounce_frame'] = bounce
        if not np.isfinite(ball[contact:bounce+2]).all():
            reject('missing_ball_in_hit_crossing_bounce_path'); continue
        xy = normalized_landing(ball[bounce,:2], player)
        row.update(landing_world_x_m=float(ball[bounce,0]), landing_world_y_m=float(ball[bounce,1]),
                   landing_depth_normalized=float(xy[0]), landing_lateral_normalized=float(xy[1]))
        if not (0 < xy[0] <= 1 and abs(xy[1]) <= 1) or abs(ball[bounce,2]-TABLE_Z) > .02:
            reject('bounce_outside_opponent_half_or_table_height'); continue
        if HALF_LENGTH-abs(ball[bounce,0]) <= config.edge_margin_m or HALF_WIDTH-abs(ball[bounce,1]) <= config.edge_margin_m:
            reject('table_edge_potentially_clipped_by_reconstructor'); continue
        dz_in, dz_out = ball[bounce,2]-ball[bounce-1,2], ball[bounce+1,2]-ball[bounce,2]
        row.update(bounce_dz_in_m=float(dz_in), bounce_dz_out_m=float(dz_out))
        if dz_in >= 0 or dz_out <= 0:
            reject('no_vertical_bounce_reversal'); continue
        row['valid_opponent_landing'] = True
        prefix = 'crossing' if config.position_time == 'crossing' else 'pre_contact'
        if any(f'p{p}_{prefix}_y_m' not in row for p in (1,2)):
            reject('missing_selected_position_time'); continue
        for p in (1,2):
            row[f'p{p}_position'] = position_bin(row[f'p{p}_{prefix}_y_m'],p,config.center_normalized)
        row['human_skill_proxy'] = landing_category(xy,config.robot_profile)
        row['classified'] = row['human_skill_proxy'] is not None
        if not row['classified']: row['exclusion_reason'] = 'unclassifiable_landing'
        rows.append(row)
    return rows
