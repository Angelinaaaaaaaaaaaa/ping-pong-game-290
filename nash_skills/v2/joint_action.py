"""
Shared joint-(attack,defend)-action index/one-hot-write helpers for
Design B's attacker/defender extension, used by BOTH the Phi/Q trainer
(train_q_model_5skill_v3_attack_defend.py) and the decision-rule module
(eval_matchup_attack_defend.py) -- a single shared implementation so the
two can never silently diverge on the joint-index convention or the
one-hot-write logic, the exact failure mode already caught twice this
session (eval_matchup.py's _write_skill_pair bug; the legacy-vs-one-hot
apply_skill_pair split).

A per-player action in Design B is a (attack_idx, defend_idx) pair -- 5
attack skills x 3 defend positions = 15 combinations. Where a single flat
index is needed (e.g. a 15x15 Phi/Q table's row/column), it's row-major:
joint_action_index(attack_idx, defend_idx) = attack_idx * N_DEFEND_POSITIONS
+ defend_idx.

Layout written (nash_skills.v2.state_encoder_attack_defend's 90-dim
state): ego attack one-hot [74:79], opp attack one-hot [79:84], ego
defend one-hot [84:87], opp defend one-hot [87:90] -- the 84-dim one-hot
prefix (state_encoder_one_hot) unchanged, defend blocks appended after.
"""
import torch

from nash_skills.skills import N_SKILLS
from nash_skills.defend_positions import N_DEFEND_POSITIONS

__all__ = [
    "N_JOINT_ACTIONS",
    "joint_action_index",
    "joint_action_from_index",
    "write_joint_action_pair",
    "write_joint_pair_by_index",
    "write_mixed_strategy_pair",
]

N_JOINT_ACTIONS = N_SKILLS * N_DEFEND_POSITIONS  # 15

_EGO_ATTACK_START = 74
_OPP_ATTACK_START = 74 + N_SKILLS
_EGO_DEFEND_START = 74 + 2 * N_SKILLS
_OPP_DEFEND_START = 74 + 2 * N_SKILLS + N_DEFEND_POSITIONS

_EGO_ATTACK = slice(_EGO_ATTACK_START, _EGO_ATTACK_START + N_SKILLS)
_OPP_ATTACK = slice(_OPP_ATTACK_START, _OPP_ATTACK_START + N_SKILLS)
_EGO_DEFEND = slice(_EGO_DEFEND_START, _EGO_DEFEND_START + N_DEFEND_POSITIONS)
_OPP_DEFEND = slice(_OPP_DEFEND_START, _OPP_DEFEND_START + N_DEFEND_POSITIONS)


def joint_action_index(attack_idx: int, defend_idx: int) -> int:
    """Flatten a per-player (attack_idx, defend_idx) pair to 0..14."""
    return attack_idx * N_DEFEND_POSITIONS + defend_idx


def joint_action_from_index(joint_idx: int):
    """Inverse of joint_action_index: 0..14 -> (attack_idx, defend_idx)."""
    return joint_idx // N_DEFEND_POSITIONS, joint_idx % N_DEFEND_POSITIONS


def write_joint_action_pair(
    state_tensor: torch.Tensor,
    ego_attack: int, ego_defend: int,
    opp_attack: int, opp_defend: int,
) -> torch.Tensor:
    """
    Clone a base state (or batch of states) and overwrite all four
    one-hot blocks: ego attack [74:79], opp attack [79:84], ego defend
    [84:87], opp defend [87:90]. Zeroes each block before setting its
    index, so the result is always a clean one-hot regardless of what
    was in those dims before. Works on both a single (90,) row and a
    batched (N, 90) tensor via `...` indexing.
    """
    out = state_tensor.clone()
    out[..., _EGO_ATTACK] = 0.0
    out[..., _EGO_ATTACK_START + ego_attack] = 1.0
    out[..., _OPP_ATTACK] = 0.0
    out[..., _OPP_ATTACK_START + opp_attack] = 1.0
    out[..., _EGO_DEFEND] = 0.0
    out[..., _EGO_DEFEND_START + ego_defend] = 1.0
    out[..., _OPP_DEFEND] = 0.0
    out[..., _OPP_DEFEND_START + opp_defend] = 1.0
    return out


def write_joint_pair_by_index(
    state_tensor: torch.Tensor,
    ego_joint: int,
    opp_joint: int,
) -> torch.Tensor:
    """
    Like write_joint_action_pair, but takes each player's action as a
    single flat joint index (0..14, e.g. a 15x15 Phi/Q table's row or
    column) instead of an already-decomposed (attack_idx, defend_idx)
    pair -- decomposes via joint_action_from_index, then delegates.
    Shared by the Phi/Q trainer and the decision-rule module so neither
    reimplements the decomposition and risks drifting from the other.
    """
    ego_attack, ego_defend = joint_action_from_index(ego_joint)
    opp_attack, opp_defend = joint_action_from_index(opp_joint)
    return write_joint_action_pair(state_tensor, ego_attack, ego_defend, opp_attack, opp_defend)


def write_mixed_strategy_pair(state_tensor: torch.Tensor, ego_attack_w, ego_defend_w,
                              opp_attack_w, opp_defend_w) -> torch.Tensor:
    """
    Like write_joint_action_pair, but writes each player's MIXED strategy
    -- the paper's compact policy family theta^i = (w^{A,i}, w^{D,i}) in
    Delta_5 x Delta_3 -- into the same four blocks, instead of a one-hot.
    One-hot inputs reproduce write_joint_action_pair exactly (the pure
    action IS theta at a simplex vertex); a regression test pins that.

    Differentiable in the four w arguments (index assignment into the
    clone keeps the autograd graph), which is what lets the online phase
    run projected gradient ascent on Phi(x, theta) directly. Accepts
    numpy arrays or tensors for w; broadcasts over a batched state.
    """
    out = state_tensor.clone()

    def _as(w):
        return w if isinstance(w, torch.Tensor) else torch.as_tensor(w, dtype=out.dtype)

    out[..., _EGO_ATTACK] = _as(ego_attack_w)
    out[..., _OPP_ATTACK] = _as(opp_attack_w)
    out[..., _EGO_DEFEND] = _as(ego_defend_w)
    out[..., _OPP_DEFEND] = _as(opp_defend_w)
    return out
