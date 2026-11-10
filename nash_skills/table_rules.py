"""
Decision rules matching the rows of Kexin's V3 table (v3/eval_q_policy.py,
v3/potential_policy.py), on the Design B game where each player picks a full
(attack, preparation) pair: 15 x 15 joint tables instead of V3's 5 x 3.

  nash-q      zero-sum equilibrium of the Q game, payoff (Q1 - Q2) / 2 for player 1,
              solved by linear programming; the player samples its own mix.
  br-q        best response by the player's own Q to the opponent's previous pair;
              the first decision of a rally uses nash-q (V3's fallback).
  nash-p-argmax  the paper's online phase (Eq. 12): evaluate Phi on all 15 x 15 = 225 joint
              fixed-pair profiles, take the best one, and play this player's half of it.
  nash-p-ibr  coordinate ascent on Phi: the players alternately switch to the pair
              that raises Phi most, from the current pairs (center_safe/center at a
              rally's first decision, as in V3), until no switch helps, a pair
              repeats, or 15 rounds; the player plays its final pair.

Pickers follow make_picker_joint's contract: pick(player, obs, other_joint_idx, info).
"""
import numpy as np
import torch

from nash_skills.defend_positions import defend_position_index
from nash_skills.skills import skill_index
from nash_skills.v2.joint_action import N_JOINT_ACTIONS, joint_action_index, write_joint_pair_by_index
from nash_skills.v2.matrix_game import solve_zero_sum

__all__ = ["joint_value_table", "make_nash_q_picker", "make_br_q_picker", "make_potential_ibr_picker",
           "make_potential_argmax_picker", "make_potential_exploit_picker"]

IBR_MAX_ROUNDS = 15
EXPLOIT_REPEATS = 3
_START = joint_action_index(skill_index("center_safe"), defend_position_index("center"))


def joint_value_table(base_vec, model) -> np.ndarray:
    """model(state with player 1 on pair i, player 2 on pair j) for every (i, j)."""
    base = torch.as_tensor(np.asarray(base_vec), dtype=torch.float32)
    rows = [write_joint_pair_by_index(base, i, j) for i in range(N_JOINT_ACTIONS) for j in range(N_JOINT_ACTIONS)]
    with torch.no_grad():
        values = model(torch.stack(rows))[:, 0]
    return values.reshape(N_JOINT_ACTIONS, N_JOINT_ACTIONS).numpy()


def _nash_pair(q1, q2):
    p, q, _value = solve_zero_sum((q1 - q2) / 2.0)
    return p, q


def make_nash_q_picker(model1, model2, base_fn, rng):
    def pick_nash_q(player, obs, _other, info=None):
        base = base_fn(obs, info, player)
        p, q = _nash_pair(joint_value_table(base, model1), joint_value_table(base, model2))
        mix = p if player == 1 else q
        return int(rng.choice(N_JOINT_ACTIONS, p=mix))
    return pick_nash_q


def make_br_q_picker(model1, model2, base_fn, rng):
    nash = make_nash_q_picker(model1, model2, base_fn, rng)

    def pick_br_q(player, obs, other, info=None):
        if info is not None and info.get("decision_index", 1) == 0:
            return nash(player, obs, other, info)
        base = base_fn(obs, info, player)
        if player == 1:
            return int(np.argmax(joint_value_table(base, model1)[:, other]))
        return int(np.argmax(joint_value_table(base, model2)[other, :]))
    return pick_br_q


def coordinate_ascent(phi, i, j, max_rounds=IBR_MAX_ROUNDS, tol=1e-9):
    """Alternate player-1 / player-2 improvements of phi[i, j]; returns the final (i, j)."""
    seen = {(i, j)}
    for _ in range(max_rounds):
        changed = False
        best_i = int(np.argmax(phi[:, j]))
        if phi[best_i, j] > phi[i, j] + tol:
            i, changed = best_i, True
            if (i, j) in seen:
                break
            seen.add((i, j))
        best_j = int(np.argmax(phi[i, :]))
        if phi[i, best_j] > phi[i, j] + tol:
            j, changed = best_j, True
            if (i, j) in seen:
                break
            seen.add((i, j))
        if not changed:
            break
    return i, j


def make_potential_ibr_picker(model_p, base_fn):
    def pick_potential_ibr(player, obs, other, info=None):
        phi = joint_value_table(base_fn(obs, info, player), model_p)
        if info is None or info.get("decision_index", 0) == 0 or "current_joints" not in info:
            i, j = _START, _START
        else:
            i, j = info["current_joints"]
        i, j = coordinate_ascent(phi, int(i), int(j))
        return i if player == 1 else j
    return pick_potential_ibr


def make_potential_argmax_picker(model_p, base_fn):
    def pick_potential_argmax(player, obs, _other, info=None):
        phi = joint_value_table(base_fn(obs, info, player), model_p)
        i, j = np.unravel_index(int(np.argmax(phi)), phi.shape)
        return int(i) if player == 1 else int(j)
    return pick_potential_argmax


def make_potential_exploit_picker(model_p, model1, model2, base_fn, repeats=EXPLOIT_REPEATS):
    """Eq. 12 (argmax of Phi over the joint profiles), except against an opponent whose last `repeats` observed
    pairs are identical: then the best response to that pair by the player's own Q. A simple opponent-exploitation
    rule in the spirit of Ganzfried & Sandholm (2015), "Safe opponent exploitation", but without their safety
    guarantee: an opponent can repeat a pair to bait it. Observations persist across rallies; the placeholder pair
    in force at a rally's first decision is not one. pick.stats counts which branch decided."""
    argmax = make_potential_argmax_picker(model_p, base_fn)
    seen = {1: [], 2: []}

    def pick(player, obs, other, info=None):
        if info is None or info.get("decision_index", 1) > 0:
            seen[player].append(int(other))
        recent = seen[player][-repeats:]
        if len(recent) == repeats and len(set(recent)) == 1:
            pick.stats["exploit"] += 1
            base = base_fn(obs, info, player)
            if player == 1:
                return int(np.argmax(joint_value_table(base, model1)[:, recent[0]]))
            return int(np.argmax(joint_value_table(base, model2)[recent[0], :]))
        pick.stats["equilibrium"] += 1
        return argmax(player, obs, other, info)
    pick.stats = {"equilibrium": 0, "exploit": 0}
    return pick
