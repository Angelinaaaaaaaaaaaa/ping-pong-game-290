"""
Kexin's V3 alpha-potential player (her v3/ package, the pipeline behind the draft's Table 1) inside the Design B
match runner, so it can play our methods head to head. At each exchange it encodes the state in her
attacker-centred frame (v3.state_encoder.state_for_attacker), evaluates her 5 x 3 potential table (the attacker's
shot x the defender's pre-position) with her trained Phi, and applies one of her rules for its current role the way
v3/eval_q_policy.py does: Potential IBR starts from the current attacker's and defender's last choices in the
rally, Potential BR holds the opponent's last choice, Potential Adaptive Match answers the opponent's smoothed
choice counts over the match, Potential Minimax takes the best worst case.

Names: her shots are named from the hitter's own side (v3/skill_semantics.py), so her index a is our index 4 - a
(her "left" is our "right"); her pre-positions are ordered (left, center, right) and share our preparation names.
Execution follows the host match (decisions at racket contacts, calibrated preparation poses), not her net-crossing
decisions and gantry pre-positions.
"""
import numpy as np

from nash_skills.defend_positions import DEFEND_POSITION_NAMES, defend_position_index
from nash_skills.skills import N_SKILLS
from nash_skills.v2.joint_action import joint_action_from_index, joint_action_index
from v3.constants import DEFENSE_POSITIONS as V3_DEFENSE_POSITIONS
from v3.potential_policy import (
    load_potential_model, potential_adaptive_action, potential_ibr, potential_matrix, potential_minimax_action,
)
from v3.state_encoder import state_for_attacker

__all__ = ["RULES", "make_kexin_v3_picker", "ours_attack_from_v3", "v3_attack_from_ours", "ours_defense_from_v3",
           "v3_defense_from_ours"]

RULES = ("potential_ibr", "potential_br", "potential_adaptive_match", "potential_minimax")
_CENTER_ATTACK = N_SKILLS // 2                 # her default when nothing has been seen in the rally
_CENTER_DEFENSE = len(V3_DEFENSE_POSITIONS) // 2


def ours_attack_from_v3(a):
    return N_SKILLS - 1 - int(a)


def v3_attack_from_ours(a):
    return N_SKILLS - 1 - int(a)


def ours_defense_from_v3(d):
    return defend_position_index(V3_DEFENSE_POSITIONS[int(d)])


def v3_defense_from_ours(d):
    return V3_DEFENSE_POSITIONS.index(DEFEND_POSITION_NAMES[int(d)])


class _Memory:
    """What her evaluator remembers for one player: last choices in the rally, opponent counts over the match."""

    def __init__(self):
        self.counts = {"attack": np.zeros(N_SKILLS, dtype=np.int64),
                       "defense": np.zeros(len(V3_DEFENSE_POSITIONS), dtype=np.int64)}
        self.new_rally()

    def new_rally(self):
        self.own_attack = self.own_defense = self.opp_attack = self.opp_defense = None


def _choose(rule, phi, role, mem):
    if rule == "potential_minimax":
        return potential_minimax_action(phi, role)[0]
    if rule == "potential_adaptive_match":
        return potential_adaptive_action(phi, role, mem.counts["attack"], mem.counts["defense"])[0]
    if rule == "potential_br":
        if role == "attacker":
            held = _CENTER_DEFENSE if mem.opp_defense is None else mem.opp_defense
            return int(np.argmax(phi[:, held]))
        held = _CENTER_ATTACK if mem.opp_attack is None else mem.opp_attack
        return int(np.argmax(phi[held, :]))
    # potential_ibr: from (attacker's last shot, defender's last pre-position) when both are known, else the centre
    attack, defense = (mem.own_attack, mem.opp_defense) if role == "attacker" else (mem.opp_attack, mem.own_defense)
    if attack is None or defense is None:
        attack, defense = _CENTER_ATTACK, _CENTER_DEFENSE
    final = potential_ibr(phi, attack, defense)["final_joint_action"]
    return final["attack_index"] if role == "attacker" else final["defense_index"]


def make_kexin_v3_picker(rule, potential_model_path=None, potential_model=None, device="cpu"):
    """pick(player, obs, other_joint_idx, info) -> joint index in our names; info needs attacker,
    decision_index and current_joints, as run_matchup_joint provides."""
    if rule not in RULES:
        raise ValueError(f"Unknown V3 rule {rule!r}. Choose from: {RULES}")
    model = potential_model if potential_model is not None else load_potential_model(potential_model_path, device)
    memory = {1: _Memory(), 2: _Memory()}

    def pick(player, obs, other, info):
        attacker = info["attacker"]
        role = "attacker" if player == attacker else "defender"
        mem = memory[player]
        if info.get("decision_index", 0) == 0:
            mem.new_rally()
        else:
            # roles alternate, so the opponent's half chosen at the previous exchange is the fresh one
            opp_attack, opp_defense = joint_action_from_index(int(other))
            if role == "attacker":
                mem.opp_attack = v3_attack_from_ours(opp_attack)
                mem.counts["attack"][mem.opp_attack] += 1
            else:
                mem.opp_defense = v3_defense_from_ours(opp_defense)
                mem.counts["defense"][mem.opp_defense] += 1
        phi = potential_matrix(model, state_for_attacker(obs, info, attacker), device=device)
        choice = _choose(rule, phi, role, mem)
        own_attack, own_defense = joint_action_from_index(int(info["current_joints"][player - 1]))
        if role == "attacker":
            mem.own_attack = choice
            return joint_action_index(ours_attack_from_v3(choice), own_defense)
        mem.own_defense = choice
        return joint_action_index(own_attack, ours_defense_from_v3(choice))

    pick.opponent_counts = {p: m.counts for p, m in memory.items()}
    return pick
