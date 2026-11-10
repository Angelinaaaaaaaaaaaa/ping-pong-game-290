"""
Per-player, role-gated self-play rollouts (paper Sec. 4.2).

Each player i has its own actor-critic psi_i (a MetaPolicy: attack head,
preparation head, value head). Decisions happen at racket contacts
(nash_skills.decision_clock, timing "contact"): the receiver is the attacker
and only its attack head acts; the player who just hit is the defender and
only its preparation head acts. The other half of each player's choice is
left as it was -- it does not act during this exchange. Both players decide
simultaneously from the same state (neither sees the other's new pick).

PPO credit therefore goes to exactly the head that acted
(role_head_log_prob_entropy); rewards follow player identity through the
role switches.
"""
import json

import numpy as np
import torch

from nash_skills.decision_clock import DecisionClock, live_halves
from nash_skills.defend_mechanisms import set_defend_choices
from nash_skills.defend_positions import N_DEFEND_POSITIONS, defend_position_from_index
from nash_skills.skills import N_SKILLS, skill_from_index
from nash_skills.v2.reward import rally_rewards
from nash_skills.v2.selfplay_attack_defend import MetaPolicy, _build_ppo_obs, _swallow_step
from nash_skills.v2.state_encoder_attack_defend import (
    ENCODER_DIMS, STATE_ENCODER_NAME, drop_choice_slots, encode_ego, encode_opp, state_dim,
)
from nash_skills.winner_inference import infer_terminal_winner

__all__ = ["learner_for_iteration", "is_round_start", "role_head_log_prob_entropy", "play_rally",
           "save_per_player", "load_per_player", "encode_for_player", "STATE_MODES", "state_dim_for",
           "DEFAULT_SELFPLAY_TIMING"]

# Actor input. "joint": the joint state, including both players' current attack /
# preparation choices in [74:90]. "no_choices": those slots dropped -- the paper's
# x = (ball, robot 1, robot 2, role) and Dvij's pi(s1, s2, role); alpha-potential needs the slots
# (they hold its policy parameters), self-play does not. The encoder (v2: 90 dims, v3: 97 with the
# ball's spin and the rally status) sets the size: no_choices is 74 (v2) or 81 (v3).
STATE_MODES = ("joint", "no_choices")
_N_CHOICE_SLOTS = 16
DEFAULT_SELFPLAY_TIMING = "contact"


def state_dim_for(state_mode: str, encoder: str = STATE_ENCODER_NAME) -> int:
    if state_mode not in STATE_MODES:
        raise ValueError(f"Unknown state mode '{state_mode}'. Choose from: {STATE_MODES}")
    return state_dim(encoder) - (0 if state_mode == "joint" else _N_CHOICE_SLOTS)


def learner_for_iteration(it: int, round_iters: int) -> int:
    """Alternating update rounds: player 1 learns for round_iters iterations, then player 2, ..."""
    return 1 if ((it - 1) // round_iters) % 2 == 0 else 2


def is_round_start(it: int, round_iters: int) -> bool:
    return (it - 1) % round_iters == 0


def role_head_log_prob_entropy(attack_logits, defend_logits, actions, is_attack):
    """Log-prob and entropy of the head that acted at each decision."""
    attack_dist = torch.distributions.Categorical(logits=attack_logits)
    defend_dist = torch.distributions.Categorical(logits=defend_logits)
    attack_lp = attack_dist.log_prob(actions.clamp(max=attack_logits.shape[-1] - 1))
    defend_lp = defend_dist.log_prob(actions.clamp(max=defend_logits.shape[-1] - 1))
    lp = torch.where(is_attack, attack_lp, defend_lp)
    ent = torch.where(is_attack, attack_dist.entropy(), defend_dist.entropy())
    return lp, ent


def encode_for_player(player, obs, info, choice, state_mode="joint", encoder=STATE_ENCODER_NAME):
    """Actor input from `player`'s seat: the joint state (with both players' current choices),
    or for state_mode "no_choices" the same state without the choice slots."""
    state_dim_for(state_mode, encoder)
    d1, d2 = defend_position_from_index(choice[1][1]), defend_position_from_index(choice[2][1])
    state = (encode_ego if player == 1 else encode_opp)(obs, info, d1, d2, encoder=encoder)
    return state if state_mode == "joint" else drop_choice_slots(state)


def _apply(env, mechanism, choice):
    env.set_skills(skill_from_index(choice[1][0]), skill_from_index(choice[2][0]))
    set_defend_choices(env, mechanism, defend_position_from_index(choice[1][1]),
                       defend_position_from_index(choice[2][1]))


def _decide(policy, state, role, device, argmax=False):
    with torch.no_grad():
        attack_logits, defend_logits, value = policy(torch.from_numpy(state).float().unsqueeze(0).to(device))
    logits = attack_logits[0] if role == "attack" else defend_logits[0]
    dist = torch.distributions.Categorical(logits=logits)
    action = logits.argmax() if argmax else dist.sample()
    return int(action.item()), float(dist.log_prob(action).item()), float(value.item())


def _ppo_step(env, ppo, obs, info):
    a1, _ = ppo.predict(_build_ppo_obs(obs, info, 1), deterministic=True)
    a2, _ = ppo.predict(_build_ppo_obs(obs, info, 2), deterministic=True)
    action = np.zeros(18, dtype=np.float32)
    action[:9], action[9:] = a1[:9], a2[:9]
    return _swallow_step(env, action)


def play_rally(env, ppo, policies, learner, device, defend_mechanism, max_steps, rng, init_choice=None,
               state_mode="joint", encoder=STATE_ENCODER_NAME, decision_timing=DEFAULT_SELFPLAY_TIMING):
    """One rally; records the learner's decisions. policies: {1: psi_1, 2: psi_2}.

    init_choice ((a1, d1), (a2, d2)) is in force until the first decision
    (the initial latch, a few steps in); random when None.
    decision_timing: "contact" (default: each racket contact) or "crossing"
    (each net crossing), see nash_skills.decision_clock.
    """
    clock = DecisionClock(decision_timing)
    if init_choice is None:
        init_choice = tuple((int(rng.integers(N_SKILLS)), int(rng.integers(N_DEFEND_POSITIONS))) for _ in (1, 2))
    choice = {1: list(init_choice[0]), 2: list(init_choice[1])}
    _apply(env, defend_mechanism, choice)
    obs, info = env.reset()
    clock.reset(env, obs)
    rec = {"states": [], "roles": [], "actions": [], "old_log_probs": [], "old_values": []}
    usage = {"attack": np.zeros(N_SKILLS, int), "defend": np.zeros(N_DEFEND_POSITIONS, int)}
    steps, exchanges, done = 0, 0, False
    while not done and steps < max_steps:
        obs, _r, done, _t, info = _ppo_step(env, ppo, obs, info)
        steps += 1
        attacker = clock.update(env, obs) if not (done and decision_timing == "crossing") else None
        if attacker is None:
            continue
        exchanges += 1
        states = {p: encode_for_player(p, obs, info, choice, state_mode, encoder) for p in (1, 2)}
        for p, role in live_halves(attacker).items():
            action, lp, value = _decide(policies[p], states[p], role, device)
            choice[p][0 if role == "attack" else 1] = action
            if p == learner:
                rec["states"].append(states[p])
                rec["roles"].append(role)
                rec["actions"].append(action)
                rec["old_log_probs"].append(lp)
                rec["old_values"].append(value)
                usage[role][action] += 1
        _apply(env, defend_mechanism, choice)

    winner = infer_terminal_winner(obs, info, fallback="position") if done else None
    r1, r2 = rally_rewards(done, winner == "ego", exchanges)
    bootstrap = 0.0
    if not done:
        _a, _l, bootstrap = _decide(policies[learner], encode_for_player(learner, obs, info, choice, state_mode, encoder),
                                    "attack", device)
    return {**rec, "reward": r1 if learner == 1 else r2, "done": done, "winner": winner,
            "bootstrap_value": bootstrap, "steps": steps, "exchanges": exchanges, "usage": usage}


def save_per_player(policies, path, meta=None):
    torch.save({"p1": policies[1].state_dict(), "p2": policies[2].state_dict()}, path)
    sidecar = {"encoder": STATE_ENCODER_NAME, "per_player": True, "decision_timing": DEFAULT_SELFPLAY_TIMING,
               "role_gated": True, **(meta or {})}
    sidecar["decision_clock_version"] = ("kexin_net_crossing_v1"
        if sidecar["decision_timing"] == "crossing" else "contact_v1")
    with open(path + ".json", "w") as f:
        json.dump(sidecar, f, indent=2)


def load_per_player(path):
    try:
        with open(path + ".json") as f:
            meta = json.load(f)
    except FileNotFoundError:
        meta = {}
    if not meta.get("per_player"):
        raise ValueError(f"{path} is not a per-player self-play checkpoint (no per_player flag in its sidecar)")
    if meta.get("encoder") not in ENCODER_DIMS:
        raise ValueError(f"{path} was trained with state encoder {meta.get('encoder')!r}, "
                         f"not one of {sorted(ENCODER_DIMS)}")
    state = torch.load(path, map_location="cpu")
    policies = {}
    for p in (1, 2):
        policies[p] = MetaPolicy(state_dim=state_dim_for(meta.get("state_mode", "joint"), meta["encoder"]))
        policies[p].load_state_dict(state[f"p{p}"])
        policies[p].eval()
    return policies, meta
