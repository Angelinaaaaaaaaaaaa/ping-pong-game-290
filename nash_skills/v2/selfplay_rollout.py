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
from nash_skills.v2.state_encoder_attack_defend import STATE_ENCODER_NAME, encode_ego, encode_opp
from nash_skills.winner_inference import infer_terminal_winner

__all__ = ["learner_for_iteration", "is_round_start", "role_head_log_prob_entropy", "play_rally",
           "save_per_player", "load_per_player", "encode_for_player"]


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


def encode_for_player(player, obs, info, choice):
    """Joint state from `player`'s seat, with both players' current preparation choices."""
    d1, d2 = defend_position_from_index(choice[1][1]), defend_position_from_index(choice[2][1])
    return (encode_ego if player == 1 else encode_opp)(obs, info, d1, d2)


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


def play_rally(env, ppo, policies, learner, device, defend_mechanism, max_steps, rng, init_choice=None):
    """One rally; records the learner's decisions. policies: {1: psi_1, 2: psi_2}.

    init_choice ((a1, d1), (a2, d2)) is in force until the first decision
    (the initial latch, a few steps in); random when None.
    """
    if init_choice is None:
        init_choice = tuple((int(rng.integers(N_SKILLS)), int(rng.integers(N_DEFEND_POSITIONS))) for _ in (1, 2))
    choice = {1: list(init_choice[0]), 2: list(init_choice[1])}
    _apply(env, defend_mechanism, choice)
    obs, info = env.reset()
    clock = DecisionClock("contact")
    clock.reset()
    rec = {"states": [], "roles": [], "actions": [], "old_log_probs": [], "old_values": []}
    usage = {"attack": np.zeros(N_SKILLS, int), "defend": np.zeros(N_DEFEND_POSITIONS, int)}
    steps, exchanges, done = 0, 0, False
    while not done and steps < max_steps:
        obs, _r, done, _t, info = _ppo_step(env, ppo, obs, info)
        steps += 1
        attacker = clock.update(env)
        if attacker is None:
            continue
        exchanges += 1
        states = {p: encode_for_player(p, obs, info, choice) for p in (1, 2)}
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
        _a, _l, bootstrap = _decide(policies[learner], encode_for_player(learner, obs, info, choice),
                                    "attack", device)
    return {**rec, "reward": r1 if learner == 1 else r2, "done": done, "winner": winner,
            "bootstrap_value": bootstrap, "steps": steps, "exchanges": exchanges, "usage": usage}


def save_per_player(policies, path, meta=None):
    torch.save({"p1": policies[1].state_dict(), "p2": policies[2].state_dict()}, path)
    sidecar = {"encoder": STATE_ENCODER_NAME, "per_player": True, "decision_timing": "contact",
               "role_gated": True, **(meta or {})}
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
    if meta.get("encoder") != STATE_ENCODER_NAME:
        raise ValueError(f"{path} was trained with state encoder {meta.get('encoder')!r}, not {STATE_ENCODER_NAME!r}")
    state = torch.load(path, map_location="cpu")
    policies = {}
    for p in (1, 2):
        policies[p] = MetaPolicy()
        policies[p].load_state_dict(state[f"p{p}"])
        policies[p].eval()
    return policies, meta
