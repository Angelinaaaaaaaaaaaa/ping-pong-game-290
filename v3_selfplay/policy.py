"""
Actor-critic for Kexin's V3 game: her 74-dim attacker-centred state plus a role flag in, a shot head (5), a
pre-position head (3) and a value head out, on one (64, 32) trunk -- the network of our Design B self-play
(nash_skills.v2.selfplay_attack_defend.MetaPolicy). Only the head of the player's current role acts.
"""
import json

import numpy as np
import torch
from torch import nn

__all__ = ["ATTACKER", "DEFENDER", "ROLES", "STATE_DIM", "INPUT_DIM", "N_SHOTS", "N_POSITIONS", "GAME",
           "policy_input", "RolePolicy", "decide", "action_distribution", "save_policies", "load_policies"]

ATTACKER, DEFENDER = "attacker", "defender"     # her role names (v3/eval_q_policy.py)
ROLES = (ATTACKER, DEFENDER)
STATE_DIM = 74                                  # v3.state_encoder.STATE_DIM
INPUT_DIM = STATE_DIM + 1
N_SHOTS, N_POSITIONS = 5, 3
HIDDEN = (64, 32)
GAME = "kexin_v3"


def policy_input(state, role) -> np.ndarray:
    """Her state with a role flag appended (1 = attacker); a new array, the state is not modified."""
    if role not in ROLES:
        raise ValueError(f"Unknown role {role!r}; expected one of {ROLES}")
    flat = np.asarray(state, dtype=np.float32).reshape(-1)
    if flat.shape[0] != STATE_DIM:
        raise ValueError(f"Expected her {STATE_DIM}-dim state, got {flat.shape[0]} values")
    return np.concatenate([flat, np.array([1.0 if role == ATTACKER else 0.0], dtype=np.float32)])


class RolePolicy(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, hidden=HIDDEN):
        super().__init__()
        layers, prev = [], input_dim
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ReLU()]
            prev = width
        self.trunk = nn.Sequential(*layers)
        self.shot_head = nn.Linear(prev, N_SHOTS)
        self.position_head = nn.Linear(prev, N_POSITIONS)
        self.value_head = nn.Linear(prev, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.shot_head(h), self.position_head(h), self.value_head(h).squeeze(-1)


def _role_logits(policy, x, role):
    with torch.no_grad():
        shots, positions, value = policy(torch.as_tensor(x, dtype=torch.float32).unsqueeze(0))
    return (shots[0] if role == ATTACKER else positions[0]), float(value.item())


def decide(policy, x, role, argmax=False):
    """(action, log-prob, value) from the head of `role`; samples with torch's global RNG unless argmax."""
    logits, value = _role_logits(policy, x, role)
    dist = torch.distributions.Categorical(logits=logits)
    action = logits.argmax() if argmax else dist.sample()
    return int(action.item()), float(dist.log_prob(action).item()), value


def action_distribution(policy, x, role, argmax=False) -> np.ndarray:
    """Probabilities over the role's actions (one-hot on the most likely one when argmax), for her evaluator,
    which draws the action itself."""
    logits, _value = _role_logits(policy, x, role)
    if argmax:
        return np.eye(logits.shape[0], dtype=float)[int(logits.argmax().item())]
    probs = torch.softmax(logits.double(), dim=-1).numpy()
    return probs / probs.sum()          # float64 and exactly normalised: her sampler checks the sum to ~1e-8


def save_policies(policies, path, meta):
    torch.save({"p1": policies[1].state_dict(), "p2": policies[2].state_dict()}, path)
    sidecar = {"game": GAME, "state_dim": STATE_DIM, "input_dim": INPUT_DIM, "hidden": list(HIDDEN),
               "role_flag": True, "per_player": True, **meta}
    with open(path + ".json", "w") as f:
        json.dump(sidecar, f, indent=2)


def load_policies(path):
    """({1: psi_1, 2: psi_2} in eval mode, sidecar meta); refuses checkpoints made for another game."""
    with open(path + ".json") as f:
        meta = json.load(f)
    if meta.get("game") != GAME:
        raise ValueError(f"{path} was trained for game {meta.get('game')!r}, not {GAME!r}")
    state = torch.load(path, map_location="cpu")
    policies = {}
    for p in (1, 2):
        net = RolePolicy(meta["input_dim"], tuple(meta["hidden"]))
        net.load_state_dict(state[f"p{p}"])
        policies[p] = net.eval()
    return policies, meta
