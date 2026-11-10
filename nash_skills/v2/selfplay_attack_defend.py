"""
Design B self-play: two-headed meta-policy over (attack-skill,
defend-position), generating its own rollouts -- Phase 4 of the Table 1
plan. Can run independently of the offline alpha-potential pipeline
(Phases 2/3): it never touches the collected dataset, only the live env
and the frozen low-level PPO.

Direct extension of ping-pong-game-290-selfplay/selfplay_5skill.py's
single-headed design (read in full this session): same shared trunk, a
second N_DEFEND_POSITIONS-wide head alongside the existing N_SKILLS-wide
one. Both heads are ALWAYS sampled at every crossing, by both players,
unconditionally -- Design B's own structure (both players always hold a
joint attack+defend choice; the environment resolves which half is
physically live via SkillEnv's internal _role_tracker) means there is
no role-gating to add here at all.

UPDATE -- training algorithm, REINFORCE -> actor-critic -> PPO:
selfplay_5skill.py's own training loop (and this file's own first
version) used plain REINFORCE with a single GLOBAL SCALAR baseline (an
EMA of recent mean reward) -- every decision in a rally, regardless of
how easy or hard ITS specific state was, got the same baseline
subtracted. This file now implements real PPO (Schulman et al., 2017),
matching what the paper draft's Section 4.2 actually claims ("train each
player using proximal policy optimization"), which the REINFORCE version
did not:
  - MetaPolicy gained a value_head (shares the existing trunk) --
    forward() returns (attack_logits, defend_logits, value) instead of
    2 values. The value is a per-STATE baseline, replacing the old
    global scalar -- this directly fixes the credit-assignment problem
    the scalar baseline had (unable to distinguish "lost because of a
    bad decision" from "lost because the state was intrinsically hard"
    -- e.g. the attack=right/defend=right matchup, the stubborn outlier
    cell found this session, which is harder regardless of choice).
  - compute_gae(): standard Generalized Advantage Estimation. This game
    has no intermediate per-decision reward (only a terminal point
    win/loss) -- rewards is [0, 0, ..., terminal_reward] per rally per
    player; GAE still applies cleanly to a sparse/terminal-only reward
    sequence, it is not restricted to per-step-reward environments.
  - ppo_clipped_loss(): the clipped surrogate objective, using the
    PROBABILITY RATIO between the policy now being updated and the
    (frozen, as of collection time) policy that actually acted --
    bounds how far a single update can move the policy away from what
    it did during collection, which plain REINFORCE has no mechanism
    for at all.
  - train()'s loop is now two phases per iteration: COLLECT a full
    batch of rallies under torch.no_grad() (recording states, actions,
    old log-probs, old values -- no gradients needed yet), THEN run
    several epochs of minibatch updates re-evaluating those same stored
    states under the policy as it updates within the epoch loop. The
    old single-collect-single-gradient-step loop is gone.
Two-headed credit assignment: attack and defend are independently-
sampled categoricals from two heads sharing one trunk, so the joint
log-prob used for the PPO ratio is the SUM of the two heads' log-probs
(log of the product under a conditional-independence assumption between
heads) -- a standard, direct way to extend PPO's ratio to a
multi-discrete action space without inventing a new mechanism.

Deliberately placed in root nash_skills/v2/, NOT the
ping-pong-game-290-selfplay/ worktree (unlike selfplay_5skill.py) -- that
worktree's own nash_skills/ copy predates the entire attacker/defender
extension (no defend_role.py, defend_reactive_pose.py, _role_tracker, or
set_defend_reactive_positions on env_wrapper.py) and its own script's
sys.path resolution order means it resolves against that stale copy,
not root (confirmed by reading it this session). Keeping this file in
root's own nash_skills/v2/ avoids that path-resolution risk entirely by
construction, rather than fighting it with explicit root-first path
insertion inside the worktree directory.

Uses set_defend_reactive_positions (the velocity/trajectory-informed
mechanism, mechanism_validation_reactive_pose.py: avg_delta_tested_wr
-0.068, avg_delta_contacts +9.483 at full scale -- the first no-retrain
mechanism this session where that sign is right), NOT
set_defend_target_positions (confirmed net-harmful) or
set_defend_positions (confirmed net-harmful). MAX_STEPS_PER_RALLY raised
to 2500 from selfplay_5skill.py's 1000 -- confirmed this session that
1000 steps under-counts decisive outcomes once the reactive mechanism
makes rallies far more sustained (contacts roughly tripled).

Run:
    MUJOCO_GL=cgl PYTHONPATH=. venv/bin/python3.12 \\
        nash_skills/v2/selfplay_attack_defend.py \\
        --iterations 500 --rallies-per-iter 32 --entropy-coef 0.05 \\
        --output models/selfplay_attack_defend.pth
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import csv
import io
import json
import random
import time
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO

from nash_skills.env_wrapper import SkillEnv
from nash_skills.skills import SKILL_NAMES, N_SKILLS, skill_from_index
from nash_skills.defend_positions import (
    DEFEND_POSITION_NAMES, N_DEFEND_POSITIONS, defend_position_from_index,
)
from nash_skills.v2.reward import rally_rewards
from nash_skills.defend_mechanisms import (
    DEFEND_MECHANISMS, DEFAULT_DEFEND_MECHANISM, set_defend_choices, validate_defend_mechanism,
)
from nash_skills.v2.state_encoder_attack_defend import (
    encode_ego, encode_opp, STATE_DIM_ATTACK_DEFEND, STATE_ENCODER_NAME,
)
from nash_skills.winner_inference import infer_terminal_winner

# --------------------------------------------------------------------------- #
PPO_MODEL_PATH      = "logs/best_model_tracker1/best_model"
DEFAULT_OUTPUT      = "models/selfplay_attack_defend.pth"
DEFAULT_LOG         = "logs/selfplay_attack_defend.csv"
HISTORY             = 4
TABLE_SHIFT         = 1.5
MAX_STEPS_PER_RALLY = 2500
# UPDATE: was -0.5 (a fixed, artificial penalty for both players on a
# truncated rally, matching selfplay_5skill.py's own original design).
# Now 0.0 -- truncation is handled by bootstrapping the critic's own
# value estimate at the cutoff state instead (see compute_gae's
# bootstrap_value and play_one_rally's bootstrap_ego_value/
# bootstrap_opp_value), which is the paper's own framing: "bootstrapped
# from its critic rather than treated as a lost point." Keeping an
# artificial penalty on top would double-count that signal.
TRUNCATED_PENALTY = 0.0
# --------------------------------------------------------------------------- #


class MetaPolicy(nn.Module):
    """90-dim state -> two categoricals sharing one trunk (N_SKILLS
    attack, N_DEFEND_POSITIONS defend) PLUS a state-value estimate from
    a third head on the same trunk -- the critic PPO (and plain
    actor-critic) needs. See module docstring's "UPDATE" paragraph."""

    def __init__(self, state_dim: int = STATE_DIM_ATTACK_DEFEND,
                 n_attack: int = N_SKILLS, n_defend: int = N_DEFEND_POSITIONS,
                 hidden=(64, 32)):
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.attack_head = nn.Linear(prev, n_attack)
        self.defend_head = nn.Linear(prev, n_defend)
        self.value_head = nn.Linear(prev, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.attack_head(h), self.defend_head(h), self.value_head(h).squeeze(-1)

    @torch.no_grad()
    def sample(self, state_np: np.ndarray, device):
        """Used only for frozen opponents (snapshots / fixed baselines)
        -- no value needed since their decisions are never trained on."""
        x = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
        attack_logits, defend_logits, _value = self.forward(x)
        attack_probs = F.softmax(attack_logits[0], dim=-1)
        defend_probs = F.softmax(defend_logits[0], dim=-1)
        attack_idx = int(torch.multinomial(attack_probs, 1).item())
        defend_idx = int(torch.multinomial(defend_probs, 1).item())
        return (attack_idx, defend_idx,
                attack_probs.detach().cpu().numpy(), defend_probs.detach().cpu().numpy())


class FixedJointPolicy:
    """
    Stateless opp for epsilon-baseline exploration: a fixed
    (attack_mode, defend_mode) pair -- matches the paper's static
    baseline family (every fixed attack-target x prep-position pair,
    15 total) plus "random" for either axis independently. Implements
    only the interface used in the `not opp_uses_current` branch of
    play_one_rally: `.sample(state, device) -> (attack_idx, defend_idx,
    attack_probs, defend_probs)`; `state`/`device` are ignored.
    """

    def __init__(self, attack_mode: str, defend_mode: str):
        self.attack_mode = attack_mode
        self.defend_mode = defend_mode
        self._uniform_attack = np.full(N_SKILLS, 1.0 / N_SKILLS, dtype=np.float32)
        self._uniform_defend = np.full(N_DEFEND_POSITIONS, 1.0 / N_DEFEND_POSITIONS, dtype=np.float32)

        if attack_mode == "random":
            self._fixed_attack_idx = None
        elif attack_mode in SKILL_NAMES:
            self._fixed_attack_idx = SKILL_NAMES.index(attack_mode)
        else:
            raise ValueError(f"Unknown attack mode: {attack_mode!r}")

        if defend_mode == "random":
            self._fixed_defend_idx = None
        elif defend_mode in DEFEND_POSITION_NAMES:
            self._fixed_defend_idx = DEFEND_POSITION_NAMES.index(defend_mode)
        else:
            raise ValueError(f"Unknown defend mode: {defend_mode!r}")

    def sample(self, _state_np, _device):
        attack_idx = (random.randint(0, N_SKILLS - 1) if self._fixed_attack_idx is None
                      else self._fixed_attack_idx)
        defend_idx = (random.randint(0, N_DEFEND_POSITIONS - 1) if self._fixed_defend_idx is None
                      else self._fixed_defend_idx)
        return attack_idx, defend_idx, self._uniform_attack, self._uniform_defend


def compute_gae(rewards, values, gamma: float, gae_lambda: float, bootstrap_value: float = 0.0):
    """
    Standard Generalized Advantage Estimation over one trajectory
    (one player's sequence of decisions across one rally).

    rewards[t] is 0 for every non-terminal decision and the realized
    terminal reward only at the LAST index -- this game has no
    intermediate per-decision reward. values[t] is the critic's own
    estimate at that decision's state, AS OF COLLECTION TIME (always
    treated as fixed here, per standard practice: the advantage
    estimator never uses the policy/value currently being updated).

    bootstrap_value is the assumed value of whatever comes AFTER the
    last entry -- 0.0 (the default) for a real terminal state (the
    rally actually ended, decisively, nothing more to gain or lose).
    For a TRUNCATED rally (hit the step cap, not a real terminal), pass
    the critic's own value estimate at the cutoff state instead -- this
    is the "bootstrapped from its critic rather than treated as a lost
    point" behavior: it stops reading a mere timeout as proof the
    trajectory was headed for a loss.

    Returns (advantages, returns) as plain Python lists, same length as
    the inputs. returns = advantages + values -- the critic's own
    regression target.
    """
    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else bootstrap_value
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
    returns = [advantages[t] + values[t] for t in range(T)]
    return advantages, returns


def ppo_clipped_loss(new_log_probs, old_log_probs, advantages, clip_epsilon: float):
    """
    The PPO clipped surrogate policy loss (Schulman et al., 2017),
    mean-reduced and already negated (ready to minimize). All three
    tensor args must be equal-length, one entry per decision in the
    current minibatch. new_log_probs comes from the policy as it
    currently stands (with gradients); old_log_probs is detached, from
    collection time.
    """
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    return -torch.min(unclipped, clipped).mean()


def parse_baseline_pool_entry(entry: str):
    """'attack:defend' -> (attack, defend), e.g. 'left:center' or
    'random:random'. Raises ValueError if there's no ':' separator."""
    if ":" not in entry:
        raise ValueError(
            f"Baseline pool entry '{entry}' must be 'attack:defend' (e.g. 'left:center' or 'random:random')"
        )
    attack, defend = entry.split(":", 1)
    return attack, defend


def save_checkpoint(policy, path: str):
    """Save the state dict plus a sidecar naming the state encoder it was trained with."""
    torch.save(policy.state_dict(), path)
    with open(path + ".json", "w") as f:
        json.dump({"encoder": STATE_ENCODER_NAME}, f)


def check_checkpoint_encoder(path: str):
    """Refuse checkpoints trained on another state encoder (same input size, different meaning)."""
    try:
        with open(path + ".json") as f:
            encoder = json.load(f).get("encoder")
    except FileNotFoundError:
        encoder = None
    if encoder != STATE_ENCODER_NAME:
        raise ValueError(f"{path} was trained with state encoder {encoder!r}, not {STATE_ENCODER_NAME!r}; "
                         "retrain it with the current encoder.")


def history_checkpoint_path(output_path: str, iteration: int) -> str:
    """Derive a per-iteration snapshot path from the main --output path --
    same convention as selfplay_5skill.py's own helper."""
    p = Path(output_path)
    return str(p.with_name(f"{p.stem}_iter{iteration}{p.suffix}"))


def _build_ppo_obs(obs, info, player):
    """Low-level PPO's own observation -- unrelated to the meta-policy's
    state encoding, unchanged from selfplay_5skill.py."""
    o = np.zeros(9 + 9 + 7 + 7 + 9 * HISTORY, dtype=np.float32)
    if player == 1:
        o[:9]    = obs[:9]
        o[9:18]  = obs[18:27]
        o[18:21] = info["diff_pos"]
        o[21:25] = info["diff_quat"]
        o[25:32] = info["target"]
        o[32:]   = obs[42: 42 + HISTORY * 9]
    else:
        o[:9]    = obs[9:18]
        o[9:18]  = obs[27:36]
        o[18:21] = info["diff_pos_opp"]
        o[21:25] = info["diff_quat_opp"]
        o[25:32] = info["target_opp"]
        o[32:]   = obs[42 + HISTORY * 9: 42 + 2 * HISTORY * 9]
    return o


def _swallow_step(env, action):
    buf = io.StringIO()
    with redirect_stdout(buf):
        return env.step(action)


def play_one_rally(env, ppo, ego_policy, opp_policy, device,
                    ego_attack_init=0, opp_attack_init=0,
                    ego_defend_init=0, opp_defend_init=0,
                    trunc_penalty=TRUNCATED_PENALTY,
                    shaping_coef=0.0, trunc_shaping_coef=0.0,
                    defend_mechanism=DEFAULT_DEFEND_MECHANISM):
    """
    Collects one rally's trajectory under torch.no_grad() -- PPO's
    collection phase never needs gradients; those are computed later,
    in train()'s separate multi-epoch update phase, by re-evaluating
    these same recorded states under the policy as it stands at update
    time. Records states/actions/old log-probs/old values for ego, and
    for opp ONLY when opp_uses_current (a frozen snapshot/baseline
    opponent's decisions are never trained on, so there is nothing
    beyond the choice itself worth recording for them).
    `defend_mechanism`: see nash_skills.defend_mechanisms.
    """
    validate_defend_mechanism(defend_mechanism)
    ego_attack_name = skill_from_index(ego_attack_init)
    opp_attack_name = skill_from_index(opp_attack_init)
    ego_defend_name = defend_position_from_index(ego_defend_init)
    opp_defend_name = defend_position_from_index(opp_defend_init)

    env.set_skills(ego_attack_name, opp_attack_name)
    set_defend_choices(env, defend_mechanism, ego_defend_name, opp_defend_name)
    obs, info = env.reset()
    prev_ball_x = float(obs[36])

    ego_states, ego_attack_actions, ego_defend_actions = [], [], []
    ego_old_attack_log_probs, ego_old_defend_log_probs, ego_old_values = [], [], []
    opp_states, opp_attack_actions, opp_defend_actions = [], [], []
    opp_old_attack_log_probs, opp_old_defend_log_probs, opp_old_values = [], [], []
    ego_attacks, ego_defends = [ego_attack_init], [ego_defend_init]
    opp_attacks, opp_defends = [opp_attack_init], [opp_defend_init]
    opp_uses_current = (opp_policy is ego_policy)

    crossings = 0
    steps = 0
    done = False
    while True:
        ppo1 = _build_ppo_obs(obs, info, 1)
        ppo2 = _build_ppo_obs(obs, info, 2)
        a1, _ = ppo.predict(ppo1, deterministic=True)
        a2, _ = ppo.predict(ppo2, deterministic=True)
        action = np.zeros(18, dtype=np.float32)
        action[:9] = a1[:9]; action[9:] = a2[:9]

        obs, _r, done, _t, info = _swallow_step(env, action)
        steps += 1
        curr_ball_x = float(obs[36])

        if (prev_ball_x - TABLE_SHIFT) * (curr_ball_x - TABLE_SHIFT) < 0:
            crossings += 1
            ego_state = encode_ego(obs, info, ego_defend_name, opp_defend_name)
            opp_state = encode_opp(obs, info, ego_defend_name, opp_defend_name)

            with torch.no_grad():
                ego_attack_logits, ego_defend_logits, ego_value = ego_policy(
                    torch.from_numpy(ego_state).float().unsqueeze(0).to(device))
                ego_attack_dist = torch.distributions.Categorical(logits=ego_attack_logits[0])
                ego_defend_dist = torch.distributions.Categorical(logits=ego_defend_logits[0])
                ego_attack_action = ego_attack_dist.sample()
                ego_defend_action = ego_defend_dist.sample()
                ego_attack_idx = int(ego_attack_action.item())
                ego_defend_idx = int(ego_defend_action.item())

            ego_states.append(ego_state)
            ego_attack_actions.append(ego_attack_idx)
            ego_defend_actions.append(ego_defend_idx)
            ego_old_attack_log_probs.append(float(ego_attack_dist.log_prob(ego_attack_action).item()))
            ego_old_defend_log_probs.append(float(ego_defend_dist.log_prob(ego_defend_action).item()))
            ego_old_values.append(float(ego_value.item()))
            ego_attacks.append(ego_attack_idx)
            ego_defends.append(ego_defend_idx)

            if opp_uses_current:
                with torch.no_grad():
                    opp_attack_logits, opp_defend_logits, opp_value = opp_policy(
                        torch.from_numpy(opp_state).float().unsqueeze(0).to(device))
                    opp_attack_dist = torch.distributions.Categorical(logits=opp_attack_logits[0])
                    opp_defend_dist = torch.distributions.Categorical(logits=opp_defend_logits[0])
                    opp_attack_action = opp_attack_dist.sample()
                    opp_defend_action = opp_defend_dist.sample()
                    opp_attack_idx = int(opp_attack_action.item())
                    opp_defend_idx = int(opp_defend_action.item())
                opp_states.append(opp_state)
                opp_attack_actions.append(opp_attack_idx)
                opp_defend_actions.append(opp_defend_idx)
                opp_old_attack_log_probs.append(float(opp_attack_dist.log_prob(opp_attack_action).item()))
                opp_old_defend_log_probs.append(float(opp_defend_dist.log_prob(opp_defend_action).item()))
                opp_old_values.append(float(opp_value.item()))
            else:
                with torch.no_grad():
                    opp_attack_idx, opp_defend_idx, _ap, _dp = opp_policy.sample(opp_state, device)
            opp_attacks.append(opp_attack_idx)
            opp_defends.append(opp_defend_idx)

            ego_attack_name = skill_from_index(ego_attack_idx)
            opp_attack_name = skill_from_index(opp_attack_idx)
            ego_defend_name = defend_position_from_index(ego_defend_idx)
            opp_defend_name = defend_position_from_index(opp_defend_idx)
            env.set_skills(ego_attack_name, opp_attack_name)
            set_defend_choices(env, defend_mechanism, ego_defend_name, opp_defend_name)

        prev_ball_x = curr_ball_x

        if done or steps >= MAX_STEPS_PER_RALLY:
            break

    winner = None
    if done:
        winner = infer_terminal_winner(obs, info, fallback="position") or "opp"
    ego_reward, opp_reward = rally_rewards(
        done, winner == "ego", crossings,
        trunc_penalty=trunc_penalty,
        shaping_coef=shaping_coef,
        trunc_shaping_coef=trunc_shaping_coef,
    )

    # Truncation bootstrap: a fresh value estimate at the ACTUAL cutoff
    # state (not just reusing the last decision's value -- the ball may
    # have moved meaningfully since then), used in place of "no further
    # value" for GAE's last step. Only meaningful when NOT done -- a
    # real terminal has no "value beyond this point" to bootstrap.
    bootstrap_ego_value = 0.0
    bootstrap_opp_value = 0.0
    if not done:
        final_ego_state = encode_ego(obs, info, ego_defend_name, opp_defend_name)
        with torch.no_grad():
            _ego_al, _ego_dl, ego_value_t = ego_policy(
                torch.from_numpy(final_ego_state).float().unsqueeze(0).to(device))
        bootstrap_ego_value = float(ego_value_t.item())
        if opp_uses_current:
            final_opp_state = encode_opp(obs, info, ego_defend_name, opp_defend_name)
            with torch.no_grad():
                _opp_al, _opp_dl, opp_value_t = opp_policy(
                    torch.from_numpy(final_opp_state).float().unsqueeze(0).to(device))
            bootstrap_opp_value = float(opp_value_t.item())

    return {
        "ego_states": ego_states,
        "ego_attack_actions": ego_attack_actions, "ego_defend_actions": ego_defend_actions,
        "ego_old_attack_log_probs": ego_old_attack_log_probs, "ego_old_defend_log_probs": ego_old_defend_log_probs,
        "ego_old_values": ego_old_values,
        "opp_states": opp_states,
        "opp_attack_actions": opp_attack_actions, "opp_defend_actions": opp_defend_actions,
        "opp_old_attack_log_probs": opp_old_attack_log_probs, "opp_old_defend_log_probs": opp_old_defend_log_probs,
        "opp_old_values": opp_old_values,
        "bootstrap_ego_value": bootstrap_ego_value,
        "bootstrap_opp_value": bootstrap_opp_value,
        "ego_reward": ego_reward,
        "opp_reward": opp_reward,
        "opp_uses_current": opp_uses_current,
        "steps": steps,
        "crossings": crossings,
        "done": done,
        "winner": winner,
        "ego_attacks": ego_attacks,
        "ego_defends": ego_defends,
        "opp_attacks": opp_attacks,
        "opp_defends": opp_defends,
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Seeded with {args.seed}")

    env = SkillEnv(proc_id=1, history=HISTORY, skill_profile="aggressive",
                   reset_targets=args.reset_targets)
    print(f"Loading PPO from {PPO_MODEL_PATH} (CPU — MlpPolicy is faster there) ...")
    ppo = PPO.load(PPO_MODEL_PATH, device="cpu")

    policy = MetaPolicy().to(device)
    if args.resume and os.path.exists(args.output):
        check_checkpoint_encoder(args.output)
        policy.load_state_dict(torch.load(args.output, map_location=device))
        print(f"Resumed weights from {args.output}")

    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)

    snapshot_buffer = []
    baseline_pool = [FixedJointPolicy(*parse_baseline_pool_entry(e)) for e in args.baseline_pool] \
        if args.epsilon_baseline > 0 else []
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    log_f = open(args.log, "w", newline="")
    log_w = csv.writer(log_f)
    log_w.writerow([
        "iter",
        "mean_reward", "reward_std",
        "win_rate", "loss_rate", "trunc_rate", "draw_rate",
        "mean_crossings", "mean_steps",
    ] + [f"usage_attack_{s}" for s in SKILL_NAMES]
      + [f"usage_defend_{d}" for d in DEFEND_POSITION_NAMES]
      + [
        "dominant_fraction_attack", "dominant_fraction_defend", "effective_n_attack", "effective_n_defend",
    ] + [f"p_attack_{s}" for s in SKILL_NAMES] + [f"p_defend_{d}" for d in DEFEND_POSITION_NAMES] + [
        "policy_loss", "value_loss", "entropy_bonus", "entropy",
        "entropy_attack", "entropy_defend",
        "grad_norm", "advantage_mean", "advantage_std", "wall_time",
    ])

    rally_log_f = None
    rally_log_id = 0
    if args.log_rallies:
        rally_log_f = open(args.log_rallies, "w")

    print(f"\nSelf-play (Design B, attack x defend): {args.iterations} iters x {args.rallies_per_iter} "
          f"rallies/iter (entropy_coef={args.entropy_coef}, snapshot_prob={args.snapshot_prob}, "
          f"epsilon_baseline={args.epsilon_baseline}, defend_mechanism={args.defend_mechanism})")
    if baseline_pool:
        print(f"Baseline pool: {args.baseline_pool}")
    if rally_log_f is not None:
        print(f"Rally JSONL: {args.log_rallies}")
    print()

    try:
        for it in range(1, args.iterations + 1):
            iter_start_t = time.time()
            # === Collection phase: play a full batch of rallies under
            # torch.no_grad() (inside play_one_rally), pooling every
            # recorded decision's (state, action, old_log_prob,
            # old_value, advantage, return) across all rallies this
            # iteration -- for ego always, and for opp too whenever
            # opp_uses_current (direct self-play, not a frozen
            # snapshot/baseline). GAE is computed per-rally, per-player,
            # right after that rally's own old_values are known, then
            # flattened into the shared batch lists below.
            batch_states, batch_attack_actions, batch_defend_actions = [], [], []
            batch_old_attack_lp, batch_old_defend_lp = [], []
            batch_advantages, batch_returns = [], []

            rewards_this_iter = []
            wins = 0
            losses = 0
            truncs = 0
            crossings_this_iter = []
            steps_this_iter = []
            attack_usage = [0] * N_SKILLS
            defend_usage = [0] * N_DEFEND_POSITIONS

            for _ in range(args.rallies_per_iter):
                r = random.random()
                if baseline_pool and r < args.epsilon_baseline:
                    opp_policy = random.choice(baseline_pool)
                    opp_type = "baseline"
                elif snapshot_buffer and r < args.epsilon_baseline + args.snapshot_prob:
                    opp_policy = random.choice(snapshot_buffer)
                    opp_type = "snapshot"
                else:
                    opp_policy = policy
                    opp_type = "self"

                ego_attack_init = random.randint(0, N_SKILLS - 1)
                opp_attack_init = random.randint(0, N_SKILLS - 1)
                ego_defend_init = random.randint(0, N_DEFEND_POSITIONS - 1)
                opp_defend_init = random.randint(0, N_DEFEND_POSITIONS - 1)

                rally = play_one_rally(
                    env, ppo, policy, opp_policy, device,
                    ego_attack_init=ego_attack_init, opp_attack_init=opp_attack_init,
                    ego_defend_init=ego_defend_init, opp_defend_init=opp_defend_init,
                    trunc_penalty=args.trunc_penalty,
                    shaping_coef=args.shaping_coef,
                    trunc_shaping_coef=args.trunc_shaping_coef,
                    defend_mechanism=args.defend_mechanism,
                )

                ego_r = rally["ego_reward"]
                opp_r = rally["opp_reward"]
                rewards_this_iter.append(ego_r)
                crossings_this_iter.append(rally["crossings"])
                steps_this_iter.append(rally["steps"])
                for sk in rally["ego_attacks"]:
                    attack_usage[sk] += 1
                for dp in rally["ego_defends"]:
                    defend_usage[dp] += 1

                if not rally["done"]:
                    truncs += 1
                elif rally["winner"] == "ego":
                    wins += 1
                else:
                    losses += 1

                n_ego_decisions = len(rally["ego_old_values"])
                if n_ego_decisions > 0:
                    ego_rewards_seq = [0.0] * (n_ego_decisions - 1) + [ego_r]
                    ego_adv, ego_ret = compute_gae(ego_rewards_seq, rally["ego_old_values"],
                                                    args.gamma, args.gae_lambda,
                                                    bootstrap_value=rally["bootstrap_ego_value"])
                    batch_states += rally["ego_states"]
                    batch_attack_actions += rally["ego_attack_actions"]
                    batch_defend_actions += rally["ego_defend_actions"]
                    batch_old_attack_lp += rally["ego_old_attack_log_probs"]
                    batch_old_defend_lp += rally["ego_old_defend_log_probs"]
                    batch_advantages += ego_adv
                    batch_returns += ego_ret

                if rally["opp_uses_current"]:
                    n_opp_decisions = len(rally["opp_old_values"])
                    if n_opp_decisions > 0:
                        opp_rewards_seq = [0.0] * (n_opp_decisions - 1) + [opp_r]
                        opp_adv, opp_ret = compute_gae(opp_rewards_seq, rally["opp_old_values"],
                                                        args.gamma, args.gae_lambda,
                                                        bootstrap_value=rally["bootstrap_opp_value"])
                        batch_states += rally["opp_states"]
                        batch_attack_actions += rally["opp_attack_actions"]
                        batch_defend_actions += rally["opp_defend_actions"]
                        batch_old_attack_lp += rally["opp_old_attack_log_probs"]
                        batch_old_defend_lp += rally["opp_old_defend_log_probs"]
                        batch_advantages += opp_adv
                        batch_returns += opp_ret

                if rally_log_f is not None:
                    rally_log_id += 1
                    json.dump({
                        "iter": it,
                        "rally_id": rally_log_id,
                        "opp_type": opp_type,
                        "ego_attack_init": ego_attack_init, "opp_attack_init": opp_attack_init,
                        "ego_defend_init": ego_defend_init, "opp_defend_init": opp_defend_init,
                        "crossings": rally["crossings"],
                        "steps": rally["steps"],
                        "done": rally["done"],
                        "winner": rally["winner"],
                        "ego_reward": ego_r,
                        "opp_reward": opp_r,
                        "ego_attacks": rally["ego_attacks"], "ego_defends": rally["ego_defends"],
                        "opp_attacks": rally["opp_attacks"], "opp_defends": rally["opp_defends"],
                    }, rally_log_f)
                    rally_log_f.write("\n")

            n = args.rallies_per_iter
            mean_r = float(np.mean(rewards_this_iter))
            reward_std = float(np.std(rewards_this_iter))
            win_rate = wins / n
            loss_rate = losses / n
            trunc_rate = truncs / n
            draw_rate = trunc_rate
            mean_crossings = float(np.mean(crossings_this_iter))
            mean_steps = float(np.mean(steps_this_iter))

            total_attack_usage = sum(attack_usage)
            if total_attack_usage > 0:
                attack_usage_probs = np.array(attack_usage, dtype=np.float64) / total_attack_usage
                dominant_fraction_attack = float(attack_usage_probs.max())
                attack_ent = float(-np.sum(attack_usage_probs * np.log(attack_usage_probs + 1e-12)))
            else:
                dominant_fraction_attack = 0.0
                attack_ent = 0.0
            effective_n_attack = float(np.exp(attack_ent))

            total_defend_usage = sum(defend_usage)
            if total_defend_usage > 0:
                defend_usage_probs = np.array(defend_usage, dtype=np.float64) / total_defend_usage
                dominant_fraction_defend = float(defend_usage_probs.max())
                defend_ent = float(-np.sum(defend_usage_probs * np.log(defend_usage_probs + 1e-12)))
            else:
                dominant_fraction_defend = 0.0
                defend_ent = 0.0
            effective_n_defend = float(np.exp(defend_ent))

            # === Update phase: several epochs of clipped minibatch
            # updates over the batch just collected. Advantages are
            # normalized across the whole batch first (standard PPO
            # practice -- rallies vary a lot in length/scale here, so
            # this keeps the loss scale comparable iteration to
            # iteration). old_log_probs/returns are fixed targets from
            # collection time; only the forward pass is re-run (with
            # gradients) each epoch, against the policy as it stands
            # at that point in the epoch loop -- the ratio is exactly
            # what makes this PPO rather than plain actor-critic.
            n_decisions = len(batch_states)
            policy_losses, value_losses, entropies, grad_norms = [], [], [], []
            entropies_attack, entropies_defend = [], []
            if n_decisions > 0:
                states_t = torch.from_numpy(np.stack(batch_states)).float().to(device)
                attack_actions_t = torch.tensor(batch_attack_actions, dtype=torch.long, device=device)
                defend_actions_t = torch.tensor(batch_defend_actions, dtype=torch.long, device=device)
                old_lp_t = (torch.tensor(batch_old_attack_lp, dtype=torch.float32, device=device)
                            + torch.tensor(batch_old_defend_lp, dtype=torch.float32, device=device))
                returns_t = torch.tensor(batch_returns, dtype=torch.float32, device=device)
                adv_raw_t = torch.tensor(batch_advantages, dtype=torch.float32, device=device)
                adv_mean = float(adv_raw_t.mean().item())
                adv_std = float(adv_raw_t.std().item()) if adv_raw_t.numel() > 1 else 0.0
                adv_t = (adv_raw_t - adv_mean) / (adv_std + 1e-8) if adv_raw_t.numel() > 1 else adv_raw_t

                for _epoch in range(args.ppo_epochs):
                    perm = torch.randperm(n_decisions)
                    for start in range(0, n_decisions, args.minibatch_size):
                        idx = perm[start:start + args.minibatch_size]
                        attack_logits, defend_logits, values = policy(states_t[idx])
                        attack_dist = torch.distributions.Categorical(logits=attack_logits)
                        defend_dist = torch.distributions.Categorical(logits=defend_logits)
                        new_lp = (attack_dist.log_prob(attack_actions_t[idx])
                                  + defend_dist.log_prob(defend_actions_t[idx]))
                        ent_attack = attack_dist.entropy().mean()
                        ent_defend = defend_dist.entropy().mean()
                        entropy = ent_attack + ent_defend

                        policy_loss = ppo_clipped_loss(new_lp, old_lp_t[idx], adv_t[idx], args.clip_epsilon)
                        value_loss = F.mse_loss(values, returns_t[idx])
                        loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

                        optim.zero_grad()
                        loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                        optim.step()

                        policy_losses.append(float(policy_loss.item()))
                        value_losses.append(float(value_loss.item()))
                        entropies.append(float(entropy.item()))
                        entropies_attack.append(float(ent_attack.item()))
                        entropies_defend.append(float(ent_defend.item()))
                        grad_norms.append(float(grad_norm.item()))

                policy_loss_val = float(np.mean(policy_losses))
                value_loss_val = float(np.mean(value_losses))
                ent_val = float(np.mean(entropies))
                ent_attack_val = float(np.mean(entropies_attack))
                ent_defend_val = float(np.mean(entropies_defend))
                ent_bonus = float(args.entropy_coef * ent_val)
                grad_norm_val = float(np.mean(grad_norms))
            else:
                policy_loss_val = float('nan')
                value_loss_val = float('nan')
                ent_val = float('nan')
                ent_attack_val = float('nan')
                ent_defend_val = float('nan')
                ent_bonus = float('nan')
                grad_norm_val = float('nan')
                adv_mean = float('nan')
                adv_std = float('nan')

            with torch.no_grad():
                probe = torch.zeros(1, STATE_DIM_ATTACK_DEFEND, device=device)
                attack_logits_p, defend_logits_p, _value_p = policy(probe)
                p_attack_zero = F.softmax(attack_logits_p, dim=-1)[0].cpu().numpy()
                p_defend_zero = F.softmax(defend_logits_p, dim=-1)[0].cpu().numpy()

            wall_time = time.time() - iter_start_t

            log_w.writerow([
                it,
                mean_r, reward_std,
                win_rate, loss_rate, trunc_rate, draw_rate,
                mean_crossings, mean_steps,
            ] + list(attack_usage) + list(defend_usage) + [
                dominant_fraction_attack, dominant_fraction_defend, effective_n_attack, effective_n_defend,
            ] + list(p_attack_zero) + list(p_defend_zero) + [
                policy_loss_val, value_loss_val, ent_bonus, ent_val,
                ent_attack_val, ent_defend_val,
                grad_norm_val, adv_mean, adv_std, wall_time,
            ])
            log_f.flush()
            if rally_log_f is not None:
                rally_log_f.flush()

            if it % args.print_every == 0 or it == 1:
                print(f"[iter {it:4d}/{args.iterations}]  "
                      f"mean_r={mean_r:+.3f}  win={win_rate:.2f}  loss={loss_rate:.2f}  trunc={trunc_rate:.2f}  "
                      f"cross={mean_crossings:.1f}  dom_a={dominant_fraction_attack:.2f}  "
                      f"dom_d={dominant_fraction_defend:.2f}  "
                      f"pg={policy_loss_val:+.3f}  vf={value_loss_val:.3f}  "
                      f"ent_a={ent_attack_val:.3f}  ent_d={ent_defend_val:.3f}  "
                      f"grad={grad_norm_val:.2f}  buf={len(snapshot_buffer)}", flush=True)

            if it % args.snapshot_every == 0:
                snap = deepcopy(policy).eval()
                for p in snap.parameters():
                    p.requires_grad_(False)
                snapshot_buffer.append(snap)
                if len(snapshot_buffer) > args.snapshot_buffer_size:
                    snapshot_buffer.pop(0)

            if it % args.save_every == 0 or it == args.iterations:
                save_checkpoint(policy, args.output)
                if args.save_history:
                    save_checkpoint(policy, history_checkpoint_path(args.output, it))
    finally:
        env.close()
        log_f.close()
        if rally_log_f is not None:
            rally_log_f.close()
    print(f"\nDone. Saved policy to {args.output}, log to {args.log}"
          + (f", rallies to {args.log_rallies}" if args.log_rallies else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Design B self-play: two-headed (attack, defend) meta-policy")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--rallies-per-iter", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.05,
                        help="Entropy bonus alpha, applied to both heads' combined entropy.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor for GAE (default: 0.99 -- a standard default; "
                             "rallies here are short, a handful to a dozen decisions, so this "
                             "barely matters in practice, but 1.0 would still weight every "
                             "decision in a long rally equally, which GAE's lambda already "
                             "handles via the bias-variance tradeoff).")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda (bias-variance tradeoff for advantage estimation).")
    parser.add_argument("--clip-epsilon", type=float, default=0.2,
                        help="PPO clip range for the probability ratio (standard default).")
    parser.add_argument("--ppo-epochs", type=int, default=4,
                        help="Number of epochs of minibatch updates per collected batch.")
    parser.add_argument("--minibatch-size", type=int, default=64,
                        help="Minibatch size for PPO's update phase.")
    parser.add_argument("--value-coef", type=float, default=0.5,
                        help="Weight on the critic's MSE loss in the total loss (standard default).")
    parser.add_argument("--snapshot-prob", type=float, default=0.5)
    parser.add_argument("--snapshot-every", type=int, default=25)
    parser.add_argument("--snapshot-buffer-size", type=int, default=10)
    parser.add_argument("--epsilon-baseline", type=float, default=0.0,
                        help="Probability of sampling opp from the fixed-baseline pool "
                             "instead of snapshot/current (default: 0.0 = disabled).")
    parser.add_argument("--baseline-pool", nargs="+",
                        default=["random:random"],
                        help="Fixed (attack, defend) strategies as 'attack:defend' entries, "
                             "e.g. 'left:center' or 'random:random'.")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--save-history", action="store_true")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--trunc-penalty", type=float, default=TRUNCATED_PENALTY)
    parser.add_argument("--shaping-coef", type=float, default=0.0)
    parser.add_argument("--trunc-shaping-coef", type=float, default=0.0)
    parser.add_argument("--log-rallies", default=None)
    parser.add_argument("--defend-mechanism", choices=DEFEND_MECHANISMS, default=DEFAULT_DEFEND_MECHANISM,
                        help="'reactive' (receiver's approach bias) or 'preparation' "
                             "(the player who just hit prepares, paper Sec 3.2)")
    parser.add_argument("--reset-targets", action="store_true",
                        help="Restore both racket targets before every rally")
    args = parser.parse_args()

    train(args)
