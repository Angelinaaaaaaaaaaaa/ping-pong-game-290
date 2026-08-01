"""
Revised headless evaluation for the 5-skill Nash pipeline.

Merged version: teammate's probabilistic selection modes (softmax / epsilon_argmax /
epsilon_softmax + per-matchup RNGs + --model-dir override) + hailey's dual-model
head-to-head (--dual-* flags) + CUDA→CPU checkpoint fix.

Run from the project root:
    MUJOCO_GL=cgl venv/bin/python nash_skills/eval_matchup.py
    MUJOCO_GL=cgl venv/bin/python nash_skills/eval_matchup.py \
        --episodes 60 --steps 600 \
        --output-csv  skill_eval/matchup_results_5skill.csv \
        --output-json skill_eval/matchup_results_5skill.json

    # Dual head-to-head (hailey): player 1 uses model set A, player 2 uses set B
    MUJOCO_GL=cgl venv/bin/python nash_skills/eval_matchup.py \
        --v3-5skill --dual-model-p-a models/model_p_5skill_v3_tie.pth \
        --dual-model1-b models/model1_5skill_v3_discard.pth \
        --dual-model2-b models/model2_5skill_v3_discard.pth \
        --dual-model-p-b models/model_p_5skill_v3_discard.pth \
        --dual-strategy1 nash-p-hard --dual-strategy2 nash-p-hard --dual-swap
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import csv
import dataclasses
import io
import json
from contextlib import redirect_stdout
from typing import Dict, List, Optional

import numpy as np
import torch

from nash_skills.skills import SKILL_NAMES, N_SKILLS, skill_index, skill_from_index
from nash_skills.winner_inference import infer_terminal_winner
from nash_skills.v2.scorecard import compute_scorecard, format_scorecard


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

PPO_MODEL_PATH        = "logs/best_model_tracker1/best_model"
MODEL1_5SK_PATH       = "models/model1_5skill.pth"
MODEL2_5SK_PATH       = "models/model2_5skill.pth"
MODEL_P_5SK_PATH      = "models/model_p_5skill.pth"
# v2 4-skill pipeline — discounted returns, 76-dim state
MODEL1_V2_PATH        = "models/model1_v2.pth"
MODEL2_V2_PATH        = "models/model2_v2.pth"
MODEL_P_V2_PATH       = "models/model_p_v2.pth"
# v2 5-skill pipeline — discounted returns, 76-dim state, all 5 skills
MODEL1_5SK_V2_PATH    = "models/model1_5skill_v2.pth"
MODEL2_5SK_V2_PATH    = "models/model2_5skill_v2.pth"
MODEL_P_5SK_V2_PATH   = "models/model_p_5skill_v2.pth"
# v3 5-skill pipeline — same-state per-sample potential training
MODEL1_5SK_V3_PATH    = "models/model1_5skill_v3.pth"
MODEL2_5SK_V3_PATH    = "models/model2_5skill_v3.pth"
MODEL_P_5SK_V3_PATH   = "models/model_p_5skill_v3.pth"
# FactoredModel weights for the 5-skill v2 pipeline (116-dim).
MODEL1_5SK_FACTORED_PATH  = "models/model1_5skill_factored.pth"
MODEL2_5SK_FACTORED_PATH  = "models/model2_5skill_factored.pth"
MODEL_P_5SK_FACTORED_PATH = "models/model_p_5skill_factored.pth"
# FactoredModel weights for the 5-skill v3 pipeline.
MODEL1_5SK_V3_FACTORED_PATH  = "models/model1_5skill_v3_factored.pth"
MODEL2_5SK_V3_FACTORED_PATH  = "models/model2_5skill_v3_factored.pth"
MODEL_P_5SK_V3_FACTORED_PATH = "models/model_p_5skill_v3_factored.pth"

HISTORY = 4

TABLE_SHIFT = 1.5
TABLE_X_MIN = 0.0
TABLE_X_MAX = TABLE_SHIFT + 1.37
TABLE_Y_ABS_MAX = 0.75

VALID_STRATEGIES = [
    "nash-p-hard",      # joint argmax over full Φ table (optimistic)
    "nash-p-br",        # conditional best response fixing opp's current skill
    "nash-p-minimax",   # worst-case-safe: argmax over per-ego min-over-opp Φ
    "nash-p-adaptive",  # minimax scores + softmax when gap < margin
    "ibr",              # Q-based alternating best response (Φ-independent)
    "ibr-q",            # Q-based empirical-mix best response
    "nash-p",           # alias for nash-p-br (backwards compat)
    "random",
] + SKILL_NAMES

_LEARNED_STRATEGIES = {
    "nash-p-hard", "nash-p-br", "nash-p-minimax", "nash-p-adaptive", "ibr", "ibr-q", "nash-p"
}

_ALL_OPPONENTS = ["random", "left", "right", "left_short", "right_short", "center_safe"]

DEFAULT_MATCHUPS = [
    (strategy, opp)
    for strategy in ["nash-p-hard", "nash-p-br", "nash-p-minimax", "nash-p-adaptive", "ibr", "ibr-q"]
    for opp in _ALL_OPPONENTS
]
DEFAULT_MATCHUPS += [
    ("nash-p-hard", "ibr"),
    ("ibr", "nash-p-hard"),
    ("nash-p-br", "ibr"),
    ("ibr", "nash-p-br"),
    ("nash-p-minimax", "ibr"),
    ("ibr", "nash-p-minimax"),
    ("nash-p-adaptive", "ibr"),
    ("ibr", "nash-p-adaptive"),
    ("nash-p-hard", "ibr-q"),
    ("ibr-q", "nash-p-hard"),
    ("nash-p-br", "ibr-q"),
    ("ibr-q", "nash-p-br"),
    ("nash-p-minimax", "ibr-q"),
    ("ibr-q", "nash-p-minimax"),
    ("nash-p-adaptive", "ibr-q"),
    ("ibr-q", "nash-p-adaptive"),
]

LONG_RALLY_THRESHOLD = 100


# --------------------------------------------------------------------------- #
# Result dataclass
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class MatchupResult:
    strategy1: str
    strategy2: str
    episodes: int
    ego_wins: int
    opp_wins: int
    ego_contacts: int
    opp_contacts: int
    ego_successes: int
    opp_successes: int
    rally_lengths: List[int]
    # Fields added later — defaulted so old test helpers that omit them still work
    truncated_episodes: int = 0
    episode_steps: List[int] = dataclasses.field(default_factory=list)
    skill_usage: Dict[str, int] = dataclasses.field(default_factory=dict)
    total_steps: int = 0

    @property
    def win_rate(self) -> Optional[float]:
        if self.episodes == 0:
            return None
        return self.ego_wins / self.episodes

    @property
    def avg_rally_length(self) -> Optional[float]:
        if not self.rally_lengths:
            return None
        return float(np.mean(self.rally_lengths))

    @property
    def avg_steps_per_episode(self) -> Optional[float]:
        if not self.episode_steps:
            return None
        return float(np.mean(self.episode_steps))

    @property
    def ego_success_rate(self) -> Optional[float]:
        if self.ego_contacts == 0:
            return None
        return self.ego_successes / self.ego_contacts

    @property
    def opp_success_rate(self) -> Optional[float]:
        if self.opp_contacts == 0:
            return None
        return self.opp_successes / self.opp_contacts

    @property
    def done_episodes(self) -> int:
        return self.episodes

    @property
    def win_rate_clean(self) -> Optional[float]:
        if self.done_episodes == 0:
            return None
        return self.ego_wins / self.done_episodes


# --------------------------------------------------------------------------- #
# PPO observation builders
# --------------------------------------------------------------------------- #

def _build_obs1(obs, info):
    o = np.zeros(9 + 9 + 7 + 7 + 9 * HISTORY, dtype=np.float32)
    o[:9] = obs[:9]
    o[9:18] = obs[18:27]
    o[18:21] = info["diff_pos"]
    o[21:25] = info["diff_quat"]
    o[25:32] = info["target"]
    o[32:] = obs[42: 42 + HISTORY * 9]
    return o


def _build_obs2(obs, info):
    o = np.zeros(9 + 9 + 7 + 7 + 9 * HISTORY, dtype=np.float32)
    o[:9] = obs[9:18]
    o[9:18] = obs[27:36]
    o[18:21] = info["diff_pos_opp"]
    o[21:25] = info["diff_quat_opp"]
    o[25:32] = info["target_opp"]
    o[32:] = obs[42 + HISTORY * 9: 42 + 2 * HISTORY * 9]
    return o


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _safe_load_state_dict(path: str):
    """
    Load a PyTorch checkpoint and support older torch versions.
    Maps storages to CPU so checkpoints saved on CUDA load on CPU-only machines.
    """
    try:
        return torch.load(path, weights_only=True, map_location="cpu")
    except TypeError:
        return torch.load(path, map_location="cpu")


def _capture_env_step(env, action):
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = env.step(action)
    return result, buf.getvalue().splitlines()


def _parse_contact_lines(lines):
    ego_contacts = 0
    opp_contacts = 0
    ego_successes = 0
    opp_successes = 0

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue

        try:
            x_land = float(parts[-2])
            y_land = float(parts[-1])
        except ValueError:
            continue

        if "by ego" in line:
            ego_contacts += 1
            if TABLE_SHIFT < x_land < TABLE_X_MAX and abs(y_land) < TABLE_Y_ABS_MAX:
                ego_successes += 1

        elif "by opp" in line:
            opp_contacts += 1
            if TABLE_X_MIN < x_land < TABLE_SHIFT and abs(y_land) < TABLE_Y_ABS_MAX:
                opp_successes += 1

    return ego_contacts, opp_contacts, ego_successes, opp_successes


def _infer_winner(obs, info):
    # Delegates to the shared inference: explicit env winner (info['winner']) ->
    # racket-boundary reconstruction -> ball x-velocity -> position fallback.
    # Without this active return the function yielded None and run_matchup scored
    # EVERY completed episode as an opp win (win_rate flat 0.0 for all matchups).
    return infer_terminal_winner(obs, info, fallback="position") or "opp"


def _initial_skill_idx(name: str, fallback: int = 0) -> int:
    try:
        return skill_index(name)
    except Exception:
        return fallback


# --------------------------------------------------------------------------- #
# Strategy picker
# --------------------------------------------------------------------------- #

def _build_phi_table(obs_vec, info, player, model_p, state_encoder_fn):
    if state_encoder_fn is not None and info is not None:
        base_vec = state_encoder_fn(obs_vec, info, player)
    else:
        base_vec = obs_vec
    rows = []
    for ego_s in range(N_SKILLS):
        for opp_s in range(N_SKILLS):
            base = torch.tensor(base_vec, dtype=torch.float32)
            base[-2] = ego_s / (N_SKILLS - 1)
            base[-1] = opp_s / (N_SKILLS - 1)
            rows.append(base)
    batch = torch.stack(rows)
    with torch.no_grad():
        vals = model_p(batch)[:, 0]
    return vals.reshape(N_SKILLS, N_SKILLS)


def _pick_with_softmax_fallback(
    action_scores: torch.Tensor,
    tau: float,
    confidence_margin: float,
) -> int:
    action_scores = action_scores.reshape(-1)
    if action_scores.numel() == 1:
        return 0

    sorted_scores, _ = torch.sort(action_scores, descending=True)
    gap = (sorted_scores[0] - sorted_scores[1]).item()
    if gap >= confidence_margin or tau <= 0:
        return int(torch.argmax(action_scores).item())

    probs = torch.softmax(action_scores / tau, dim=0)
    return int(torch.multinomial(probs, 1).item())


def make_picker(strategy: str, model_p, state_encoder_fn=None,
                tau: float = 0.2, confidence_margin: float = 0.05,
                model1=None, model2=None,
                selection_mode: str = "argmax",
                temperature: float = 1.0,
                epsilon: float = 0.0,
                rng=None):
    """
    Return pick_fn(player, obs_vec, other_skill_idx, info=None) -> skill_idx.

    See docstring in original file for strategy descriptions.
    """
    if strategy in SKILL_NAMES:
        fixed_idx = skill_index(strategy)
        return lambda _player, _obs, _other, _info=None: fixed_idx

    if strategy == "random":
        return lambda _player, _obs, _other, _info=None: np.random.randint(N_SKILLS)

    _PHI_STRATEGIES = {"nash-p-hard", "nash-p-br", "nash-p", "nash-p-minimax", "nash-p-adaptive"}
    if strategy in _PHI_STRATEGIES and model_p is None:
        raise ValueError(
            f"Strategy '{strategy}' requires a loaded potential model. "
            "Pass it to make_picker(model_p=...)."
        )

    # Dispatch helper: converts ego action_scores tensor → skill index.
    # 'argmax' preserves the existing confidence-margin softmax fallback;
    # other modes use select_skill_from_values from skill_selection.py.
    if selection_mode == "argmax":
        def _dispatch(action_scores: "torch.Tensor") -> int:
            return _pick_with_softmax_fallback(action_scores, tau, confidence_margin)
    else:
        from nash_skills.v2.skill_selection import select_skill_from_values as _ssv

        def _dispatch(action_scores: "torch.Tensor") -> int:
            return _ssv(
                action_scores.numpy(),
                mode=selection_mode,
                temperature=temperature,
                epsilon=epsilon,
                rng=rng,
            )

    if strategy == "nash-p-hard":
        def pick_hard(player, obs_vec, _other_skill_idx, info=None):
            phi = _build_phi_table(obs_vec, info, player, model_p, state_encoder_fn)
            action_scores = phi.max(dim=1).values
            return _dispatch(action_scores)
        return pick_hard

    if strategy in ("nash-p-br", "nash-p"):
        def pick_br(player, obs_vec, other_skill_idx, info=None):
            phi = _build_phi_table(obs_vec, info, player, model_p, state_encoder_fn)
            if player == 1:
                action_scores = phi[:, other_skill_idx]
            else:
                action_scores = -phi[other_skill_idx, :]
            return _dispatch(action_scores)
        return pick_br

    if strategy == "nash-p-minimax":
        def pick_minimax(player, obs_vec, other_skill_idx, info=None):
            phi = _build_phi_table(obs_vec, info, player, model_p, state_encoder_fn)
            if player == 1:
                action_scores = phi.min(dim=1).values
            else:
                action_scores = phi.min(dim=0).values
            return _dispatch(action_scores)
        return pick_minimax

    if strategy == "nash-p-adaptive":
        def pick_adaptive(player, obs_vec, other_skill_idx, info=None):
            phi = _build_phi_table(obs_vec, info, player, model_p, state_encoder_fn)
            if player == 1:
                action_scores = phi.min(dim=1).values
            else:
                action_scores = phi.min(dim=0).values
            return _dispatch(action_scores)
        return pick_adaptive

    if strategy == "ibr":
        if model1 is None or model2 is None:
            raise ValueError(
                "ibr requires model1 and model2 Q-value models. "
                "Pass them to make_picker(model1=..., model2=...)."
            )

        IBR_STEPS = 10

        def pick_ibr(player, obs_vec, other_skill_idx, info=None):
            if state_encoder_fn is not None and info is not None:
                base_enc = torch.tensor(
                    state_encoder_fn(obs_vec, info, player), dtype=torch.float32
                )
            else:
                base_enc = torch.tensor(obs_vec, dtype=torch.float32)

            s1 = other_skill_idx
            s2 = other_skill_idx

            for _ in range(IBR_STEPS):
                q1_vals = []
                for ego_s in range(N_SKILLS):
                    x = base_enc.clone().unsqueeze(0)
                    x[0, -2] = ego_s / (N_SKILLS - 1)
                    x[0, -1] = s2    / (N_SKILLS - 1)
                    with torch.no_grad():
                        q1_vals.append(model1(x).item())
                s1 = int(np.argmax(q1_vals))

                q2_vals = []
                for opp_s in range(N_SKILLS):
                    x = base_enc.clone().unsqueeze(0)
                    x[0, -2] = s1    / (N_SKILLS - 1)
                    x[0, -1] = opp_s / (N_SKILLS - 1)
                    with torch.no_grad():
                        q2_vals.append(model2(x).item())
                s2 = int(np.argmin(q2_vals))

            return s1 if player == 1 else s2

        return pick_ibr

    if strategy == "ibr-q":
        if model1 is None or model2 is None:
            raise ValueError(
                "ibr-q requires model1 and model2 Q-value models. "
                "Pass them to make_picker(model1=..., model2=...)."
            )

        opp_counts = [1.0] * N_SKILLS

        def pick_ibr_q(player, obs_vec, other_skill_idx, info=None):
            opp_counts[other_skill_idx] += 1.0
            total = sum(opp_counts)
            opp_mix = [c / total for c in opp_counts]

            if state_encoder_fn is not None and info is not None:
                base_enc = torch.tensor(
                    state_encoder_fn(obs_vec, info, player), dtype=torch.float32
                )
            else:
                base_enc = torch.tensor(obs_vec, dtype=torch.float32)

            q_model = model1 if player == 1 else model2
            ego_vals = []
            for ego_s in range(N_SKILLS):
                val = 0.0
                for opp_s in range(N_SKILLS):
                    x = base_enc.clone().unsqueeze(0)
                    if player == 1:
                        x[0, -2] = ego_s / (N_SKILLS - 1)
                        x[0, -1] = opp_s / (N_SKILLS - 1)
                    else:
                        x[0, -2] = opp_s / (N_SKILLS - 1)
                        x[0, -1] = ego_s / (N_SKILLS - 1)
                    with torch.no_grad():
                        val += opp_mix[opp_s] * q_model(x).item()
                ego_vals.append(val)
            return int(np.argmax(ego_vals))

        return pick_ibr_q

    raise ValueError(
        f"Unknown strategy '{strategy}'. "
        f"Choose from: {VALID_STRATEGIES}"
    )


# --------------------------------------------------------------------------- #
# Core eval loop (hailey's refactor for dual-model support)
# --------------------------------------------------------------------------- #

def _run_eval_loop(
    pick1,
    pick2,
    strategy1: str,
    strategy2: str,
    ppo,
    n_episodes: int,
    max_steps_per_episode: int,
    warmup_steps: int = 300,
    max_total_steps: Optional[int] = None,
) -> MatchupResult:
    """
    Core headless eval loop. Takes pre-built pickers, runs `n_episodes`
    completed rallies, returns a MatchupResult.

    Used by both `run_matchup` (single model set) and `run_matchup_dual`
    (player A and player B use different model sets).
    """
    from nash_skills.env_wrapper import SkillEnv

    env = SkillEnv(proc_id=1, history=HISTORY)

    curr_idx1 = _initial_skill_idx("left", 0)
    curr_idx2 = _initial_skill_idx("right", 0)
    env.set_skills(skill_from_index(curr_idx1), skill_from_index(curr_idx2))

    obs, info = env.reset()
    prev_ball_x = float(obs[36])

    total_steps = 0

    # One-time warmup
    while total_steps < warmup_steps:
        obs1 = _build_obs1(obs, info)
        obs2 = _build_obs2(obs, info)

        action1, _ = ppo.predict(obs1, deterministic=True)
        action2, _ = ppo.predict(obs2, deterministic=True)

        action = np.zeros(18)
        action[:9] = action1[:9]
        action[9:] = action2[:9]

        (obs, _, done, _, info), _ = _capture_env_step(env, action)
        total_steps += 1

        curr_ball_x = float(obs[36])

        if (prev_ball_x - TABLE_SHIFT) * (curr_ball_x - TABLE_SHIFT) < 0:
            curr_idx1 = pick1(1, obs, curr_idx2, info)
            curr_idx2 = pick2(2, obs, curr_idx1, info)
            env.set_skills(skill_from_index(curr_idx1), skill_from_index(curr_idx2))

        prev_ball_x = curr_ball_x

        if done:
            env.set_skills(skill_from_index(curr_idx1), skill_from_index(curr_idx2))
            obs, info = env.reset()
            prev_ball_x = float(obs[36])

    # Counted evaluation
    completed_episodes = 0
    truncated_episodes = 0

    ego_wins = 0
    opp_wins = 0

    ego_contacts = 0
    opp_contacts = 0
    ego_successes = 0
    opp_successes = 0

    rally_lengths: List[int] = []
    episode_steps: List[int] = []

    skill_usage: Dict[str, int] = {name: 0 for name in SKILL_NAMES}

    curr_rally_len = 0
    steps_in_episode = 0

    while completed_episodes < n_episodes:
        if max_total_steps is not None and total_steps >= max_total_steps:
            print(
                f"WARNING: hit max_total_steps={max_total_steps} with "
                f"{completed_episodes}/{n_episodes} done episodes collected. "
                "Stopping early - results are based on done episodes only."
            )
            break

        obs1 = _build_obs1(obs, info)
        obs2 = _build_obs2(obs, info)

        action1, _ = ppo.predict(obs1, deterministic=True)
        action2, _ = ppo.predict(obs2, deterministic=True)

        action = np.zeros(18)
        action[:9] = action1[:9]
        action[9:] = action2[:9]

        (obs, _, done, _, info), lines = _capture_env_step(env, action)

        total_steps += 1
        steps_in_episode += 1

        e_c, o_c, e_s, o_s = _parse_contact_lines(lines)
        ego_contacts += e_c
        opp_contacts += o_c
        ego_successes += e_s
        opp_successes += o_s

        curr_ball_x = float(obs[36])

        if (prev_ball_x - TABLE_SHIFT) * (curr_ball_x - TABLE_SHIFT) < 0:
            curr_rally_len += 1
            curr_idx1 = pick1(1, obs, curr_idx2, info)
            curr_idx2 = pick2(2, obs, curr_idx1, info)
            env.set_skills(skill_from_index(curr_idx1), skill_from_index(curr_idx2))

            if strategy1 in _LEARNED_STRATEGIES:
                skill_usage[skill_from_index(curr_idx1)] += 1

        prev_ball_x = curr_ball_x

        if done:
            winner = _infer_winner(obs, info)
            if winner == "ego":
                ego_wins += 1
            else:
                opp_wins += 1

            rally_lengths.append(curr_rally_len)
            episode_steps.append(steps_in_episode)
            completed_episodes += 1

            curr_rally_len = 0
            steps_in_episode = 0

            env.set_skills(skill_from_index(curr_idx1), skill_from_index(curr_idx2))
            obs, info = env.reset()
            prev_ball_x = float(obs[36])
            continue

        if steps_in_episode >= max_steps_per_episode:
            truncated_episodes += 1
            curr_rally_len = 0
            steps_in_episode = 0

            env.set_skills(skill_from_index(curr_idx1), skill_from_index(curr_idx2))
            obs, info = env.reset()
            prev_ball_x = float(obs[36])

    env.close()

    if max_total_steps is not None and completed_episodes < n_episodes:
        print(
            f"WARNING: only completed {completed_episodes}/{n_episodes} episodes "
            f"before hitting max_total_steps={max_total_steps}"
        )

    return MatchupResult(
        strategy1=strategy1,
        strategy2=strategy2,
        episodes=completed_episodes,
        truncated_episodes=truncated_episodes,
        ego_wins=ego_wins,
        opp_wins=opp_wins,
        ego_contacts=ego_contacts,
        opp_contacts=opp_contacts,
        ego_successes=ego_successes,
        opp_successes=opp_successes,
        rally_lengths=rally_lengths,
        episode_steps=episode_steps,
        skill_usage=skill_usage,
        total_steps=total_steps,
    )


def run_matchup(
    strategy1: str,
    strategy2: str,
    ppo,
    model_p,
    n_episodes: int,
    max_steps_per_episode: int,
    warmup_steps: int = 300,
    max_total_steps: Optional[int] = None,
    state_encoder_fn=None,
    tau: float = 0.2,
    confidence_margin: float = 0.05,
    model1=None,
    model2=None,
    selection_mode: str = "argmax",
    temperature: float = 1.0,
    epsilon: float = 0.0,
    rng1=None,
    rng2=None,
) -> MatchupResult:
    """
    Run one 5-skill matchup headlessly. Both players use the same model set.

    - rng1 / rng2: independent numpy Generators for player 1 / player 2
      probabilistic skill selection. Keep them separate so each player's
      sampling is reproducible but not correlated with the other's.
    """
    pick1 = make_picker(strategy1, model_p, state_encoder_fn=state_encoder_fn,
                        tau=tau, confidence_margin=confidence_margin,
                        model1=model1, model2=model2,
                        selection_mode=selection_mode, temperature=temperature,
                        epsilon=epsilon, rng=rng1)
    pick2 = make_picker(strategy2, model_p, state_encoder_fn=state_encoder_fn,
                        tau=tau, confidence_margin=confidence_margin,
                        model1=model1, model2=model2,
                        selection_mode=selection_mode, temperature=temperature,
                        epsilon=epsilon, rng=rng2)
    return _run_eval_loop(
        pick1, pick2, strategy1, strategy2, ppo,
        n_episodes=n_episodes,
        max_steps_per_episode=max_steps_per_episode,
        warmup_steps=warmup_steps,
        max_total_steps=max_total_steps,
    )


def run_matchup_dual(
    strategy1: str,
    strategy2: str,
    ppo,
    model_p_a,
    model_p_b,
    n_episodes: int,
    max_steps_per_episode: int,
    warmup_steps: int = 300,
    max_total_steps: Optional[int] = None,
    state_encoder_fn=None,
    tau: float = 0.2,
    confidence_margin: float = 0.05,
    model1_a=None, model2_a=None,
    model1_b=None, model2_b=None,
    selection_mode: str = "argmax",
    temperature: float = 1.0,
    epsilon: float = 0.0,
    rng1=None,
    rng2=None,
) -> MatchupResult:
    """
    Head-to-head: player 1 uses model set A, player 2 uses model set B.

    Lets you compare two trained pipelines (e.g. tie-trained Q+Phi vs
    discard-trained Q+Phi) by having them play each other directly.
    """
    pick1 = make_picker(strategy1, model_p_a, state_encoder_fn=state_encoder_fn,
                        tau=tau, confidence_margin=confidence_margin,
                        model1=model1_a, model2=model2_a,
                        selection_mode=selection_mode, temperature=temperature,
                        epsilon=epsilon, rng=rng1)
    pick2 = make_picker(strategy2, model_p_b, state_encoder_fn=state_encoder_fn,
                        tau=tau, confidence_margin=confidence_margin,
                        model1=model1_b, model2=model2_b,
                        selection_mode=selection_mode, temperature=temperature,
                        epsilon=epsilon, rng=rng2)
    return _run_eval_loop(
        pick1, pick2, strategy1, strategy2, ppo,
        n_episodes=n_episodes,
        max_steps_per_episode=max_steps_per_episode,
        warmup_steps=warmup_steps,
        max_total_steps=max_total_steps,
    )


# --------------------------------------------------------------------------- #
# Skill usage helpers
# --------------------------------------------------------------------------- #

def most_used_skill(result: MatchupResult) -> Optional[str]:
    if not result.skill_usage:
        return None
    total = sum(result.skill_usage.values())
    if total == 0:
        return None
    return max(result.skill_usage, key=result.skill_usage.get)


def dominant_skill_fraction(result: MatchupResult) -> Optional[float]:
    if not result.skill_usage:
        return None
    total = sum(result.skill_usage.values())
    if total == 0:
        return None
    return max(result.skill_usage.values()) / total


# --------------------------------------------------------------------------- #
# Shared scorecard (nash_skills/v2/scorecard.py, meeting note item 19)         #
# --------------------------------------------------------------------------- #

def matchup_scorecard(result: MatchupResult) -> dict:
    """
    Adapt a MatchupResult into the shared scorecard metric set: adds median
    rally length, skill-usage entropy, and dominant-skill fraction on top of
    what print_summary/save_csv already report.
    """
    return compute_scorecard(
        wins=result.ego_wins,
        losses=result.opp_wins,
        truncated=result.truncated_episodes,
        rally_lengths=result.rally_lengths,
        skill_usage=result.skill_usage,
    )


def print_full_scorecards(results: List[MatchupResult], file=None) -> None:
    """Print the full shared scorecard for every matchup result, one block each."""
    if file is None:
        file = sys.stdout
    for r in results:
        label = f"{r.strategy1} vs {r.strategy2}"
        sc = matchup_scorecard(r)
        print(format_scorecard(sc, label=label), file=file)
        print(file=file)


# --------------------------------------------------------------------------- #
# CSV / summary
# --------------------------------------------------------------------------- #

def save_csv(results: List[MatchupResult], path: str):
    rows = []
    for r in results:
        row = {
            "strategy1": r.strategy1,
            "strategy2": r.strategy2,
            "episodes": r.episodes,
            "truncated_episodes": r.truncated_episodes,
            "ego_wins": r.ego_wins,
            "opp_wins": r.opp_wins,
            "win_rate": round(r.win_rate, 4) if r.win_rate is not None else "",
            "total_steps": r.total_steps,
            "avg_steps_per_episode": round(r.avg_steps_per_episode, 2) if r.avg_steps_per_episode is not None else "",
            "ego_contacts": r.ego_contacts,
            "opp_contacts": r.opp_contacts,
            "ego_successes": r.ego_successes,
            "opp_successes": r.opp_successes,
            "ego_success_rate": round(r.ego_success_rate, 4) if r.ego_success_rate is not None else "",
            "opp_success_rate": round(r.opp_success_rate, 4) if r.opp_success_rate is not None else "",
            "avg_rally_length": round(r.avg_rally_length, 2) if r.avg_rally_length is not None else "",
            "most_used_skill": most_used_skill(r) or "",
            "dominant_fraction": round(dominant_skill_fraction(r), 4) if dominant_skill_fraction(r) is not None else "",
        }
        for skill_name in SKILL_NAMES:
            row[f"usage_{skill_name}"] = r.skill_usage.get(skill_name, 0)
        rows.append(row)

    if not rows:
        return

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(results: List[MatchupResult], file=None,
                  title: str = "5-SKILL STRATEGY EVALUATION RESULTS"):
    if file is None:
        file = sys.stdout

    header = (
        f"{'Matchup':<30} "
        f"{'CompEp':>7} "
        f"{'Trunc':>7} "
        f"{'WinRate':>8} "
        f"{'AvgStep':>8} "
        f"{'AvgRally':>9} "
        f"{'EgoSucc%':>9}"
    )
    sep = "-" * len(header)

    print(f"\n{title:^{len(header)}}", file=file)
    print(sep, file=file)
    print(header, file=file)
    print(sep, file=file)

    for r in results:
        matchup = f"{r.strategy1} vs {r.strategy2}"
        wr = f"{r.win_rate:.0%}" if r.win_rate is not None else "---"
        avg_step = f"{r.avg_steps_per_episode:.1f}" if r.avg_steps_per_episode is not None else "---"
        avg_rally = f"{r.avg_rally_length:.1f}" if r.avg_rally_length is not None else "---"
        esr = f"{r.ego_success_rate:.0%}" if r.ego_success_rate is not None else "---"

        print(
            f"{matchup:<30} "
            f"{r.episodes:>7} "
            f"{r.truncated_episodes:>7} "
            f"{wr:>8} "
            f"{avg_step:>8} "
            f"{avg_rally:>9} "
            f"{esr:>9}",
            file=file,
        )

    print(sep, file=file)

    scored = [(r.win_rate, r) for r in results if r.win_rate is not None]
    if scored:
        best = max(scored, key=lambda x: x[0])
        worst = min(scored, key=lambda x: x[0])
        print(
            f"\nBest matchup:  {best[1].strategy1} vs {best[1].strategy2} "
            f"— {best[0]:.0%}",
            file=file,
        )
        print(
            f"Worst matchup: {worst[1].strategy1} vs {worst[1].strategy2} "
            f"— {worst[0]:.0%}",
            file=file,
        )


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #

def analyse_results(results: List[MatchupResult]) -> dict:
    by_opp = {r.strategy2: r for r in results}

    cs_result = by_opp.get("center_safe")
    if cs_result is not None and cs_result.avg_rally_length is not None:
        other_avgs = [
            r.avg_rally_length
            for r in results
            if r.strategy2 != "center_safe" and r.avg_rally_length is not None
        ]
        if other_avgs:
            mean_other = sum(other_avgs) / len(other_avgs)
            center_safe_long = cs_result.avg_rally_length > max(
                LONG_RALLY_THRESHOLD,
                mean_other * 3,
            )
        else:
            center_safe_long = cs_result.avg_rally_length > LONG_RALLY_THRESHOLD
    else:
        center_safe_long = False

    ls_result = by_opp.get("left_short")
    left_short_win_rate = ls_result.win_rate if ls_result is not None else None

    other_non_artifact_wrs = [
        r.win_rate
        for r in results
        if r.win_rate is not None and r.strategy2 not in ("center_safe", "left_short")
    ]
    baseline_wr = (
        sum(other_non_artifact_wrs) / len(other_non_artifact_wrs)
        if other_non_artifact_wrs
        else 0.5
    )

    left_short_problematic = (
        left_short_win_rate is not None and left_short_win_rate < 0.30
    )

    if center_safe_long and left_short_problematic:
        recommendation = "reduce_to_4"
    elif center_safe_long:
        recommendation = "drop_center_safe"
    elif left_short_problematic:
        recommendation = "drop_left_short"
    elif baseline_wr >= 0.45:
        recommendation = "keep_all_5"
    else:
        recommendation = "accept_as_final"

    return {
        "center_safe_long_rallies": center_safe_long,
        "left_short_win_rate": left_short_win_rate,
        "recommendation": recommendation,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="5-skill Nash matchup evaluator (5-skill v1 or v2 pipeline)"
    )
    parser.add_argument("--episodes", type=int, default=60,
        help="Number of COMPLETED episodes per matchup (default: 60)")
    parser.add_argument("--steps", type=int, default=600,
        help="Maximum steps per episode before truncation/reset (default: 600)")
    parser.add_argument("--warmup", type=int, default=300,
        help="One-time warmup steps before counting results (default: 300)")
    parser.add_argument("--max-total-steps", type=int, default=None,
        help="Optional safety cap on total simulator steps per matchup.")
    parser.add_argument("--output-csv", default="skill_eval/matchup_results_5skill.csv")
    parser.add_argument("--output-json", default="skill_eval/matchup_results_5skill.json")
    parser.add_argument("--v2", action="store_true", default=False,
        help="Use the 4-skill v2 pipeline.")
    parser.add_argument("--v2-5skill", action="store_true", default=False, dest="v2_5skill",
        help="Use the 5-skill v2 pipeline.")
    parser.add_argument("--v3-5skill", action="store_true", default=False, dest="v3_5skill",
        help="Use the 5-skill v3 pipeline.")
    parser.add_argument("--arch", choices=["simple", "factored"], default="simple",
        help="Estimator architecture: simple or factored")
    parser.add_argument("--tau", type=float, default=0.2,
        help="Softmax temperature for flat-surface fallback (default: 0.2)")
    parser.add_argument("--confidence-margin", type=float, default=0.05, dest="confidence_margin",
        help="Top-2 score gap below which nash-p-hard/br/adaptive use softmax (default: 0.05)")
    parser.add_argument("--selection-mode", default="argmax", dest="selection_mode",
        choices=["argmax", "softmax", "epsilon_argmax", "epsilon_softmax"],
        help="Skill-selection mode after computing action scores from Φ.")
    parser.add_argument("--temperature", type=float, default=1.0,
        help="Softmax temperature for --selection-mode softmax/epsilon_softmax (default: 1.0)")
    parser.add_argument("--epsilon", type=float, default=0.0,
        help="Exploration rate in [0,1] for --selection-mode epsilon_argmax/epsilon_softmax")
    parser.add_argument("--seed", type=int, default=None,
        help="RNG seed for probabilistic selection modes")
    parser.add_argument("--model-dir", default=None, dest="model_dir",
        help="Override the directory from which model .pth files are loaded")
    # ----------------------------- Dual mode --------------------------------
    parser.add_argument("--dual-model1-a", default=None,
        help="Override Q model 1 path for player A")
    parser.add_argument("--dual-model2-a", default=None,
        help="Override Q model 2 path for player A")
    parser.add_argument("--dual-model-p-a", default=None,
        help="Override potential model path for player A")
    parser.add_argument("--dual-model1-b", default=None,
        help="Q model 1 path for player B (enables dual mode)")
    parser.add_argument("--dual-model2-b", default=None,
        help="Q model 2 path for player B")
    parser.add_argument("--dual-model-p-b", default=None,
        help="Potential model path for player B")
    parser.add_argument("--dual-strategy1", default="nash-p-hard",
        help="Strategy for player 1 in dual mode (default: nash-p-hard)")
    parser.add_argument("--dual-strategy2", default="nash-p-hard",
        help="Strategy for player 2 in dual mode (default: nash-p-hard)")
    parser.add_argument("--dual-swap", action="store_true",
        help="Also run swapped match (B as p1, A as p2)")
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from model_arch import SimpleModel, FactoredModel

    print("Loading models...")
    ppo = PPO.load(PPO_MODEL_PATH)

    def _model_path(default_path: str) -> str:
        if args.model_dir is None:
            return default_path
        return os.path.join(args.model_dir, os.path.basename(default_path))

    if args.arch == "factored":
        if args.v3_5skill:
            model_p_path = _model_path(MODEL_P_5SK_V3_FACTORED_PATH)
            pipeline_tag = "v3-5skill-factored"
        elif args.v2_5skill:
            model_p_path = _model_path(MODEL_P_5SK_FACTORED_PATH)
            pipeline_tag = "v2-5skill-factored"
        else:
            raise SystemExit(
                "--arch factored requires --v2-5skill or --v3-5skill."
            )
        model_p = FactoredModel(state_dim=74, skill_dim=2, last_layer_activation=None)
    elif args.v3_5skill:
        from nash_skills.v2.state_encoder import STATE_DIM as V2_STATE_DIM
        model_p_path = _model_path(MODEL_P_5SK_V3_PATH)
        model_p = SimpleModel(V2_STATE_DIM, [64, 32, 16], 1, last_layer_activation=None)
        pipeline_tag = "v3-5skill"
    elif args.v2_5skill:
        from nash_skills.v2.state_encoder import STATE_DIM as V2_STATE_DIM
        model_p_path = _model_path(MODEL_P_5SK_V2_PATH)
        model_p = SimpleModel(V2_STATE_DIM, [64, 32, 16], 1, last_layer_activation=None)
        pipeline_tag = "v2-5skill"
    elif args.v2:
        from nash_skills.v2.state_encoder import STATE_DIM as V2_STATE_DIM
        model_p_path = _model_path(MODEL_P_V2_PATH)
        model_p = SimpleModel(V2_STATE_DIM, [64, 32, 16], 1, last_layer_activation=None)
        pipeline_tag = "v2-4skill"
    else:
        model_p_path = _model_path(MODEL_P_5SK_PATH)
        model_p = SimpleModel(116, [64, 32, 16], 1, last_layer_activation=None)
        pipeline_tag = "v1-5skill"

    # Dual mode: optionally override player-A potential path
    if args.dual_model_p_a:
        model_p_path = args.dual_model_p_a
    model_p.load_state_dict(_safe_load_state_dict(model_p_path))
    model_p.eval()

    # Q-value models — needed for ibr / ibr-q
    needs_q = any(s in {"ibr", "ibr-q"} for s, _ in DEFAULT_MATCHUPS) or any(
        s in {"ibr", "ibr-q"} for _, s in DEFAULT_MATCHUPS
    )
    model1 = model2 = None
    if needs_q:
        if args.arch == "factored":
            if args.v3_5skill:
                _q1_path = _model_path(MODEL1_5SK_V3_FACTORED_PATH)
                _q2_path = _model_path(MODEL2_5SK_V3_FACTORED_PATH)
            else:
                _q1_path = _model_path(MODEL1_5SK_FACTORED_PATH)
                _q2_path = _model_path(MODEL2_5SK_FACTORED_PATH)
            model1 = FactoredModel(state_dim=74, skill_dim=2)
            model2 = FactoredModel(state_dim=74, skill_dim=2)
        else:
            if args.v3_5skill:
                from nash_skills.v2.state_encoder import STATE_DIM as V2_STATE_DIM
                _sdim = V2_STATE_DIM
                _q1_path = _model_path(MODEL1_5SK_V3_PATH)
                _q2_path = _model_path(MODEL2_5SK_V3_PATH)
            elif args.v2_5skill:
                from nash_skills.v2.state_encoder import STATE_DIM as V2_STATE_DIM
                _sdim = V2_STATE_DIM
                _q1_path = _model_path(MODEL1_5SK_V2_PATH)
                _q2_path = _model_path(MODEL2_5SK_V2_PATH)
            elif args.v2:
                from nash_skills.v2.state_encoder import STATE_DIM as V2_STATE_DIM
                _sdim = V2_STATE_DIM
                _q1_path = _model_path(MODEL1_V2_PATH)
                _q2_path = _model_path(MODEL2_V2_PATH)
            else:
                _sdim = 116
                _q1_path = _model_path(MODEL1_5SK_PATH)
                _q2_path = _model_path(MODEL2_5SK_PATH)
            model1 = SimpleModel(_sdim, [64, 32, 16], 1)
            model2 = SimpleModel(_sdim, [64, 32, 16], 1)
        # Dual mode: optionally override player-A Q paths
        if args.dual_model1_a:
            _q1_path = args.dual_model1_a
        if args.dual_model2_a:
            _q2_path = args.dual_model2_a
        model1.load_state_dict(_safe_load_state_dict(_q1_path))
        model2.load_state_dict(_safe_load_state_dict(_q2_path))
        model1.eval()
        model2.eval()
        print(f"  Loaded Q-models:    {_q1_path}, {_q2_path}")

    if args.v3_5skill or args.v2_5skill or args.v2:
        from nash_skills.v2.state_encoder import encode_ego, encode_opp

        def _v2_state_encoder(obs, info, player):
            if player == 1:
                return encode_ego(obs, info)
            else:
                return encode_opp(obs, info)

        state_encoder_fn = _v2_state_encoder
    else:
        state_encoder_fn = None

    print(f"  Loaded PPO:         {PPO_MODEL_PATH}")
    print(f"  Loaded potential:   {model_p_path}  ({pipeline_tag})")

    # ------------------------------------------------------------------ #
    # Dual mode: head-to-head model A (loaded above) vs model B          #
    # ------------------------------------------------------------------ #
    dual_paths = (args.dual_model1_b, args.dual_model2_b, args.dual_model_p_b)
    if any(dual_paths):
        if not all(dual_paths):
            raise SystemExit(
                "Dual mode requires all three: --dual-model1-b, --dual-model2-b, "
                "--dual-model-p-b. Got " + repr(dual_paths)
            )
        if args.arch == "factored":
            raise SystemExit("Dual mode currently only supports --arch simple.")

        sdim_a = model_p.batch_norm.num_features
        print(f"\nLoading model set B (state_dim={sdim_a}) ...")
        model1_b = SimpleModel(sdim_a, [64, 32, 16], 1)
        model2_b = SimpleModel(sdim_a, [64, 32, 16], 1)
        model_p_b = SimpleModel(sdim_a, [64, 32, 16], 1, last_layer_activation=None)
        model1_b.load_state_dict(_safe_load_state_dict(args.dual_model1_b))
        model2_b.load_state_dict(_safe_load_state_dict(args.dual_model2_b))
        model_p_b.load_state_dict(_safe_load_state_dict(args.dual_model_p_b))
        model1_b.eval(); model2_b.eval(); model_p_b.eval()
        print(f"  Loaded {args.dual_model1_b}")
        print(f"  Loaded {args.dual_model2_b}")
        print(f"  Loaded {args.dual_model_p_b}")

        if model1 is None or model2 is None:
            needs_q_dual = {args.dual_strategy1, args.dual_strategy2} & {"ibr", "ibr-q"}
            if needs_q_dual:
                raise SystemExit(
                    f"Dual strategy {needs_q_dual} requires player-A Q models."
                )

        # Independent seeds for dual mode RNGs
        if args.seed is not None:
            dual_rng1 = np.random.default_rng(args.seed)
            dual_rng2 = np.random.default_rng(args.seed + 1)
            dual_rng1_swap = np.random.default_rng(args.seed + 2)
            dual_rng2_swap = np.random.default_rng(args.seed + 3)
        else:
            dual_rng1 = dual_rng2 = dual_rng1_swap = dual_rng2_swap = None

        def _run_one(label, p1_strategy, p2_strategy,
                     mp1, m1_1, m2_1, mp2, m1_2, m2_2, r1, r2):
            print(f"\n=== {label}: P1={p1_strategy} (A)  vs  P2={p2_strategy} (B) ===")
            res = run_matchup_dual(
                strategy1=p1_strategy, strategy2=p2_strategy,
                ppo=ppo,
                model_p_a=mp1, model_p_b=mp2,
                model1_a=m1_1, model2_a=m2_1,
                model1_b=m1_2, model2_b=m2_2,
                n_episodes=args.episodes,
                max_steps_per_episode=args.steps,
                warmup_steps=args.warmup,
                max_total_steps=args.max_total_steps,
                state_encoder_fn=state_encoder_fn,
                tau=args.tau,
                confidence_margin=args.confidence_margin,
                selection_mode=args.selection_mode,
                temperature=args.temperature,
                epsilon=args.epsilon,
                rng1=r1, rng2=r2,
            )
            wr = res.ego_wins / res.episodes if res.episodes else 0.0
            print(f"  ego_wins={res.ego_wins}/{res.episodes} = {wr:.1%}  "
                  f"truncated={res.truncated_episodes}")
            return wr

        wr_a_p1 = _run_one("Match 1 (A as P1)",
                           args.dual_strategy1, args.dual_strategy2,
                           model_p,  model1,  model2,
                           model_p_b, model1_b, model2_b,
                           dual_rng1, dual_rng2)

        if args.dual_swap:
            wr_b_p1 = _run_one("Match 2 (B as P1, swapped)",
                               args.dual_strategy1, args.dual_strategy2,
                               model_p_b, model1_b, model2_b,
                               model_p,  model1,  model2,
                               dual_rng1_swap, dual_rng2_swap)
            print(f"\n=== Dual summary (ego-side controlled) ===")
            print(f"  A as P1 win rate: {wr_a_p1:.1%}")
            print(f"  B as P1 win rate: {wr_b_p1:.1%}")
            a_total = wr_a_p1 + (1 - wr_b_p1)
            b_total = (1 - wr_a_p1) + wr_b_p1
            print(f"  A net (both directions): {a_total:.2f}  vs  B net: {b_total:.2f}")
            if a_total > b_total + 0.05:
                print(f"  → A stronger by {a_total - b_total:.2f}")
            elif b_total > a_total + 0.05:
                print(f"  → B stronger by {b_total - a_total:.2f}")
            else:
                print(f"  → roughly equal")

        return  # skip DEFAULT_MATCHUPS loop

    print(
        f"\nRunning {len(DEFAULT_MATCHUPS)} matchups "
        f"to {args.episodes} completed episodes each "
        f"(warmup={args.warmup}, max_steps_per_episode={args.steps}) ...\n"
    )

    results: List[MatchupResult] = []

    for matchup_idx, (s1, s2) in enumerate(DEFAULT_MATCHUPS):
        print(f"  [{s1} vs {s2}] ...")

        # Independent, per-matchup, per-player seeds
        if args.seed is not None:
            rng1 = np.random.default_rng(args.seed + 2 * matchup_idx)
            rng2 = np.random.default_rng(args.seed + 2 * matchup_idx + 1)
        else:
            rng1 = rng2 = None

        r = run_matchup(
            strategy1=s1,
            strategy2=s2,
            ppo=ppo,
            model_p=model_p,
            n_episodes=args.episodes,
            max_steps_per_episode=args.steps,
            warmup_steps=args.warmup,
            max_total_steps=args.max_total_steps,
            state_encoder_fn=state_encoder_fn,
            tau=args.tau,
            confidence_margin=args.confidence_margin,
            model1=model1,
            model2=model2,
            selection_mode=args.selection_mode,
            temperature=args.temperature,
            epsilon=args.epsilon,
            rng1=rng1,
            rng2=rng2,
        )
        results.append(r)

        wr = f"{r.win_rate:.0%}" if r.win_rate is not None else "---"
        arl = f"{r.avg_rally_length:.1f}" if r.avg_rally_length is not None else "---"
        avg_step = f"{r.avg_steps_per_episode:.1f}" if r.avg_steps_per_episode is not None else "---"

        print(
            f"    completed_eps={r.episodes}  "
            f"truncated={r.truncated_episodes}  "
            f"win_rate={wr}  "
            f"avg_steps={avg_step}  "
            f"avg_rally={arl}"
        )

    print_summary(results)

    analysis = analyse_results(results)
    print("\n=== ANALYSIS ===")
    print(f"center_safe long rallies: {analysis['center_safe_long_rallies']}")
    if analysis["left_short_win_rate"] is not None:
        print(f"left_short win rate:      {analysis['left_short_win_rate']:.0%}")
    print(f"recommendation:           {analysis['recommendation']}")

    print("\n=== EGO SKILL USAGE (learned strategies) ===")
    for r in results:
        total_picks = sum(r.skill_usage.values())
        if total_picks == 0:
            continue
        usage_str = "  ".join(
            f"{k}={v}({v / total_picks:.0%})"
            for k, v in r.skill_usage.items()
        )
        print(f"vs {r.strategy2:<14} {usage_str}")

    save_csv(results, args.output_csv)
    print(f"\nCSV saved to: {args.output_csv}")

    os.makedirs(
        os.path.dirname(args.output_json) if os.path.dirname(args.output_json) else ".",
        exist_ok=True,
    )

    json_data = [dataclasses.asdict(r) for r in results]
    for i, r in enumerate(results):
        json_data[i]["win_rate"] = r.win_rate
        json_data[i]["avg_rally_length"] = r.avg_rally_length
        json_data[i]["avg_steps_per_episode"] = r.avg_steps_per_episode
        json_data[i]["ego_success_rate"] = r.ego_success_rate
        json_data[i]["opp_success_rate"] = r.opp_success_rate
        json_data[i]["most_used_skill"] = most_used_skill(r)
        json_data[i]["dominant_fraction"] = dominant_skill_fraction(r)

    with open(args.output_json, "w") as f:
        json.dump({"results": json_data, "analysis": analysis}, f, indent=2)

    print(f"JSON saved to: {args.output_json}")


if __name__ == "__main__":
    main()
