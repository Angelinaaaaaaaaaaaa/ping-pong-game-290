"""
Revised headless evaluation for the 5-skill Nash pipeline.

What this version fixes:
1. Evaluates until a target number of COMPLETED episodes.
2. Uses a one-time warmup before counting results.
3. Adds per-episode timeout and reset.
4. Uses safe stdout capture for env.step().
5. Keeps the 5-skill evaluator pure 5-skill only.
   Do NOT mix the 2-skill baseline into this file.

Run from the project root:
    MUJOCO_GL=cgl venv/bin/python nash_skills/eval_matchup.py
    MUJOCO_GL=cgl venv/bin/python nash_skills/eval_matchup.py \
        --episodes 60 --steps 600 \
        --output-csv  skill_eval/matchup_results_5skill.csv \
        --output-json skill_eval/matchup_results_5skill.json
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
# Trained by nash_skills/v2/train_q_model_5skill_factored.py.
MODEL1_5SK_FACTORED_PATH  = "models/model1_5skill_factored.pth"
MODEL2_5SK_FACTORED_PATH  = "models/model2_5skill_factored.pth"
MODEL_P_5SK_FACTORED_PATH = "models/model_p_5skill_factored.pth"
# FactoredModel weights for the 5-skill v3 pipeline (same-state per-sample
# potential training). Trained by train_q_model_5skill_v3_factored.py.
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
        """Episodes that ended with a real done signal (not truncated by step cap)."""
        return self.episodes - self.truncated_episodes

    @property
    def win_rate_clean(self) -> Optional[float]:
        """
        Win rate over done-only episodes.

        Use this instead of win_rate when many episodes are truncated by the
        per-episode step cap, because truncated episodes have no winner and
        including them in the denominator artificially deflates the win rate.

        Returns None when done_episodes == 0 (all episodes were truncated).
        """
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
    """
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)


def _capture_env_step(env, action):
    """
    Run env.step(action) and capture printed contact lines.
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = env.step(action)
    return result, buf.getvalue().splitlines()


def _parse_contact_lines(lines):
    """
    Parse env print lines like:
      Returned successfully by ego 1.876 0.198
      Returned successfully by opp 1.019 -0.249

    Returns:
        ego_contacts, opp_contacts, ego_successes, opp_successes
    """
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
    """
    Infer which player won from the terminal observation.

    Preference:
    1. Use common winner keys in info if present.
    2. Fall back to ball VELOCITY x-component (obs[39]):
         ball_vel_x > 0  → ball heading toward opp side → opp missed → EGO wins
         ball_vel_x < 0  → ball heading toward ego side → ego missed → OPP wins
         ball_vel_x == 0 → ambiguous; conservative default to 'opp'

    Using ball position (obs[36] > TABLE_SHIFT) was wrong: done fires when the
    ball passes >0.3m past a racket, which can happen at any x-position.
    """
    if isinstance(info, dict):
        for key in ["winner", "point_winner", "episode_winner"]:
            if key in info:
                val = info[key]
                if val in ("ego", 0, "player0", "p0", "left"):
                    return "ego"
                if val in ("opp", 1, "player1", "p1", "right"):
                    return "opp"

    ball_vel_x = float(obs[39])
    if ball_vel_x > 0:
        return "ego"
    elif ball_vel_x < 0:
        return "opp"
    else:
        return "opp"


def _initial_skill_idx(name: str, fallback: int = 0) -> int:
    """
    Get a skill index safely.
    """
    try:
        return skill_index(name)
    except Exception:
        return fallback


# --------------------------------------------------------------------------- #
# Strategy picker
# --------------------------------------------------------------------------- #

def _build_phi_table(obs_vec, info, player, model_p, state_encoder_fn):
    """
    Evaluate Φ(s, ego_s, opp_s) for all N×N skill pairs and return a
    (N_SKILLS, N_SKILLS) tensor where rows = ego skills, cols = opp skills.

    Batched: builds N² inputs in one forward pass. Encodes state once.
    """
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
    batch = torch.stack(rows)                    # (N*N, state_dim)
    with torch.no_grad():
        vals = model_p(batch)[:, 0]              # (N*N,)
    return vals.reshape(N_SKILLS, N_SKILLS)      # [ego_skill, opp_skill]


def make_picker(strategy: str, model_p, state_encoder_fn=None,
                tau: float = 0.2, confidence_margin: float = 0.05,
                model1=None, model2=None):
    """
    Return pick_fn(player, obs_vec, other_skill_idx, info=None) -> skill_idx.

    Five learned strategies
    -----------------------
    nash-p-hard      Joint argmax over the full Φ table (optimistic: assumes
                     both players coordinate on the global best joint action).
    nash-p-br        Conditional best response: argmax_ego Φ(s, ego, opp_current).
                     Reactive — reads the opponent's currently-observed skill.
                     (Alias: "nash-p" for backwards compatibility.)
    nash-p-minimax   Worst-case-safe: for each ego skill take min over opponent
                     responses, then pick the ego skill with the best worst-case.
    nash-p-adaptive  Same minimax scores, but uses softmax when the top-2 gap
                     is below `confidence_margin`; argmax otherwise.
    ibr              Q-based alternating best response (Φ-independent).
                     Requires model1 and model2 Q-value models.
                     Alternates argmax_ego Q1 / argmax_opp Q2 for ibr_steps
                     rounds and returns the converged ego skill.
    ibr-q            Q-based empirical-mix best response.
                     Tracks the opponent's observed skill frequencies and best
                     responds to that mixture using model1 / model2.

    Parameters
    ----------
    model_p           : nn.Module — learned potential Φ (required for all Φ strategies)
    model1            : nn.Module or None — ego Q-value model (required for ibr / ibr-q)
    model2            : nn.Module or None — opp Q-value model (required for ibr / ibr-q)
    state_encoder_fn  : callable or None — maps (obs, info, player) -> encoded vector
    tau               : float — softmax temperature for nash-p-adaptive
    confidence_margin : float — gap threshold for nash-p-adaptive argmax/softmax switch
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

    # ------------------------------------------------------------------ #
    # nash-p-hard: joint argmax over full Φ table                         #
    # ------------------------------------------------------------------ #
    if strategy == "nash-p-hard":
        def pick_hard(player, obs_vec, _other_skill_idx, info=None):
            phi = _build_phi_table(obs_vec, info, player, model_p, state_encoder_fn)
            best = int(phi.argmax().item())        # flat index into N×N
            ego_best = best // N_SKILLS
            return ego_best
        return pick_hard

    # ------------------------------------------------------------------ #
    # nash-p-br: best response fixing opp's current skill                 #
    # (original "nash-p" — kept as alias too)                             #
    # ------------------------------------------------------------------ #
    if strategy in ("nash-p-br", "nash-p"):
        def pick_br(player, obs_vec, other_skill_idx, info=None):
            phi = _build_phi_table(obs_vec, info, player, model_p, state_encoder_fn)
            if player == 1:
                return int(phi[:, other_skill_idx].argmax().item())
            else:
                return int(phi[other_skill_idx, :].argmin().item())
        return pick_br

    # ------------------------------------------------------------------ #
    # nash-p-minimax: worst-case-safe argmax                               #
    # ------------------------------------------------------------------ #
    if strategy == "nash-p-minimax":
        def pick_minimax(player, obs_vec, other_skill_idx, info=None):
            phi = _build_phi_table(obs_vec, info, player, model_p, state_encoder_fn)
            if player == 1:
                action_scores = phi.min(dim=1).values   # worst opp response per ego skill
            else:
                action_scores = -phi.max(dim=0).values  # worst ego response per opp skill
            return int(torch.argmax(action_scores).item())
        return pick_minimax

    # ------------------------------------------------------------------ #
    # nash-p-adaptive: minimax + softmax fallback when surface is flat    #
    # ------------------------------------------------------------------ #
    if strategy == "nash-p-adaptive":
        def pick_adaptive(player, obs_vec, other_skill_idx, info=None):
            phi = _build_phi_table(obs_vec, info, player, model_p, state_encoder_fn)
            if player == 1:
                action_scores = phi.min(dim=1).values
            else:
                action_scores = -phi.max(dim=0).values
            sorted_scores, _ = torch.sort(action_scores, descending=True)
            gap = (sorted_scores[0] - sorted_scores[1]).item()
            if gap >= confidence_margin:
                return int(torch.argmax(action_scores).item())
            probs = torch.softmax(action_scores / tau, dim=0)
            return int(torch.multinomial(probs, 1).item())
        return pick_adaptive

    # ------------------------------------------------------------------ #
    # ibr: Q-based alternating best response (Φ-independent)              #
    # ------------------------------------------------------------------ #
    if strategy == "ibr":
        if model1 is None or model2 is None:
            raise ValueError(
                "ibr requires model1 and model2 Q-value models. "
                "Pass them to make_picker(model1=..., model2=...)."
            )

        IBR_STEPS = 10   # alternating rounds before returning

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
                # Ego best-responds to s2 using Q1
                q1_vals = []
                for ego_s in range(N_SKILLS):
                    x = base_enc.clone().unsqueeze(0)
                    x[0, -2] = ego_s / (N_SKILLS - 1)
                    x[0, -1] = s2    / (N_SKILLS - 1)
                    with torch.no_grad():
                        q1_vals.append(model1(x).item())
                s1 = int(np.argmax(q1_vals))

                # Opp best-responds to s1 using Q2 (opp minimises Q2 = maximises -Q2)
                q2_vals = []
                for opp_s in range(N_SKILLS):
                    x = base_enc.clone().unsqueeze(0)
                    x[0, -2] = s1    / (N_SKILLS - 1)
                    x[0, -1] = opp_s / (N_SKILLS - 1)
                    with torch.no_grad():
                        q2_vals.append(model2(x).item())
                # opp minimises Q2 (it's the ego-perspective value, opp wants it low)
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
# Single matchup runner
# --------------------------------------------------------------------------- #

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
) -> MatchupResult:
    """
    Run one 5-skill matchup headlessly.

    Important:
    - One-time global warmup.
    - After warmup, keep running until we complete n_episodes.
    - If an episode hits max_steps_per_episode, truncate and reset it.
    - model1 / model2: Q-value models required when strategy1 or strategy2 is
      'ibr' or 'ibr-q'.
    """
    from nash_skills.env_wrapper import SkillEnv

    if max_total_steps is None:
        max_total_steps = warmup_steps + n_episodes * max_steps_per_episode * 5

    pick1 = make_picker(strategy1, model_p, state_encoder_fn=state_encoder_fn,
                        tau=tau, confidence_margin=confidence_margin,
                        model1=model1, model2=model2)
    pick2 = make_picker(strategy2, model_p, state_encoder_fn=state_encoder_fn,
                        tau=tau, confidence_margin=confidence_margin,
                        model1=model1, model2=model2)

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

    while completed_episodes < n_episodes and total_steps < max_total_steps:
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

    if completed_episodes < n_episodes:
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
    parser.add_argument(
        "--episodes",
        type=int,
        default=60,
        help="Number of COMPLETED episodes per matchup (default: 60)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=600,
        help="Maximum steps per episode before truncation/reset (default: 600)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=300,
        help="One-time warmup steps before counting results (default: 300)",
    )
    parser.add_argument(
        "--max-total-steps",
        type=int,
        default=None,
        help="Optional safety cap on total simulator steps per matchup",
    )
    parser.add_argument(
        "--output-csv",
        default="skill_eval/matchup_results_5skill.csv",
    )
    parser.add_argument(
        "--output-json",
        default="skill_eval/matchup_results_5skill.json",
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        default=False,
        help=(
            "Use the 4-skill v2 pipeline: load model_p_v2.pth (76-dim state encoder, "
            "discounted-return training). Default: v1 5-skill pipeline."
        ),
    )
    parser.add_argument(
        "--v2-5skill",
        action="store_true",
        default=False,
        dest="v2_5skill",
        help=(
            "Use the 5-skill v2 pipeline: load model_p_5skill_v2.pth (76-dim state "
            "encoder, discounted-return labels, all 5 skills). Trained by "
            "train_q_model_5skill_v2.py."
        ),
    )
    parser.add_argument(
        "--v3-5skill",
        action="store_true",
        default=False,
        dest="v3_5skill",
        help=(
            "Use the 5-skill v3 pipeline: load model_p_5skill_v3.pth (76-dim state "
            "encoder, discounted-return labels, all 5 skills, same-state per-sample "
            "potential training). Trained by train_q_model_5skill_v3.py."
        ),
    )
    parser.add_argument(
        "--arch",
        choices=["simple", "factored"],
        default="simple",
        help=(
            "Estimator architecture (§3.6 ablation):\n"
            "  simple   — SimpleModel (flat-concat MLP; default)\n"
            "  factored — FactoredModel (separate state/skill encoders + fusion).\n"
            "             Requires --v2-5skill or --v3-5skill. Loads\n"
            "             model{1,2,p}_5skill_factored.pth   (with --v2-5skill) or\n"
            "             model{1,2,p}_5skill_v3_factored.pth (with --v3-5skill)."
        ),
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.2,
        help="Softmax temperature for nash-p-adaptive (default: 0.2)",
    )
    parser.add_argument(
        "--confidence-margin",
        type=float,
        default=0.05,
        dest="confidence_margin",
        help="Gap threshold below which nash-p-adaptive uses softmax instead of argmax (default: 0.05)",
    )
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from model_arch import SimpleModel, FactoredModel

    print("Loading models...")
    ppo = PPO.load(PPO_MODEL_PATH)

    if args.arch == "factored":
        # FactoredModel weights trained on 116-dim raw obs.
        # Pick v2 (minibatch-mean) or v3 (same-state per-sample) trained weights
        # based on the pipeline flag. One of --v2-5skill / --v3-5skill is required.
        if args.v3_5skill:
            model_p_path = MODEL_P_5SK_V3_FACTORED_PATH
            pipeline_tag = "v3-5skill-factored"
        elif args.v2_5skill:
            model_p_path = MODEL_P_5SK_FACTORED_PATH
            pipeline_tag = "v2-5skill-factored"
        else:
            raise SystemExit(
                "--arch factored requires --v2-5skill or --v3-5skill (the "
                "factored ablation is only trained for the 5-skill pipelines)."
            )
        # FactoredModel splits state vs skill internally. After re-collecting
        # 5-skill data via nash_skills/v2/collect_data.py, rallies store 76-dim
        # encoded states (74 state dims + 2 skill dims).
        model_p = FactoredModel(state_dim=74, skill_dim=2, last_layer_activation=None)
    elif args.v3_5skill:
        from nash_skills.v2.state_encoder import STATE_DIM as V2_STATE_DIM
        model_p_path = MODEL_P_5SK_V3_PATH
        model_p = SimpleModel(V2_STATE_DIM, [64, 32, 16], 1, last_layer_activation=None)
        pipeline_tag = "v3-5skill"
    elif args.v2_5skill:
        # 5-skill v2: 76-dim encoded states, all 5 skills, discounted-return training
        from nash_skills.v2.state_encoder import STATE_DIM as V2_STATE_DIM
        model_p_path = MODEL_P_5SK_V2_PATH
        model_p = SimpleModel(V2_STATE_DIM, [64, 32, 16], 1, last_layer_activation=None)
        pipeline_tag = "v2-5skill"
    elif args.v2:
        # 4-skill v2: 76-dim encoded states (original v2 diagnostic)
        from nash_skills.v2.state_encoder import STATE_DIM as V2_STATE_DIM
        model_p_path = MODEL_P_V2_PATH
        model_p = SimpleModel(V2_STATE_DIM, [64, 32, 16], 1, last_layer_activation=None)
        pipeline_tag = "v2-4skill"
    else:
        # v1: original 116-dim raw obs
        model_p_path = MODEL_P_5SK_PATH
        model_p = SimpleModel(116, [64, 32, 16], 1, last_layer_activation=None)
        pipeline_tag = "v1-5skill"

    model_p.load_state_dict(_safe_load_state_dict(model_p_path))
    model_p.eval()

    # Q-value models — needed for ibr / ibr-q
    needs_q = any(s in {"ibr", "ibr-q"} for s, _ in DEFAULT_MATCHUPS) or any(
        s in {"ibr", "ibr-q"} for _, s in DEFAULT_MATCHUPS
    )
    model1 = model2 = None
    if needs_q:
        # Architecture branch first: under --arch factored we construct
        # FactoredModel and skip the SimpleModel construction below.
        if args.arch == "factored":
            if args.v3_5skill:
                _q1_path = MODEL1_5SK_V3_FACTORED_PATH
                _q2_path = MODEL2_5SK_V3_FACTORED_PATH
            else:  # args.v2_5skill (the model_p branch above already enforced this)
                _q1_path = MODEL1_5SK_FACTORED_PATH
                _q2_path = MODEL2_5SK_FACTORED_PATH
            # 76-dim encoded data: 74 state dims + 2 skill dims (see model_p above).
            model1 = FactoredModel(state_dim=74, skill_dim=2)
            model2 = FactoredModel(state_dim=74, skill_dim=2)
        else:
            if args.v3_5skill:
                from nash_skills.v2.state_encoder import STATE_DIM as V2_STATE_DIM
                _sdim = V2_STATE_DIM
                _q1_path = MODEL1_5SK_V3_PATH
                _q2_path = MODEL2_5SK_V3_PATH
            elif args.v2_5skill:
                from nash_skills.v2.state_encoder import STATE_DIM as V2_STATE_DIM
                _sdim = V2_STATE_DIM
                _q1_path = MODEL1_5SK_V2_PATH
                _q2_path = MODEL2_5SK_V2_PATH
            elif args.v2:
                from nash_skills.v2.state_encoder import STATE_DIM as V2_STATE_DIM
                _sdim = V2_STATE_DIM
                _q1_path = MODEL1_V2_PATH
                _q2_path = MODEL2_V2_PATH
            else:
                _sdim = 116
                _q1_path = MODEL1_5SK_PATH
                _q2_path = MODEL2_5SK_PATH
            model1 = SimpleModel(_sdim, [64, 32, 16], 1)
            model2 = SimpleModel(_sdim, [64, 32, 16], 1)
        model1.load_state_dict(_safe_load_state_dict(_q1_path))
        model2.load_state_dict(_safe_load_state_dict(_q2_path))
        model1.eval()
        model2.eval()
        print(f"  Loaded Q-models:    {_q1_path}, {_q2_path}")

    # v2 state encoder: wraps encode_ego/encode_opp so make_picker can call it.
    # After re-collecting 5-skill data via nash_skills/v2/collect_data.py, ALL
    # v2/v3 models (simple AND factored) are trained on 76-dim encoded state,
    # so all of them need the encoder at eval time.
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
    print(
        f"\nRunning {len(DEFAULT_MATCHUPS)} matchups "
        f"to {args.episodes} completed episodes each "
        f"(warmup={args.warmup}, max_steps_per_episode={args.steps}) ...\n"
    )

    results: List[MatchupResult] = []

    for s1, s2 in DEFAULT_MATCHUPS:
        print(f"  [{s1} vs {s2}] ...")

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
