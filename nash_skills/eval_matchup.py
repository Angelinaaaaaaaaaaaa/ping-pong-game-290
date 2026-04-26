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

PPO_MODEL_PATH = "logs/best_model_tracker1/best_model"
MODEL_P_5SK_PATH = "models/model_p_5skill.pth"
# v2 pipeline — new model trained with discounted returns and 76-dim state
MODEL_P_V2_PATH  = "models/model_p_v2.pth"

HISTORY = 4

TABLE_SHIFT = 1.5
TABLE_X_MIN = 0.0
TABLE_X_MAX = TABLE_SHIFT + 1.37
TABLE_Y_ABS_MAX = 0.75

VALID_STRATEGIES = ["nash-p", "random"] + SKILL_NAMES

DEFAULT_MATCHUPS = [
    ("nash-p", "random"),
    ("nash-p", "left"),
    ("nash-p", "right"),
    ("nash-p", "left_short"),
    ("nash-p", "right_short"),
    ("nash-p", "center_safe"),
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

def make_picker(strategy: str, model_p, state_encoder_fn=None):
    """
    Return pick_fn(player, obs_vec, other_skill_idx, info=None) -> skill_idx.

    Parameters
    ----------
    strategy         : str       — 'nash-p', 'random', or a fixed skill name
    model_p          : nn.Module — learned potential model
    state_encoder_fn : callable or None
        If provided, called as state_encoder_fn(obs, info, player) -> encoded_state
        before passing to model_p.  Used for the v2 pipeline where model_p
        expects a 76-dim encoded state rather than the raw 116-dim obs.
        If None (default), obs is passed directly (v1 / 5-skill behaviour).

    The returned function signature:
        pick_fn(player, obs_vec, other_skill_idx, info=None) -> int

    player:
      1 -> ego, write into obs[-2] (or encoded[-2] for v2)
      2 -> opp, write into obs[-1]

    For 5-skill nash-p, the last two obs/encoded entries are normalised skill indices.
    """
    if strategy in SKILL_NAMES:
        fixed_idx = skill_index(strategy)
        return lambda player, obs, other, info=None: fixed_idx

    if strategy == "random":
        return lambda player, obs, other, info=None: np.random.randint(N_SKILLS)

    if strategy == "nash-p":
        def pick_nash(player, obs_vec, other_skill_idx, info=None):
            # Collect ALL skill values before taking argmax — avoids the
            # best_idx=0 initialization bias that always returns LEFT on flat
            # potential surfaces.
            all_vals = []
            for s in range(N_SKILLS):
                # Apply optional v2 state encoder
                if state_encoder_fn is not None and info is not None:
                    base = state_encoder_fn(obs_vec, info, player)
                else:
                    base = obs_vec
                x = torch.tensor(base, dtype=torch.float32).unsqueeze(0)

                if player == 1:
                    x[0, -2] = s / (N_SKILLS - 1)
                    x[0, -1] = other_skill_idx / (N_SKILLS - 1)
                else:
                    x[0, -2] = other_skill_idx / (N_SKILLS - 1)
                    x[0, -1] = s / (N_SKILLS - 1)

                with torch.no_grad():
                    all_vals.append(model_p(x).item())

            return int(np.argmax(all_vals))

        return pick_nash

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
) -> MatchupResult:
    """
    Run one 5-skill matchup headlessly.

    Important:
    - One-time global warmup.
    - After warmup, keep running until we complete n_episodes.
    - If an episode hits max_steps_per_episode, truncate and reset it.
    """
    from nash_skills.env_wrapper import SkillEnv

    if max_total_steps is None:
        max_total_steps = warmup_steps + n_episodes * max_steps_per_episode * 5

    pick1 = make_picker(strategy1, model_p, state_encoder_fn=state_encoder_fn)
    pick2 = make_picker(strategy2, model_p, state_encoder_fn=state_encoder_fn)

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

            if strategy1 == "nash-p":
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
            "Use the v2 pipeline: load model_p_v2.pth (76-dim state encoder, "
            "discounted-return training). Default: v1 5-skill pipeline."
        ),
    )
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from model_arch import SimpleModel

    print("Loading models...")
    ppo = PPO.load(PPO_MODEL_PATH)

    if args.v2:
        # v2: model trained on 76-dim encoded states
        from nash_skills.v2.state_encoder import STATE_DIM as V2_STATE_DIM
        model_p_path = MODEL_P_V2_PATH
        model_p = SimpleModel(V2_STATE_DIM, [64, 32, 16], 1, last_layer_activation=None)
    else:
        # v1: original 116-dim raw obs
        model_p_path = MODEL_P_5SK_PATH
        model_p = SimpleModel(116, [64, 32, 16], 1, last_layer_activation=None)

    model_p.load_state_dict(_safe_load_state_dict(model_p_path))
    model_p.eval()

    # v2 state encoder: wraps encode_ego/encode_opp so make_picker can call it
    if args.v2:
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
    print(f"  Loaded potential:   {model_p_path}  ({'v2' if args.v2 else 'v1'})")
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

    print("\n=== EGO SKILL USAGE (nash-p) ===")
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