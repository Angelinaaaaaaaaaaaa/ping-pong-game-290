"""
Revised headless evaluation for the original 2-skill (left / right) Nash pipeline.

Key fixes vs. the earlier version:
1. Evaluates until a target number of COMPLETED episodes, instead of running a
   fixed total step budget (old behavior: total_steps = episodes * steps).
2. Uses a one-time global warmup before counting results, while keeping the
   simulator running continuously afterward.
3. Uses try/finally when capturing stdout from env.step(), so stdout is always
   restored even if an exception occurs.
4. Makes winner inference a bit more robust by checking info first and then
   falling back to ball x-position if needed.
5. Supports a max_total_steps safety bound to prevent infinite loops.

Run from the project root:
    MUJOCO_GL=cgl venv/bin/python nash_skills/eval_matchup_2skill.py
    MUJOCO_GL=cgl venv/bin/python nash_skills/eval_matchup_2skill.py \
        --episodes 100 --steps 600 \
        --output-csv  skill_eval/baseline_2skill.csv \
        --output-json skill_eval/baseline_2skill.json
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import csv
import dataclasses
import io
import json
import random
from contextlib import redirect_stdout

from typing import Dict, List, Optional

import numpy as np
import torch

from nash_skills.v2.state_encoder import encode_ego, STATE_DIM as _STATE_DIM_76


# ─────────────────────────────────────────────────────────────────────────── #
# Result dataclass (self-contained copy for the 2-skill evaluator)            #
# ─────────────────────────────────────────────────────────────────────────── #

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
        # After the done-only fix, self.episodes already counts only done
        # (non-truncated) episodes. truncated_episodes is tracked separately
        # as a diagnostic counter and must NOT be subtracted here.
        return self.episodes

    @property
    def win_rate_clean(self) -> Optional[float]:
        if self.done_episodes == 0:
            return None
        return self.ego_wins / self.done_episodes


@dataclasses.dataclass
class AggregateMatchupResult:
    strategy1: str
    strategy2: str
    seeds: List[int]
    win_rate_mean: Optional[float]
    win_rate_std: Optional[float]
    trunc_mean: Optional[float]
    trunc_std: Optional[float]
    avg_rally_mean: Optional[float]
    avg_rally_std: Optional[float]
    left_usage_total: int
    right_usage_total: int
    left_fraction_mean: Optional[float]
    left_fraction_std: Optional[float]
    right_fraction_mean: Optional[float]
    right_fraction_std: Optional[float]
    dominant_skill: Optional[str]
    dominant_fraction_total: Optional[float]


# ─────────────────────────────────────────────────────────────────────────── #
# PPO observation builders (self-contained copy)                              #
# ─────────────────────────────────────────────────────────────────────────── #

_HISTORY = 4


def _build_obs1(obs, info):
    o = np.zeros(9 + 9 + 7 + 7 + 9 * _HISTORY, dtype=np.float32)
    o[:9]    = obs[:9]
    o[9:18]  = obs[18:27]
    o[18:21] = info["diff_pos"]
    o[21:25] = info["diff_quat"]
    o[25:32] = info["target"]
    o[32:]   = obs[42: 42 + _HISTORY * 9]
    return o


def _build_obs2(obs, info):
    o = np.zeros(9 + 9 + 7 + 7 + 9 * _HISTORY, dtype=np.float32)
    o[:9]    = obs[9:18]
    o[9:18]  = obs[27:36]
    o[18:21] = info["diff_pos_opp"]
    o[21:25] = info["diff_quat_opp"]
    o[25:32] = info["target_opp"]
    o[32:]   = obs[42 + _HISTORY * 9: 42 + 2 * _HISTORY * 9]
    return o


# ─────────────────────────────────────────────────────────────────────────── #
# CSV / summary (self-contained 2-skill versions)                             #
# ─────────────────────────────────────────────────────────────────────────── #

def save_csv(results: List[MatchupResult], path: str):
    rows = []
    for r in results:
        row = {
            "strategy1":            r.strategy1,
            "strategy2":            r.strategy2,
            "episodes":             r.episodes,
            "truncated_episodes":   r.truncated_episodes,
            "done_episodes":        r.done_episodes,
            "ego_wins":             r.ego_wins,
            "opp_wins":             r.opp_wins,
            "win_rate":             round(r.win_rate, 4)       if r.win_rate       is not None else "",
            "win_rate_clean":       round(r.win_rate_clean, 4) if r.win_rate_clean is not None else "",
            "total_steps":          r.total_steps,
            "avg_steps_per_episode": round(r.avg_steps_per_episode, 2) if r.avg_steps_per_episode is not None else "",
            "ego_contacts":         r.ego_contacts,
            "opp_contacts":         r.opp_contacts,
            "ego_successes":        r.ego_successes,
            "opp_successes":        r.opp_successes,
            "ego_success_rate":     round(r.ego_success_rate, 4) if r.ego_success_rate is not None else "",
            "opp_success_rate":     round(r.opp_success_rate, 4) if r.opp_success_rate is not None else "",
            "avg_rally_length":     round(r.avg_rally_length, 2) if r.avg_rally_length is not None else "",
            "most_used_skill":      most_used_skill(r) or "",
            "dominant_fraction":    round(dominant_skill_fraction(r), 4) if dominant_skill_fraction(r) is not None else "",
        }
        for skill_name in _SKILLS_2:
            row[f"usage_{skill_name}"] = r.skill_usage.get(skill_name, 0)
        rows.append(row)

    if not rows:
        return

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_aggregate_csv(results: List[AggregateMatchupResult], path: str):
    rows = []
    for r in results:
        row = {
            "strategy1": r.strategy1,
            "strategy2": r.strategy2,
            "n_seeds": len(r.seeds),
            "seeds": ",".join(str(s) for s in r.seeds),
            "win_rate_mean": round(r.win_rate_mean, 4) if r.win_rate_mean is not None else "",
            "win_rate_std": round(r.win_rate_std, 4) if r.win_rate_std is not None else "",
            "trunc_mean": round(r.trunc_mean, 4) if r.trunc_mean is not None else "",
            "trunc_std": round(r.trunc_std, 4) if r.trunc_std is not None else "",
            "avg_rally_mean": round(r.avg_rally_mean, 4) if r.avg_rally_mean is not None else "",
            "avg_rally_std": round(r.avg_rally_std, 4) if r.avg_rally_std is not None else "",
            "left_usage_total": r.left_usage_total,
            "right_usage_total": r.right_usage_total,
            "left_fraction_mean": round(r.left_fraction_mean, 4) if r.left_fraction_mean is not None else "",
            "left_fraction_std": round(r.left_fraction_std, 4) if r.left_fraction_std is not None else "",
            "right_fraction_mean": round(r.right_fraction_mean, 4) if r.right_fraction_mean is not None else "",
            "right_fraction_std": round(r.right_fraction_std, 4) if r.right_fraction_std is not None else "",
            "dominant_skill": r.dominant_skill or "",
            "dominant_fraction_total": round(r.dominant_fraction_total, 4) if r.dominant_fraction_total is not None else "",
        }
        rows.append(row)

    if not rows:
        return

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_per_seed_csv(results: List[dict], path: str):
    rows = []
    for item in results:
        r = item["result"]
        row = {
            "seed": item["seed"],
            "matchup_seed": item["matchup_seed"],
            "strategy1": r.strategy1,
            "strategy2": r.strategy2,
            "episodes": r.episodes,
            "truncated_episodes": r.truncated_episodes,
            "done_episodes": r.done_episodes,
            "ego_wins": r.ego_wins,
            "opp_wins": r.opp_wins,
            "win_rate": round(r.win_rate, 4) if r.win_rate is not None else "",
            "win_rate_clean": round(r.win_rate_clean, 4) if r.win_rate_clean is not None else "",
            "avg_rally_length": round(r.avg_rally_length, 4) if r.avg_rally_length is not None else "",
            "avg_steps_per_episode": round(r.avg_steps_per_episode, 4) if r.avg_steps_per_episode is not None else "",
            "most_used_skill": most_used_skill(r) or "",
            "dominant_fraction": round(dominant_skill_fraction(r), 4) if dominant_skill_fraction(r) is not None else "",
        }
        for skill_name in _SKILLS_2:
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
                  title: str = "2-SKILL BASELINE EVALUATION RESULTS"):
    if file is None:
        file = sys.stdout

    header = (
        f"{'Matchup':<30} "
        f"{'CompEp':>7} "
        f"{'Trunc':>7} "
        f"{'WinRate':>8} "
        f"{'WR(done)':>9} "
        f"{'AvgStep':>8} "
        f"{'AvgRally':>9}"
    )
    sep = "-" * len(header)

    print(f"\n{title:^{len(header)}}", file=file)
    print(sep, file=file)
    print(header, file=file)
    print(sep, file=file)

    for r in results:
        matchup  = f"{r.strategy1} vs {r.strategy2}"
        wr       = f"{r.win_rate:.0%}"       if r.win_rate       is not None else "---"
        wr_c     = f"{r.win_rate_clean:.0%}" if r.win_rate_clean is not None else "N/A"
        avg_step = f"{r.avg_steps_per_episode:.1f}" if r.avg_steps_per_episode is not None else "---"
        avg_rly  = f"{r.avg_rally_length:.1f}"      if r.avg_rally_length      is not None else "---"

        print(
            f"{matchup:<30} "
            f"{r.episodes:>7} "
            f"{r.truncated_episodes:>7} "
            f"{wr:>8} "
            f"{wr_c:>9} "
            f"{avg_step:>8} "
            f"{avg_rly:>9}",
            file=file,
        )

    print(sep, file=file)

    scored = [(r.win_rate, r) for r in results if r.win_rate is not None]
    if scored:
        best  = max(scored, key=lambda x: x[0])
        worst = min(scored, key=lambda x: x[0])
        print(f"\nBest matchup:  {best[1].strategy1} vs {best[1].strategy2} — {best[0]:.0%}", file=file)
        print(f"Worst matchup: {worst[1].strategy1} vs {worst[1].strategy2} — {worst[0]:.0%}", file=file)


def print_aggregate_summary(results: List[AggregateMatchupResult], file=None,
                            title: str = "2-SKILL MULTI-SEED SUMMARY"):
    if file is None:
        file = sys.stdout

    header = (
        f"{'Matchup':<30} "
        f"{'Seeds':>5} "
        f"{'WR mean±std':>14} "
        f"{'Trunc mean±std':>16} "
        f"{'Rally mean±std':>16}"
    )
    sep = "-" * len(header)

    print(f"\n{title:^{len(header)}}", file=file)
    print(sep, file=file)
    print(header, file=file)
    print(sep, file=file)

    for r in results:
        matchup = f"{r.strategy1} vs {r.strategy2}"
        wr = (
            f"{r.win_rate_mean:.0%}±{r.win_rate_std:.0%}"
            if r.win_rate_mean is not None and r.win_rate_std is not None
            else "---"
        )
        trunc = (
            f"{r.trunc_mean:.1f}±{r.trunc_std:.1f}"
            if r.trunc_mean is not None and r.trunc_std is not None
            else "---"
        )
        rally = (
            f"{r.avg_rally_mean:.1f}±{r.avg_rally_std:.1f}"
            if r.avg_rally_mean is not None and r.avg_rally_std is not None
            else "---"
        )
        print(
            f"{matchup:<30} "
            f"{len(r.seeds):>5} "
            f"{wr:>14} "
            f"{trunc:>16} "
            f"{rally:>16}",
            file=file,
        )

    print(sep, file=file)


# ─────────────────────────────────────────────────────────────────────────── #
# Skill usage helpers                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

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


def _safe_mean_std(values: List[Optional[float]]) -> tuple[Optional[float], Optional[float]]:
    usable = [float(v) for v in values if v is not None]
    if not usable:
        return None, None
    arr = np.array(usable, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def _usage_fraction(result: MatchupResult, skill_name: str) -> Optional[float]:
    total = sum(result.skill_usage.values())
    if total == 0:
        return None
    return result.skill_usage.get(skill_name, 0) / total


def aggregate_results_by_matchup(results: List[dict]) -> List[AggregateMatchupResult]:
    grouped: Dict[tuple[str, str], List[dict]] = {}
    for item in results:
        key = (item["result"].strategy1, item["result"].strategy2)
        grouped.setdefault(key, []).append(item)

    aggregate_rows: List[AggregateMatchupResult] = []
    for (strategy1, strategy2), entries in grouped.items():
        seed_values = [entry["seed"] for entry in entries]
        run_results = [entry["result"] for entry in entries]

        win_rate_mean, win_rate_std = _safe_mean_std([r.win_rate for r in run_results])
        trunc_mean, trunc_std = _safe_mean_std([r.truncated_episodes for r in run_results])
        avg_rally_mean, avg_rally_std = _safe_mean_std([r.avg_rally_length for r in run_results])
        left_fraction_mean, left_fraction_std = _safe_mean_std([
            _usage_fraction(r, "left") for r in run_results
        ])
        right_fraction_mean, right_fraction_std = _safe_mean_std([
            _usage_fraction(r, "right") for r in run_results
        ])

        left_usage_total = sum(r.skill_usage.get("left", 0) for r in run_results)
        right_usage_total = sum(r.skill_usage.get("right", 0) for r in run_results)
        total_usage = left_usage_total + right_usage_total
        if total_usage > 0:
            dominant_skill = "left" if left_usage_total >= right_usage_total else "right"
            dominant_fraction_total = max(left_usage_total, right_usage_total) / total_usage
        else:
            dominant_skill = None
            dominant_fraction_total = None

        aggregate_rows.append(
            AggregateMatchupResult(
                strategy1=strategy1,
                strategy2=strategy2,
                seeds=seed_values,
                win_rate_mean=win_rate_mean,
                win_rate_std=win_rate_std,
                trunc_mean=trunc_mean,
                trunc_std=trunc_std,
                avg_rally_mean=avg_rally_mean,
                avg_rally_std=avg_rally_std,
                left_usage_total=left_usage_total,
                right_usage_total=right_usage_total,
                left_fraction_mean=left_fraction_mean,
                left_fraction_std=left_fraction_std,
                right_fraction_mean=right_fraction_mean,
                right_fraction_std=right_fraction_std,
                dominant_skill=dominant_skill,
                dominant_fraction_total=dominant_fraction_total,
            )
        )

    return aggregate_rows


def print_skill_selection_analysis(results: List[AggregateMatchupResult], file=None):
    if file is None:
        file = sys.stdout

    print("\n=== FINAL SKILL SELECTION ANALYSIS ===", file=file)
    for r in results:
        total_usage = r.left_usage_total + r.right_usage_total
        if total_usage == 0:
            print(f"{r.strategy1} vs {r.strategy2}: no skill switches recorded", file=file)
            continue

        left_total_pct = r.left_usage_total / total_usage
        right_total_pct = r.right_usage_total / total_usage
        left_mean = f"{r.left_fraction_mean:.0%}" if r.left_fraction_mean is not None else "---"
        left_std = f"{r.left_fraction_std:.0%}" if r.left_fraction_std is not None else "---"
        right_mean = f"{r.right_fraction_mean:.0%}" if r.right_fraction_mean is not None else "---"
        right_std = f"{r.right_fraction_std:.0%}" if r.right_fraction_std is not None else "---"
        dominant = r.dominant_skill or "none"
        dominant_pct = (
            f"{r.dominant_fraction_total:.0%}"
            if r.dominant_fraction_total is not None else "---"
        )

        print(
            f"{r.strategy1} vs {r.strategy2}: "
            f"left total={r.left_usage_total}({left_total_pct:.0%})  "
            f"right total={r.right_usage_total}({right_total_pct:.0%})  "
            f"left mean±std={left_mean}±{left_std}  "
            f"right mean±std={right_mean}±{right_std}  "
            f"dominant={dominant}({dominant_pct})",
            file=file,
        )


# ─────────────────────────────────────────────────────────────────────────── #
# 2-skill constants                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

PPO_MODEL_PATH      = "logs/best_model_tracker1/best_model"
MODEL1_2SK_PATH     = "models/model1.pth"
MODEL2_2SK_PATH     = "models/model2.pth"
MODEL_P_2SK_PATH    = "models/model_p.pth"             # old buggy model (do not use)
MODEL1_V2_PATH      = "models/model1_2skill_v2.pth"
MODEL2_V2_PATH      = "models/model2_2skill_v2.pth"
MODEL_P_V2_PATH     = "models/model_p_2skill_v2.pth"   # 116-dim raw-obs model
MODEL1_76DIM_PATH   = "models/model1_76dim.pth"
MODEL2_76DIM_PATH   = "models/model2_76dim.pth"
MODEL_P_76DIM_PATH  = "models/model_p_76dim.pth"       # 76-dim encoded model (preferred)
# NOTE: models/model_p_v2.pth is written by nash_skills/v2/train_models.py with
# a 76-dim state encoder and CANNOT be loaded here; use MODEL_P_76DIM_PATH instead.

# §3.6 ablation: FactoredModel weights for the 2-skill v2 pipeline (76-dim).
# Trained by train_q_model_factored.py.
MODEL1_FACTORED_PATH  = "models/model1_factored.pth"
MODEL2_FACTORED_PATH  = "models/model2_factored.pth"
MODEL_P_FACTORED_PATH = "models/model_p_factored.pth"

HISTORY = 4
TABLE_SHIFT = 1.5
TABLE_X_MIN = 0.0
TABLE_X_MAX = TABLE_SHIFT + 1.37
TABLE_Y_ABS_MAX = 0.75

_SKILLS_2 = ["left", "right"]
N_SKILLS_2 = 2

VALID_STRATEGIES_2SKILL = [
    "nash-p-hard",
    "nash-p-minimax",
    "nash-p-adaptive",
    "nash-p-br",
    "ibr",
    "random",
    "left",
    "right",
]

DEFAULT_MATCHUPS_2SKILL = [
    ('nash-p-br','random'),
    ('nash-p-br','left'),
    ('nash-p-br','right'),
    ("nash-p-hard", "random"),
    ("nash-p-minimax", "random"),
    ("nash-p-adaptive", "random"),
    ("ibr", "random"),
    ("nash-p-hard", "left"),
    ("nash-p-minimax", "left"),
    ("nash-p-adaptive", "left"),
    ("ibr", "left"),
    ("nash-p-hard", "right"),
    ("nash-p-minimax", "right"),
    ("nash-p-adaptive", "right"),
    ("ibr", "right"),
    ("nash-p-hard", "ibr"),
    ("ibr", "nash-p-hard"),
    ("nash-p-minimax", "ibr"),
    ("ibr", "nash-p-minimax"),
    ("nash-p-adaptive", "ibr"),
    ("ibr", "nash-p-adaptive"),
    ("nash-p-hard", "nash-p-minimax"),
    ("nash-p-hard", "nash-p-adaptive")
]

# skill index -> side_target
_SIDE_TARGETS = {
    0: -1.0,  # left
    1:  1.0,  # right
}


# ─────────────────────────────────────────────────────────────────────────── #
# Environment wrapper                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

class _TwoSkillEnv:
    """Thin wrapper around the original competition env for 2-skill evaluation."""

    def __init__(self, proc_id=1):
        from mujoco_env_comp import KukaTennisEnv
        self._env = KukaTennisEnv(proc_id=proc_id)
        self._side1 = -1.0
        self._side2 =  1.0

    def set_skills(self, idx1: int, idx2: int):
        self._side1 = _SIDE_TARGETS[idx1]
        self._side2 = _SIDE_TARGETS[idx2]
        self._env.side_target = self._side1
        self._env.side_target_opp = self._side2

    def reset(self, seed=None):
        obs, info = self._env.reset()
        self._env.side_target = self._side1
        self._env.side_target_opp = self._side2
        return obs, info

    def step(self, action):
        self._env.side_target = self._side1
        self._env.side_target_opp = self._side2
        return self._env.step(action)

    def close(self):
        self._env.close()


# ─────────────────────────────────────────────────────────────────────────── #
# Strategy picker                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

def make_picker_2skill(
    strategy: str,
    model_p,
    model_state_dim: int = 116,
    model1=None,
    model2=None,
    tau: float = 0.2,
    confidence_margin: float = 0.05,
):
    """
    Return pick_fn(player, obs_vec, info, other_idx) -> skill_idx in {0, 1}.

    Parameters
    ----------
    strategy        : one of VALID_STRATEGIES_2SKILL
    model_p         : loaded potential model (SimpleModel)
    model_state_dim : input dimension of model_p.
                      116 → pass raw obs directly (legacy).
                      76  → encode via encode_ego() before passing to model_p.

    Strategies:
      'left'          -> always index 0
      'right'         -> always index 1
      'random'        -> uniform {0, 1}
      'ibr'           -> iterated best response using the learned Q-models
      'nash-p-2skill' -> argmax of the learned potential over the 2 skills
      'nash-p-adaptive-2skill' -> adaptive minimax-style selection using model_p
    """
    if strategy == "left":
        return lambda player, obs, info, other: 0

    if strategy == "right":
        return lambda player, obs, info, other: 1

    if strategy == "random":
        return lambda player, obs, info, other: np.random.randint(N_SKILLS_2)

    if strategy == "ibr":
        if model1 is None or model2 is None:
            raise ValueError("strategy='ibr' requires loaded model1 and model2")

        _sk_enc = {0: -1.0, 1: 1.0}
        _use_76 = (model_state_dim == _STATE_DIM_76)

        def pick_ibr(player, obs_vec, info, other_idx):
            if _use_76:
                base_np = encode_ego(obs_vec, info)
            else:
                base_np = obs_vec.copy()

            base = torch.tensor(base_np, dtype=torch.float32)
            other_idx = int(other_idx)
            if other_idx not in range(N_SKILLS_2):
                other_idx = 1
            if player == 1:
                curr_idx1 = 1
                curr_idx2 = other_idx
            else:
                curr_idx1 = other_idx
                curr_idx2 = 1


            with torch.no_grad():
                for _ in range(4):
                    rows_p1 = []
                    for ego_s in range(N_SKILLS_2):
                        row = base.clone()
                        row[-2] = _sk_enc[ego_s]
                        row[-1] = _sk_enc[curr_idx2]
                        rows_p1.append(row)
                    vals_p1 = model1(torch.stack(rows_p1))[:, 0]
                    curr_idx1 = int(torch.argmax(vals_p1).item())

                    rows_p2 = []
                    for opp_s in range(N_SKILLS_2):
                        row = base.clone()
                        row[-2] = _sk_enc[curr_idx1]
                        row[-1] = _sk_enc[opp_s]
                        rows_p2.append(row)
                    vals_p2 = model2(torch.stack(rows_p2))[:, 0]
                    curr_idx2 = int(torch.argmax(vals_p2).item())

            return curr_idx1 if player == 1 else curr_idx2

        return pick_ibr

    if strategy == "nash-p-hard":
        # Map skill index to the ±1 encoding the model was trained on
        # left=0 -> -1.0,  right=1 -> +1.0
        _sk_enc = {0: -1.0, 1: 1.0}
        _use_76 = (model_state_dim == _STATE_DIM_76)

        def pick_nash_hard(player, obs_vec, info, other_idx):
            # Build base feature vector: raw 116-dim or encoded 76-dim
            if _use_76:
                base_np = encode_ego(obs_vec, info)   # 76-dim
            else:
                base_np = obs_vec.copy()              # 116-dim

            base = torch.tensor(base_np, dtype=torch.float32)
            rows = []
            combos = []
            for ego_s in range(N_SKILLS_2):
                for opp_s in range(N_SKILLS_2):
                    row = base.clone()
                    row[-2] = _sk_enc[ego_s]
                    row[-1] = _sk_enc[opp_s]
                    rows.append(row)
                    combos.append((ego_s, opp_s))

            batch = torch.stack(rows)        # (4, state_dim)
            with torch.no_grad():
                vals = model_p(batch)[:, 0]  # (4,)
            best = int(torch.argmax(vals).item())
            ego_best, opp_best = combos[best]

            return ego_best if player == 1 else opp_best

        return pick_nash_hard
    
    if strategy == "nash-p-br":
        # Map skill index to the ±1 encoding the model was trained on
        # left=0 -> -1.0,  right=1 -> +1.0
        _sk_enc = {0: -1.0, 1: 1.0}
        _use_76 = (model_state_dim == _STATE_DIM_76)

        def pick_nash_br(player, obs_vec, info, other_idx):
            # Build base feature vector: raw 116-dim or encoded 76-dim
            if _use_76:
                base_np = encode_ego(obs_vec, info)   # 76-dim
            else:
                base_np = obs_vec.copy()              # 116-dim

            base = torch.tensor(base_np, dtype=torch.float32)
            rows = []
            combos = []
            for ego_s in range(N_SKILLS_2):
                for opp_s in range(N_SKILLS_2):
                    row = base.clone()
                    row[-2] = _sk_enc[ego_s]
                    row[-1] = _sk_enc[opp_s]
                    rows.append(row)
                    combos.append((ego_s, opp_s))

            batch = torch.stack(rows)        # (4, state_dim)
            with torch.no_grad():
                vals = model_p(batch)[:, 0]  # (4,)
            # BR-style selection: ego picks best response to opp's worst-case, and vice versa
            phi = vals.reshape(2, 2)  # (ego_skill, opp_skill)
            if player == 1:
                return int(torch.argmax(phi[:,other_idx]).item())
            else:
                return int(torch.argmax(phi[other_idx,:]).item())

        return pick_nash_br

    if strategy == "nash-p-minimax":
        # Map skill index to the ±1 encoding the model was trained on
        # left=0 -> -1.0,  right=1 -> +1.0
        _sk_enc = {0: -1.0, 1: 1.0}
        _use_76 = (model_state_dim == _STATE_DIM_76)

        def pick_nash_minimax(player, obs_vec, info, other_idx):
            # Build base feature vector: raw 116-dim or encoded 76-dim
            if _use_76:
                base_np = encode_ego(obs_vec, info)   # 76-dim
            else:
                base_np = obs_vec.copy()              # 116-dim

            base = torch.tensor(base_np, dtype=torch.float32)
            rows = []
            combos = []
            for ego_s in range(N_SKILLS_2):
                for opp_s in range(N_SKILLS_2):
                    row = base.clone()
                    row[-2] = _sk_enc[ego_s]
                    row[-1] = _sk_enc[opp_s]
                    rows.append(row)
                    combos.append((ego_s, opp_s))

            batch = torch.stack(rows)        # (4, state_dim)
            with torch.no_grad():
                vals = model_p(batch)[:, 0]  # (4,)
            
            # Symmetric robust selection on the learned potential surface:
            # each player picks the action whose worst-case potential is best
            # for that player. Player 2 should optimize its own column scores,
            # not minimize player 1's row scores.
            phi = vals.reshape(2, 2)  # (ego_skill, opp_skill)
            if player == 1:
                action_scores = phi.min(dim=1).values
                return int(torch.argmax(action_scores).item())
            else:
                action_scores = phi.min(dim=0).values
                return int(torch.argmax(action_scores).item())

        return pick_nash_minimax

    if strategy == "nash-p-adaptive":
        _sk_enc = {0: -1.0, 1: 1.0}
        _use_76 = (model_state_dim == _STATE_DIM_76)

        def pick_nash_adaptive(player, obs_vec, info, other_idx):
            if _use_76:
                base_np = encode_ego(obs_vec, info)
            else:
                base_np = obs_vec.copy()

            base = torch.tensor(base_np, dtype=torch.float32)
            rows = []
            for ego_s in range(N_SKILLS_2):
                for opp_s in range(N_SKILLS_2):
                    row = base.clone()
                    row[-2] = _sk_enc[ego_s]
                    row[-1] = _sk_enc[opp_s]
                    rows.append(row)

            batch = torch.stack(rows)
            with torch.no_grad():
                vals = model_p(batch)[:, 0]

            phi = vals.reshape(2, 2)
            if player == 1:
                action_scores = phi.min(dim=1).values
            else:
                # Same player-2 logic as minimax above: score each opponent
                # action by its worst-case potential over ego responses, then
                # choose the best such action for player 2.
                action_scores = phi.min(dim=0).values

            gap = torch.abs(action_scores[0] - action_scores[1]).item()
            if gap >= confidence_margin:
                return int(torch.argmax(action_scores).item())

            probs = torch.softmax(action_scores / tau, dim=0)
            return int(torch.multinomial(probs, 1).item())

        return pick_nash_adaptive

    raise ValueError(
        f"Unknown 2-skill strategy '{strategy}'. "
        f"Choose from: {VALID_STRATEGIES_2SKILL}"
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Helpers                                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

def _capture_env_step(env, action):
    """
    Run env.step(action) while capturing stdout lines emitted by the env.
    Always restores stdout even if env.step() raises.
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = env.step(action)
    return result, buf.getvalue().splitlines()


def _parse_contact_lines(lines, table_shift=TABLE_SHIFT):
    """
    Parse env print lines like:
      'Returned successfully by ego 1.876 0.198'
      'Returned successfully by opp 1.019 -0.249'

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

        # ego returns to opponent side: x in (1.5, 2.87)
        if "by ego" in line:
            ego_contacts += 1
            if table_shift < x_land < TABLE_X_MAX and abs(y_land) < TABLE_Y_ABS_MAX:
                ego_successes += 1

        # opp returns back to ego side: x in (0, 1.5)
        elif "by opp" in line:
            opp_contacts += 1
            if TABLE_X_MIN < x_land < table_shift and abs(y_land) < TABLE_Y_ABS_MAX:
                opp_successes += 1

    return ego_contacts, opp_contacts, ego_successes, opp_successes


def _infer_winner(obs, info):
    """
    Infer which player won this episode from the terminal observation.

    Preference:
    1. Use common winner keys if present in info.
    2. Fall back to ball VELOCITY x-component (obs[39]):
         ball_vel_x > 0  → ball moving toward opponent's side → opp missed → EGO wins
         ball_vel_x < 0  → ball moving toward ego's side      → ego missed → OPP wins
         ball_vel_x == 0 → inconclusive; default to 'opp' (conservative)

    WHY velocity and not position (old bug):
    The env fires done when the ball goes >0.3m past a racket, which can happen
    at any x-position.  Using ball_x > TABLE_SHIFT (the net) is wrong in cases
    where ego's racket is on the ego side and the ball has already passed it
    but hasn't crossed the net yet — the old heuristic called those OPP wins
    even though the ball was heading toward the opp (ego should have won).

    Returns:
        'ego' or 'opp'
    """
    if isinstance(info, dict):
        for key in ["winner", "point_winner", "episode_winner"]:
            if key in info:
                val = info[key]
                if val in ("ego", 0, "player0", "p0", "left"):
                    return "ego"
                if val in ("opp", 1, "player1", "p1", "right"):
                    return "opp"

    # Velocity-based fallback: obs[39] = ball velocity x-component
    ball_vel_x = float(obs[39])
    if ball_vel_x > 0:
        return "ego"   # ball heading to opp side → opp missed
    elif ball_vel_x < 0:
        return "opp"   # ball heading to ego side → ego missed
    else:
        return "opp"   # stationary / ambiguous — conservative default


# ─────────────────────────────────────────────────────────────────────────── #
# Matchup runner                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def run_matchup_2skill(
    strategy1: str,
    strategy2: str,
    ppo,
    model_p,
    model1,
    model2,
    n_episodes: int,
    max_steps_per_episode: int,
    warmup_steps: int = 300,
    max_total_steps: int | None = None,
    debug_contacts: bool = False,
    model_state_dim: int = 116,
    tau: float = 0.2,
    confidence_margin: float = 0.05,
) -> MatchupResult:
    """
    Run one 2-skill matchup headlessly, guaranteeing exactly n_episodes
    completed episodes regardless of episode length.

    Truncated episodes (hit per-episode step cap without env done=True) are
    counted toward n_episodes but recorded separately in truncated_episodes;
    they do NOT contribute a win or loss.

    The optional max_total_steps is kept as an absolute safety valve only —
    the primary termination condition is `completed_episodes == n_episodes`.

    model_state_dim : 116 for raw-obs models, 76 for encoded models trained
                      via collect_data_v2.py + train_q_model_v2.py (--76dim).
    """
    pick1 = make_picker_2skill(
        strategy1, model_p, model_state_dim, model1, model2, tau, confidence_margin
    )
    pick2 = make_picker_2skill(
        strategy2, model_p, model_state_dim, model1, model2, tau, confidence_margin
    )

    env = _TwoSkillEnv(proc_id=1)

    # Start from left vs right
    curr_idx1 = 0
    curr_idx2 = 1
    env.set_skills(curr_idx1, curr_idx2)
    obs, info = env.reset()

    prev_ball_x = float(obs[36])

    # Global one-time warmup (not counted toward episodes)
    total_steps = 0
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
            curr_idx1 = pick1(1, obs, info, curr_idx2)
            curr_idx2 = pick2(2, obs, info, curr_idx1)
            env.set_skills(curr_idx1, curr_idx2)

        prev_ball_x = curr_ball_x

        if done:
            env.set_skills(curr_idx1, curr_idx2)
            obs, info = env.reset()
            prev_ball_x = float(obs[36])

    # Counted evaluation — guaranteed exactly n_episodes iterations
    completed_episodes = 0
    truncated_episodes = 0
    ego_wins = opp_wins = 0
    ego_contacts = opp_contacts = 0
    ego_successes = opp_successes = 0
    rally_lengths: list = []
    episode_steps: list = []
    skill_usage: dict = {"left": 0, "right": 0}

    debug_step = 0  # for --debug-contacts

    curr_rally_len = 0
    steps_in_episode = 0

    while completed_episodes < n_episodes:
        # Respect optional absolute safety valve
        if max_total_steps is not None and total_steps >= max_total_steps:
            print(
                f"WARNING: hit max_total_steps={max_total_steps} with "
                f"{completed_episodes}/{n_episodes} done episodes collected. "
                "Stopping early — results are based on done episodes only."
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

        if debug_contacts and debug_step < 20:
            if lines:
                print(f"  [debug step {debug_step}] captured: {lines}")
            debug_step += 1

        e_c, o_c, e_s, o_s = _parse_contact_lines(lines)
        ego_contacts += e_c
        opp_contacts += o_c
        ego_successes += e_s
        opp_successes += o_s

        curr_ball_x = float(obs[36])

        # Rally crossing => choose next skills
        if (prev_ball_x - TABLE_SHIFT) * (curr_ball_x - TABLE_SHIFT) < 0:
            curr_rally_len += 1
            curr_idx1 = pick1(1, obs, info, curr_idx2)
            curr_idx2 = pick2(2, obs, info, curr_idx1)
            env.set_skills(curr_idx1, curr_idx2)
            skill_usage["left" if curr_idx1 == 0 else "right"] += 1

        prev_ball_x = curr_ball_x

        if done:
            winner = _infer_winner(obs, info)
            if winner == "ego":
                ego_wins += 1
            else:
                opp_wins += 1

            rally_lengths.append(curr_rally_len)
            episode_steps.append(steps_in_episode)
            # Only done episodes count toward n_episodes — this guarantees
            # exactly n_episodes statistically meaningful (won/lost) outcomes.
            # Counting truncated episodes here caused 56/60 to be truncated
            # → only 4 real outcomes → win_rate diluted to 0% despite ~52% true rate.
            completed_episodes += 1

            curr_rally_len = 0
            steps_in_episode = 0

            env.set_skills(curr_idx1, curr_idx2)
            obs, info = env.reset()
            prev_ball_x = float(obs[36])
            continue

        # Per-episode step cap → discard episode silently and reset.
        # Do NOT increment completed_episodes — truncated episodes have no winner
        # and would dilute the win_rate to near zero for stable-opponent matchups.
        if steps_in_episode >= max_steps_per_episode:
            truncated_episodes += 1
            # Still track rally length and steps for diagnostics
            rally_lengths.append(curr_rally_len)
            episode_steps.append(steps_in_episode)

            curr_rally_len = 0
            steps_in_episode = 0

            env.set_skills(curr_idx1, curr_idx2)
            obs, info = env.reset()
            prev_ball_x = float(obs[36])

    env.close()

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


# ─────────────────────────────────────────────────────────────────────────── #
# CSV / summary wrappers                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

def save_csv_2skill(results, path: str):
    save_csv(results, path)


def print_summary_2skill(results, file=None):
    print_summary(results, file=file, title="2-SKILL STRATEGY COMPARISON RESULTS")


def _raw_output_path(path: str) -> str:
    root, ext = os.path.splitext(path)
    if not ext:
        ext = ".csv"
    return f"{root}_per_seed{ext}"


# ─────────────────────────────────────────────────────────────────────────── #
# Main                                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(
        description="Revised 2-skill baseline matchup evaluator"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=60,
        help="Number of COMPLETED episodes to evaluate per matchup (default: 60)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=600,
        help="Maximum steps allowed per episode before truncation/reset (default: 600)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=300,
        help="One-time global warmup steps before counting evaluation results (default: 300)",
    )
    parser.add_argument(
        "--max-total-steps",
        type=int,
        default=None,
        help="Optional safety cap on total simulator steps per matchup",
    )
    parser.add_argument(
        "--output-csv",
        default="skill_eval/baseline_2skill.csv",
    )
    parser.add_argument(
        "--output-json",
        default="skill_eval/baseline_2skill.json",
    )
    parser.add_argument(
        "--debug-contacts",
        action="store_true",
        default=False,
        help="Print raw captured stdout lines for the first 20 evaluation steps",
    )
    parser.add_argument(
        "--model",
        choices=["76dim", "116dim", "old"],
        default="76dim",
        help=(
            "Which potential model to load:\n"
            "  76dim  — model_p_76dim.pth (76-dim encoded state, joint angles; default)\n"
            "  116dim — model_p_2skill_v2.pth (116-dim raw obs, v2 corrected)\n"
            "  old    — model_p.pth (original buggy model, not recommended)"
        ),
    )
    parser.add_argument(
        "--arch",
        choices=["simple", "factored"],
        default="simple",
        help=(
            "Estimator architecture (ablation):\n"
            "  simple   — SimpleModel (flat-concat MLP; default; uses --model paths)\n"
            "  factored — FactoredModel (separate state/skill encoders + fusion);\n"
            "             ignores --model and loads model{1,2,p}_factored.pth\n"
            "             (76-dim, trained by train_q_model_factored.py)."
        ),
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.2,
        help="Softmax temperature for nash-p-adaptive-2skill (default: 0.2)",
    )
    parser.add_argument(
        "--confidence-margin",
        type=float,
        default=0.05,
        help="Deterministic argmax threshold for nash-p-adaptive-2skill (default: 0.05)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Starting seed for multi-seed evaluation (default: 0)",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=3,
        help="Number of consecutive seeds to evaluate, starting at --seed (default: 3, so seeds 0..2)",
    )
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from model_arch import SimpleModel, FactoredModel

    seeds = list(range(args.seed, args.seed + args.num_seeds))
    random.seed(seeds[0])
    np.random.seed(seeds[0])
    torch.manual_seed(seeds[0])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seeds[0])

    print("Loading models...")
    print(f"Evaluation seeds: {seeds}")
    print(f"Adaptive Nash-p tau: {args.tau}")
    print(f"Adaptive Nash-p confidence_margin: {args.confidence_margin}")
    ppo = PPO.load(PPO_MODEL_PATH)

    if args.arch == "factored":
        # FactoredModel trained on 76-dim encoded state by
        # train_q_model_factored.py. --model is ignored under --arch factored
        # because the factored weights are only saved at one (76-dim) variant.
        model1_path  = MODEL1_FACTORED_PATH
        model2_path  = MODEL2_FACTORED_PATH
        model_p_path = MODEL_P_FACTORED_PATH
        state_dim    = 76
        if args.model != "76dim":
            print(f"  NOTE: --arch factored ignores --model={args.model}; "
                  f"using 76-dim factored weights.")
    elif args.model == "76dim":
        model1_path = MODEL1_76DIM_PATH
        model2_path = MODEL2_76DIM_PATH
        model_p_path = MODEL_P_76DIM_PATH
        state_dim = 76
    elif args.model == "116dim":
        model1_path = MODEL1_V2_PATH
        model2_path = MODEL2_V2_PATH
        model_p_path = MODEL_P_V2_PATH
        state_dim = 116
    else:
        model1_path = MODEL1_2SK_PATH
        model2_path = MODEL2_2SK_PATH
        model_p_path = MODEL_P_2SK_PATH
        state_dim = 116

    if args.arch == "factored":
        # state_dim above is the total input width (state + skill);
        # FactoredModel needs the split, so subtract the 2 skill dims.
        model1  = FactoredModel(state_dim=state_dim - 2, skill_dim=2)
        model2  = FactoredModel(state_dim=state_dim - 2, skill_dim=2)
        model_p = FactoredModel(state_dim=state_dim - 2, skill_dim=2,
                                last_layer_activation=None)
    else:
        model1  = SimpleModel(state_dim, [64, 32, 16], 1)
        model2  = SimpleModel(state_dim, [64, 32, 16], 1)
        model_p = SimpleModel(state_dim, [64, 32, 16], 1, last_layer_activation=None)
    model1.load_state_dict(torch.load(model1_path, weights_only=True))
    model2.load_state_dict(torch.load(model2_path, weights_only=True))
    model_p.load_state_dict(torch.load(model_p_path, weights_only=True))
    model1.eval()
    model2.eval()
    model_p.eval()
    print(f"  Loaded 2-skill arch:      {args.arch}")
    print(f"  Loaded 2-skill Q1:        {model1_path}  (state_dim={state_dim})")
    print(f"  Loaded 2-skill Q2:        {model2_path}  (state_dim={state_dim})")
    print(f"  Loaded 2-skill potential: {model_p_path}  (state_dim={state_dim})")

    print(
        f"\nRunning {len(DEFAULT_MATCHUPS_2SKILL)} matchups "
        f"to {args.episodes} completed episodes each "
        f"across {len(seeds)} seeds "
        f"(warmup={args.warmup}, max_steps_per_episode={args.steps}) ...\n"
    )

    seed_results: List[dict] = []
    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        current_results = []
        for idx, (s1, s2) in enumerate(DEFAULT_MATCHUPS_2SKILL):
            matchup_seed = seed * 1000 + idx
            random.seed(matchup_seed)
            np.random.seed(matchup_seed)
            torch.manual_seed(matchup_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(matchup_seed)

            print(f"  [{s1} vs {s2}] ...")
            r = run_matchup_2skill(
                strategy1=s1,
                strategy2=s2,
                ppo=ppo,
                model_p=model_p,
                model1=model1,
                model2=model2,
                n_episodes=args.episodes,
                max_steps_per_episode=args.steps,
                warmup_steps=args.warmup,
                max_total_steps=args.max_total_steps,
                debug_contacts=args.debug_contacts,
                model_state_dim=state_dim,
                tau=args.tau,
                confidence_margin=args.confidence_margin,
            )
            current_results.append(r)
            seed_results.append({"seed": seed, "matchup_seed": matchup_seed, "result": r})

            wr = f"{r.win_rate:.0%}" if r.win_rate is not None else "---"
            wr_c = f"{r.win_rate_clean:.0%}" if r.win_rate_clean is not None else "N/A"
            arl = f"{r.avg_rally_length:.1f}" if r.avg_rally_length is not None else "---"
            print(
                f"    eps={r.episodes}  done={r.done_episodes}  trunc={r.truncated_episodes}  "
                f"win_rate(all)={wr}  win_rate(done)={wr_c}  "
                f"ego_contacts={r.ego_contacts}  avg_rally={arl}"
            )

        print_summary_2skill(current_results)

    aggregate_results = aggregate_results_by_matchup(seed_results)
    print_aggregate_summary(aggregate_results)
    print_skill_selection_analysis(aggregate_results)

    raw_csv_path = _raw_output_path(args.output_csv)
    save_per_seed_csv(seed_results, raw_csv_path)
    save_aggregate_csv(aggregate_results, args.output_csv)
    print(f"\n  Aggregate CSV saved to: {args.output_csv}")
    print(f"  Per-seed CSV saved to:   {raw_csv_path}")

    os.makedirs(
        os.path.dirname(args.output_json)
        if os.path.dirname(args.output_json) else ".",
        exist_ok=True,
    )

    json_data = []
    for item in seed_results:
        r = item["result"]
        payload = dataclasses.asdict(r)
        payload["win_rate"] = r.win_rate
        payload["win_rate_clean"] = r.win_rate_clean
        payload["done_episodes"] = r.done_episodes
        payload["avg_rally_length"] = r.avg_rally_length
        payload["ego_success_rate"] = r.ego_success_rate
        payload["opp_success_rate"] = r.opp_success_rate
        payload["most_used_skill"] = most_used_skill(r)
        payload["dominant_fraction"] = dominant_skill_fraction(r)
        payload["seed"] = item["seed"]
        payload["matchup_seed"] = item["matchup_seed"]
        payload["tau"] = args.tau
        payload["confidence_margin"] = args.confidence_margin
        json_data.append(payload)

    with open(args.output_json, "w") as f:
        json.dump(
            {
                "config": {
                    "seeds": seeds,
                    "episodes": args.episodes,
                    "steps": args.steps,
                    "warmup": args.warmup,
                    "max_total_steps": args.max_total_steps,
                    "model": args.model,
                    "tau": args.tau,
                    "confidence_margin": args.confidence_margin,
                },
                "per_seed_results": json_data,
                "aggregate_results": [dataclasses.asdict(r) for r in aggregate_results],
            },
            f,
            indent=2,
        )

    print(f"  JSON saved to: {args.output_json}")


if __name__ == "__main__":
    main()
