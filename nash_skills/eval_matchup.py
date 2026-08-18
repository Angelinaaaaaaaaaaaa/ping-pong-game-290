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
from pathlib import Path
from contextlib import redirect_stdout
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from nash_skills.skills import SKILL_NAMES, SKILL_PROFILE_NAMES, N_SKILLS, skill_index, skill_from_index
from nash_skills.winner_inference import infer_terminal_winner
from diagnostic_rendering import (
    EpisodeVideoRecorder,
    encode_np_random_state,
    decode_np_random_state,
    manual_render_requested,
    prompt_manual_replays,
    render_episode_limit,
    replay_selected_episodes,
)


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
# MODEL1_5SK_V3_PATH    = "models/model1_5skill_v3_updated.pth"
# MODEL2_5SK_V3_PATH    = "models/model2_5skill_v3_updated.pth"
# MODEL_P_5SK_V3_PATH   = "models/model_p_5skill_v3_updated.pth"
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
    "nash-p-hard",      # optimistic heuristic over Φ rows/columns
    "nash-p-br",        # conditional best response fixing other player's skill
    "nash-p-minimax",   # heuristic: maximin over Φ rows/columns
    "nash-p-adaptive",  # heuristic: maximin + softmax when gap < margin
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
        """
        Episodes that ended with a real done signal.

        After the done-only evaluation fix, ``episodes`` already counts only
        non-truncated terminal episodes. ``truncated_episodes`` is tracked
        separately as a diagnostic counter and must not be subtracted here.
        """
        return self.episodes

    @property
    def win_rate_clean(self) -> Optional[float]:
        """
        Win rate over done-only episodes.

        This is currently identical to ``win_rate`` because truncated episodes
        are not included in ``episodes``. The property is kept for compatibility
        with the 2-skill evaluator and older analysis scripts.
        """
        if self.done_episodes == 0:
            return None
        return self.ego_wins / self.done_episodes


def select_player1_loss_replays(
    rows: List[Dict[str, Any]],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    selected = [row for row in rows if row.get("winner") == "opp"]
    if limit is not None:
        return selected[:limit]
    return selected


def _render_options_requested(args: argparse.Namespace) -> bool:
    return bool(
        args.render_losses_only
        or args.render_truncated_only
        or args.render_episodes is not None
        or args.save_video
        or args.experimental_post_eval_replay
    )


def _original_capture_requested(args: argparse.Namespace) -> bool:
    return bool(args.matchup is not None and args.render_losses_only)


def _json_dumps(value: Any) -> str:
    return json.dumps(value)


def _json_loads(value: Any, default: Any = None) -> Any:
    if value in ("", None):
        return default
    return json.loads(str(value))


def _float_list(values: Any) -> List[float]:
    if values is None:
        return []
    arr = np.asarray(values, dtype=float).reshape(-1)
    return [float(x) for x in arr]


def _softmax_probs(values: np.ndarray, temperature: float) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if temperature <= 0:
        probs = np.zeros(len(v), dtype=float)
        probs[int(np.argmax(v))] = 1.0
        return probs
    z = (v - np.max(v)) / temperature
    exp = np.exp(z)
    return exp / exp.sum()


def _selection_probabilities(
    values: np.ndarray,
    selection_mode: str,
    tau: float,
    confidence_margin: float,
    temperature: float,
    epsilon: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    n = len(values)
    uniform = np.ones(n, dtype=float) / n
    argmax_idx = int(np.argmax(values))
    argmax_probs = np.zeros(n, dtype=float)
    argmax_probs[argmax_idx] = 1.0

    if selection_mode == "argmax":
        if n <= 1:
            return argmax_probs
        sorted_values = np.sort(values)[::-1]
        gap = float(sorted_values[0] - sorted_values[1])
        if gap >= confidence_margin or tau <= 0:
            return argmax_probs
        return _softmax_probs(values, tau)
    if selection_mode == "softmax":
        return _softmax_probs(values, temperature)
    if selection_mode == "epsilon_argmax":
        return (1.0 - epsilon) * argmax_probs + epsilon * uniform
    if selection_mode == "epsilon_softmax":
        return (1.0 - epsilon) * _softmax_probs(values, temperature) + epsilon * uniform
    return argmax_probs


def _encode_generator_state(rng) -> str:
    if rng is None:
        return ""
    return json.dumps(rng.bit_generator.state)


def _decode_generator_state(value: Any) -> Optional[np.random.Generator]:
    state = _json_loads(value)
    if not state:
        return None
    bitgen_name = state.get("bit_generator", "PCG64")
    bitgen_cls = getattr(np.random, bitgen_name, np.random.PCG64)
    rng = np.random.Generator(bitgen_cls())
    rng.bit_generator.state = state
    return rng


def _skill_sequence_from_decisions(decisions: List[Dict[str, Any]], player_key: str) -> List[str]:
    return [str(decision[player_key]) for decision in decisions]


def _warn_if_replay_differs(original: Dict[str, Any], replayed: Dict[str, Any]) -> None:
    checks = [
        ("winner", original.get("winner"), replayed.get("winner")),
        ("truncated", bool(original.get("truncated")), bool(replayed.get("truncated"))),
        ("physics_steps", int(original.get("physics_steps", 0)), int(replayed.get("physics_steps", 0))),
        ("rally_length", int(original.get("rally_length", 0)), int(replayed.get("rally_length", 0))),
        ("p1_skill_sequence", original.get("p1_skill_sequence"), replayed.get("p1_skill_sequence")),
        ("p2_skill_sequence", original.get("p2_skill_sequence"), replayed.get("p2_skill_sequence")),
    ]
    mismatches = [
        f"{name}: original={expected!r} replay={actual!r}"
        for name, expected, actual in checks
        if expected != actual
    ]
    if mismatches:
        episode_id = original.get("episode_id", "")
        print(
            f"WARNING: replay mismatch for episode {episode_id}: "
            + "; ".join(mismatches),
            flush=True,
        )


def _final_ball_state(obs, info: Optional[dict] = None) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "position": np.asarray(obs[36:39], dtype=float).tolist(),
        "velocity": np.asarray(obs[39:42], dtype=float).tolist(),
    }
    if isinstance(info, dict):
        for key in (
            "termination_reason",
            "ball_x",
            "ball_y",
            "ball_z",
            "ball_vx",
            "ball_vy",
            "ball_vz",
            "ego_racket_x",
            "opp_racket_x",
        ):
            if key in info:
                state[key] = info[key]
    return state


def _write_video_metadata(path: Path, row: Dict[str, Any]) -> None:
    metadata = {
        "episode_id": row.get("episode_id"),
        "strategy1": row.get("strategy1"),
        "strategy2": row.get("strategy2"),
        "winner": row.get("winner"),
        "truncated": row.get("truncated"),
        "termination_reason": row.get("termination_reason"),
        "physics_steps": row.get("physics_steps"),
        "rally_length": row.get("rally_length"),
        "p1_initial_skill": row.get("p1_initial_skill"),
        "p2_initial_skill": row.get("p2_initial_skill"),
        "p1_skill_sequence": row.get("p1_skill_sequence"),
        "p2_skill_sequence": row.get("p2_skill_sequence"),
        "decision_steps": _json_loads(row.get("decision_steps"), []),
        "final_ball_state": row.get("final_ball_state"),
        "reset_mode": row.get("reset_mode"),
        "skill_profile": row.get("skill_profile"),
        "gantry_speed_scale": row.get("gantry_speed_scale"),
        "policy_selection_settings": _json_loads(row.get("policy_selection_settings"), {}),
    }
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(metadata, f, indent=2)


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
    1. Use explicit winner keys in info if present.
    2. Reconstruct the env's terminal racket-boundary condition from info.
    3. Fall back to ball VELOCITY x-component (obs[39]):
         ball_vel_x > 0  → ball heading toward opp side → opp missed → EGO wins
         ball_vel_x < 0  → ball heading toward ego side → ego missed → OPP wins
    4. Last resort: ball position relative to the net only when the previous
       signals are unavailable or exactly ambiguous.
    """
    return infer_terminal_winner(obs, info, fallback="position") or "opp"


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
    Evaluate Φ(s, p1_skill, p2_skill) for all N×N skill pairs.

    Table convention:
        rows    = Player 1 skills
        columns = Player 2 skills

    Model input convention:
        input[-2] = Player 1 skill, normalized to [0, 1]
        input[-1] = Player 2 skill, normalized to [0, 1]

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
    return vals.reshape(N_SKILLS, N_SKILLS)      # [p1_skill, p2_skill]


def _encoded_policy_state(obs_vec, info, player, state_encoder_fn):
    if state_encoder_fn is not None and info is not None:
        return np.asarray(state_encoder_fn(obs_vec, info, player), dtype=np.float32)
    return np.asarray(obs_vec, dtype=np.float32)


def _q_values_for_player(base_enc, player: int, other_skill_idx: int, model1, model2) -> List[float]:
    if model1 is None or model2 is None:
        return []
    model = model1 if player == 1 else model2
    values = []
    for candidate in range(N_SKILLS):
        x = torch.tensor(base_enc, dtype=torch.float32).clone().unsqueeze(0)
        if player == 1:
            x[0, -2] = candidate / (N_SKILLS - 1)
            x[0, -1] = other_skill_idx / (N_SKILLS - 1)
        else:
            x[0, -2] = other_skill_idx / (N_SKILLS - 1)
            x[0, -1] = candidate / (N_SKILLS - 1)
        with torch.no_grad():
            values.append(float(model(x).item()))
    return values


def _phi_values_for_player(base_enc, player: int, other_skill_idx: int, model_p) -> List[float]:
    if model_p is None:
        return []
    rows = []
    for candidate in range(N_SKILLS):
        x = torch.tensor(base_enc, dtype=torch.float32).clone()
        if player == 1:
            x[-2] = candidate / (N_SKILLS - 1)
            x[-1] = other_skill_idx / (N_SKILLS - 1)
        else:
            x[-2] = other_skill_idx / (N_SKILLS - 1)
            x[-1] = candidate / (N_SKILLS - 1)
        rows.append(x)
    with torch.no_grad():
        return _float_list(model_p(torch.stack(rows))[:, 0])


def _policy_action_values(
    strategy: str,
    base_enc,
    player: int,
    other_skill_idx: int,
    model_p,
    model1,
    model2,
    observed_opp_mix: Optional[List[float]] = None,
) -> tuple[str, List[float]]:
    if strategy in SKILL_NAMES:
        values = [0.0] * N_SKILLS
        values[skill_index(strategy)] = 1.0
        return "fixed", values
    if strategy == "random":
        return "random", [1.0 / N_SKILLS] * N_SKILLS

    if strategy == "ibr":
        if model1 is None or model2 is None:
            return "q", []
        s1 = other_skill_idx
        s2 = other_skill_idx
        q1_vals: List[float] = []
        q2_vals: List[float] = []
        for _ in range(4):
            q1_vals = _q_values_for_player(base_enc, 1, s2, model1, model2)
            s1 = int(np.argmax(q1_vals))
            q2_vals = _q_values_for_player(base_enc, 2, s1, model1, model2)
            s2 = int(np.argmax(q2_vals))
        return "q", q1_vals if player == 1 else q2_vals

    if strategy == "ibr-q":
        if model1 is None or model2 is None:
            return "q", []
        mix = observed_opp_mix or [1.0 / N_SKILLS] * N_SKILLS
        q_model = model1 if player == 1 else model2
        values = []
        for candidate in range(N_SKILLS):
            val = 0.0
            for opp_s, prob in enumerate(mix):
                x = torch.tensor(base_enc, dtype=torch.float32).clone().unsqueeze(0)
                if player == 1:
                    x[0, -2] = candidate / (N_SKILLS - 1)
                    x[0, -1] = opp_s / (N_SKILLS - 1)
                else:
                    x[0, -2] = opp_s / (N_SKILLS - 1)
                    x[0, -1] = candidate / (N_SKILLS - 1)
                with torch.no_grad():
                    val += float(prob) * float(q_model(x).item())
            values.append(val)
        return "q", values

    if strategy in {"nash-p-hard", "nash-p-br", "nash-p", "nash-p-minimax", "nash-p-adaptive"}:
        phi = _build_phi_table(base_enc, None, player, model_p, None)
        if strategy == "nash-p-hard":
            values = phi.max(dim=1).values if player == 1 else phi.max(dim=0).values
        elif strategy in {"nash-p-br", "nash-p"}:
            values = phi[:, other_skill_idx] if player == 1 else phi[other_skill_idx, :]
        else:
            values = phi.min(dim=1).values if player == 1 else phi.min(dim=0).values
        return "phi", _float_list(values)

    return "unknown", []


def build_decision_diagnostic_row(
    *,
    strategy: str,
    strategy1: str,
    strategy2: str,
    matchup_index: int,
    episode_id: int,
    player: int,
    obs,
    info,
    state_encoder_fn,
    model_p,
    model1,
    model2,
    other_skill_idx: int,
    selected_idx: int,
    decision_index: int,
    physics_step: int,
    selection_mode: str,
    tau: float,
    confidence_margin: float,
    temperature: float,
    epsilon: float,
) -> Dict[str, Any]:
    base_enc = _encoded_policy_state(obs, info, player, state_encoder_fn)
    score_type, action_values = _policy_action_values(
        strategy,
        base_enc,
        player,
        other_skill_idx,
        model_p,
        model1,
        model2,
    )
    q_values = _q_values_for_player(base_enc, player, other_skill_idx, model1, model2)
    phi_values = _phi_values_for_player(base_enc, player, other_skill_idx, model_p)

    if score_type in {"fixed", "random"}:
        if score_type == "fixed":
            probs = np.zeros(N_SKILLS, dtype=float)
            probs[selected_idx] = 1.0
        else:
            probs = np.ones(N_SKILLS, dtype=float) / N_SKILLS
    else:
        probs = _selection_probabilities(
            np.asarray(action_values, dtype=float),
            selection_mode,
            tau,
            confidence_margin,
            temperature,
            epsilon,
        ) if action_values else np.full(N_SKILLS, np.nan)

    row = {
        "matchup_index": matchup_index,
        "strategy1": strategy1,
        "strategy2": strategy2,
        "strategy": strategy,
        "episode_id": episode_id,
        "player": player,
        "player_label": "P1" if player == 1 else "P2",
        "decision_index": decision_index,
        "physics_step": physics_step,
        "opponent_skill": skill_from_index(other_skill_idx),
        "selected_skill": skill_from_index(selected_idx),
        "selected_skill_idx": selected_idx,
        "score_type": score_type,
        "action_values_json": _json_dumps(action_values),
        "q_values_json": _json_dumps(q_values),
        "phi_values_json": _json_dumps(phi_values),
        "selection_probabilities_json": _json_dumps(_float_list(probs)),
        "selected_probability": float(probs[selected_idx]) if len(probs) > selected_idx else "",
        "state_json": _json_dumps(_float_list(base_enc)),
        "ball_x": float(obs[36]),
        "ball_y": float(obs[37]),
        "ball_z": float(obs[38]),
        "ball_vx": float(obs[39]),
        "ball_vy": float(obs[40]),
        "ball_vz": float(obs[41]),
        "next_state_json": "",
        "next_action_values_json": "",
        "next_q_values_json": "",
        "next_phi_values_json": "",
        "final_winner": "",
        "player_won": "",
        "truncated": "",
        "termination_reason": "",
        "final_rally_length": "",
        "final_physics_steps": "",
    }
    return row


def _pick_with_softmax_fallback(
    action_scores: torch.Tensor,
    tau: float,
    confidence_margin: float,
) -> int:
    """
    Pick argmax unless the top two scores are too close, then sample softmax.

    This avoids deterministic tie-breaking artifacts on flat Φ surfaces while
    preserving argmax behavior when the model expresses a clear preference.
    """
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

    Five learned strategies
    -----------------------
    nash-p-hard      Optimistic Φ heuristic. Player 1 chooses the row whose best
                     column is largest; Player 2 chooses the column whose best
                     row is largest. This is not a true Nash solver.
    nash-p-br        Conditional best response under the v3 potential convention:
                     both players maximize Φ while fixing the other player's
                     currently observed skill.
                     Falls back to softmax over conditional response scores
                     when the top-2 gap is below `confidence_margin`.
                     (Alias: "nash-p" for backwards compatibility.)
    nash-p-minimax   Heuristic: maximin over Φ rows/columns.
    nash-p-adaptive  Heuristic: maximin scores, with softmax when the top-2 gap
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
    tau               : float — softmax temperature for Φ-strategy fallbacks
    confidence_margin : float — gap threshold for Φ argmax/softmax switch
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

    # ------------------------------------------------------------------ #
    # nash-p-hard: optimistic heuristic over Φ rows/columns                #
    # ------------------------------------------------------------------ #
    if strategy == "nash-p-hard":
        def pick_hard(player, obs_vec, _other_skill_idx, info=None):
            phi = _build_phi_table(obs_vec, info, player, model_p, state_encoder_fn)
            if player == 1:
                action_scores = phi.max(dim=1).values  # best column per P1 row
            else:
                action_scores = phi.max(dim=0).values  # best row per P2 column
            return _dispatch(action_scores)
        return pick_hard

    # ------------------------------------------------------------------ #
    # nash-p-br: best response fixing opp's current skill                 #
    # (original "nash-p" — kept as alias too)                             #
    # ------------------------------------------------------------------ #
    if strategy in ("nash-p-br", "nash-p"):
        def pick_br(player, obs_vec, other_skill_idx, info=None):
            phi = _build_phi_table(obs_vec, info, player, model_p, state_encoder_fn)
            if player == 1:
                action_scores = phi[:, other_skill_idx]
            else:
                action_scores = phi[other_skill_idx, :]
            return _dispatch(action_scores)
        return pick_br

    # ------------------------------------------------------------------ #
    # nash-p-minimax: maximin heuristic                                    #
    # ------------------------------------------------------------------ #
    if strategy == "nash-p-minimax":
        def pick_minimax(player, obs_vec, other_skill_idx, info=None):
            phi = _build_phi_table(obs_vec, info, player, model_p, state_encoder_fn)
            if player == 1:
                action_scores = phi.min(dim=1).values   # worst column per P1 row
            else:
                action_scores = phi.min(dim=0).values   # worst row per P2 column
            return _dispatch(action_scores)
        return pick_minimax

    # ------------------------------------------------------------------ #
    # nash-p-adaptive: maximin heuristic + softmax fallback when flat      #
    # ------------------------------------------------------------------ #
    # if strategy == "nash-p-adaptive":
    #     def pick_adaptive(player, obs_vec, other_skill_idx, info=None):
    #         phi = _build_phi_table(obs_vec, info, player, model_p, state_encoder_fn)
    #         if player == 1:
    #             action_scores = phi.min(dim=1).values
    #         else:
    #             action_scores = phi.min(dim=0).values   # worst row per P2 column
    #         return _dispatch(action_scores)
    #     return pick_adaptive
    if strategy == "nash-p-adaptive":
        # Separate opponent-skill counts for each player.
        # P1 tracks P2's observed skills.
        # P2 tracks P1's observed skills.
        opponent_counts = {
            1: np.full(N_SKILLS, 0.1, dtype=np.float64),
            2: np.full(N_SKILLS, 0.1, dtype=np.float64),
        }

        def pick_adaptive(player, obs_vec, other_skill_idx, info=None):
            phi = _build_phi_table(
                obs_vec,
                info,
                player,
                model_p,
                state_encoder_fn,
            )  # shape (5, 5): rows=P1 skills, cols=P2 skills

            # Observe the opponent's current/incoming skill.
            opponent_counts[player][other_skill_idx] += 1.0

            probs_np = opponent_counts[player] / opponent_counts[player].sum()
            probs = torch.tensor(
                probs_np,
                dtype=phi.dtype,
                device=phi.device,
            )

            if player == 1:
                # For each P1 action, expected Phi over P2 skill distribution.
                # phi shape: [P1_action, P2_action]
                action_scores = phi @ probs

            else:
                # For each P2 action, expected Phi over P1 skill distribution.
                action_scores = probs @ phi

            return _dispatch(action_scores)

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

        IBR_STEPS = 4   # alternating rounds before returning

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

                # Player 2 best-responds to s1 by maximizing Q2, its own
                # discounted utility target.
                q2_vals = []
                for opp_s in range(N_SKILLS):
                    x = base_enc.clone().unsqueeze(0)
                    x[0, -2] = s1    / (N_SKILLS - 1)
                    x[0, -1] = opp_s / (N_SKILLS - 1)
                    with torch.no_grad():
                        q2_vals.append(model2(x).item())
                s2 = int(np.argmax(q2_vals))

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
    selection_mode: str = "argmax",
    temperature: float = 1.0,
    epsilon: float = 0.0,
    rng1=None,
    rng2=None,
    reset_mode: str = "ready",
    skill_profile: str = "aggressive",
    gantry_speed_scale: float = 1.0,
    collect_replay_metadata: bool = False,
    replay_metadata: Optional[List[Dict[str, Any]]] = None,
    matchup_index: int = 0,
    capture_original_videos: bool = False,
    original_video_dir: str = "data/rendered_rallies",
    original_video_fps: int = 60,
    original_capture_every: int = 1,
    original_video_limit: Optional[int] = None,
    original_render_truncated_only: bool = False,
    saved_video_metadata: Optional[List[Dict[str, Any]]] = None,
    decision_log_rows: Optional[List[Dict[str, Any]]] = None,
) -> MatchupResult:
    """
    Run one 5-skill matchup headlessly.

    Important:
    - One-time global warmup.
    - After warmup, keep running until we complete n_episodes.
    - If an episode hits max_steps_per_episode, truncate and reset it.
    - max_total_steps is an optional absolute safety valve. By default there
      is no total-step cap, so heavy truncation cannot silently reduce the
      number of completed episodes.
    - model1 / model2: Q-value models required when strategy1 or strategy2 is
      'ibr' or 'ibr-q'.
    - rng1 / rng2: independent numpy Generators for player 1 / player 2
      probabilistic skill selection. Passing the same generator to both
      players would couple their stochastic draws; keep them separate so
      each player's sampling is reproducible but not correlated with the
      other's.
    """
    from nash_skills.env_wrapper import SkillEnv

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

    # env = SkillEnv(proc_id=1, history=HISTORY)
    # Use ready-reset mode to avoid the initial "serve" phase and start with a rally. Align with the updated env wrapper that supports ready-reset mode.
    env = SkillEnv(
        proc_id=1,
        history=HISTORY,
        reset_mode=reset_mode,
        skill_profile=skill_profile,
        gantry_speed_scale=gantry_speed_scale,
    )

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
    attempted_episodes = 0
    active_replay_row: Optional[Dict[str, Any]] = None
    active_decisions: List[Dict[str, Any]] = []
    active_recorder: Optional[EpisodeVideoRecorder] = None
    saved_original_videos = 0
    decision_rows_for_episode: List[int] = []
    last_decision_row_by_player: Dict[int, int] = {}

    policy_settings = {
        "tau": tau,
        "confidence_margin": confidence_margin,
        "selection_mode": selection_mode,
        "temperature": temperature,
        "epsilon": epsilon,
        "seeded_generators": rng1 is not None or rng2 is not None,
    }

    def _begin_replay_attempt() -> None:
        nonlocal obs, info, prev_ball_x, steps_in_episode, curr_rally_len
        nonlocal attempted_episodes, active_replay_row, active_decisions
        nonlocal active_recorder, saved_original_videos
        nonlocal decision_rows_for_episode, last_decision_row_by_player
        env.set_skills(skill_from_index(curr_idx1), skill_from_index(curr_idx2))
        active_replay_row = {
            "episode_id": attempted_episodes,
            "matchup_index": matchup_index,
            "strategy1": strategy1,
            "strategy2": strategy2,
            "p1_initial_skill": skill_from_index(curr_idx1),
            "p2_initial_skill": skill_from_index(curr_idx2),
            "np_random_state": encode_np_random_state(np.random.get_state()),
            "rng1_state": _encode_generator_state(rng1),
            "rng2_state": _encode_generator_state(rng2),
            "reset_mode": reset_mode,
            "skill_profile": skill_profile,
            "gantry_speed_scale": gantry_speed_scale,
            "policy_selection_settings": _json_dumps(policy_settings),
            "max_steps": max_steps_per_episode,
            "warmup_steps": warmup_steps,
        }
        obs, info = env.reset()
        prev_ball_x = float(obs[36])
        steps_in_episode = 0
        curr_rally_len = 0
        active_decisions = [{
            "step": 0,
            "rally_length": 0,
            "p1_skill": skill_from_index(curr_idx1),
            "p2_skill": skill_from_index(curr_idx2),
        }]
        decision_rows_for_episode = []
        last_decision_row_by_player = {}
        if (
            capture_original_videos
            and (original_video_limit is None or saved_original_videos < original_video_limit)
        ):
            active_recorder = EpisodeVideoRecorder(
                env,
                original_video_dir,
                original_video_fps,
                original_capture_every,
            )
        else:
            active_recorder = None
        attempted_episodes += 1

    def _finish_replay_attempt(winner: str, truncated: bool, termination_reason: str) -> None:
        nonlocal active_replay_row, active_decisions, active_recorder, saved_original_videos
        if active_replay_row is None or replay_metadata is None:
            row = active_replay_row
        else:
            row = active_replay_row
        if row is None:
            if active_recorder is not None:
                active_recorder.cleanup()
                active_recorder = None
            return
        active_replay_row.update({
            "winner": winner,
            "truncated": truncated,
            "termination_reason": termination_reason,
            "physics_steps": steps_in_episode,
            "rally_length": curr_rally_len,
            "decision_steps": _json_dumps(active_decisions),
            "p1_skill_sequence": _skill_sequence_from_decisions(active_decisions, "p1_skill"),
            "p2_skill_sequence": _skill_sequence_from_decisions(active_decisions, "p2_skill"),
            "final_ball_state": _final_ball_state(obs, info),
        })
        if replay_metadata is not None:
            replay_metadata.append(active_replay_row)
        if active_recorder is not None:
            qualifies = winner == "opp" and (not original_render_truncated_only or truncated)
            if qualifies and (original_video_limit is None or saved_original_videos < original_video_limit):
                final_path = Path(original_video_dir) / f"{matchup_replay_stem(active_replay_row)}.mp4"
                saved_path = active_recorder.finish(True, final_path)
                if saved_path is not None:
                    _write_video_metadata(saved_path, active_replay_row)
                    saved_original_videos += 1
                    if saved_video_metadata is not None:
                        saved_video_metadata.append(dict(active_replay_row, video_path=str(saved_path)))
                    print(f"Saved original episode video: {saved_path}", flush=True)
            else:
                active_recorder.finish(False, Path(original_video_dir) / "discarded.mp4")
            active_recorder = None
        active_replay_row = None
        active_decisions = []

    def _finish_decision_log_attempt(winner: str, truncated: bool, termination_reason: str) -> None:
        if decision_log_rows is None:
            return
        player_winner = 1 if winner == "ego" else 2 if winner == "opp" else 0
        for row_index in decision_rows_for_episode:
            row = decision_log_rows[row_index]
            row["final_winner"] = winner
            row["player_won"] = (
                int(row["player"]) == player_winner
                if player_winner in (1, 2)
                else ""
            )
            row["truncated"] = bool(truncated)
            row["termination_reason"] = termination_reason
            row["final_rally_length"] = curr_rally_len
            row["final_physics_steps"] = steps_in_episode

    def _append_decision_log_row(row: Dict[str, Any]) -> None:
        if decision_log_rows is None:
            return
        player = int(row["player"])
        previous_index = last_decision_row_by_player.get(player)
        if previous_index is not None:
            previous = decision_log_rows[previous_index]
            previous["next_state_json"] = row["state_json"]
            previous["next_action_values_json"] = row["action_values_json"]
            previous["next_q_values_json"] = row["q_values_json"]
            previous["next_phi_values_json"] = row["phi_values_json"]
        decision_log_rows.append(row)
        row_index = len(decision_log_rows) - 1
        decision_rows_for_episode.append(row_index)
        last_decision_row_by_player[player] = row_index

    track_attempt_metadata = collect_replay_metadata or capture_original_videos

    if track_attempt_metadata:
        _begin_replay_attempt()
    else:
        decision_rows_for_episode = []
        last_decision_row_by_player = {}

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
        if active_recorder is not None:
            active_recorder.capture(steps_in_episode)

        e_c, o_c, e_s, o_s = _parse_contact_lines(lines)
        ego_contacts += e_c
        opp_contacts += o_c
        ego_successes += e_s
        opp_successes += o_s

        curr_ball_x = float(obs[36])

        if (prev_ball_x - TABLE_SHIFT) * (curr_ball_x - TABLE_SHIFT) < 0:
            curr_rally_len += 1
            pre_idx1 = curr_idx1
            pre_idx2 = curr_idx2
            selected_idx1 = pick1(1, obs, pre_idx2, info)
            if decision_log_rows is not None:
                episode_id = (
                    int(active_replay_row["episode_id"])
                    if active_replay_row is not None
                    else attempted_episodes
                )
                _append_decision_log_row(
                    build_decision_diagnostic_row(
                        strategy=strategy1,
                        strategy1=strategy1,
                        strategy2=strategy2,
                        matchup_index=matchup_index,
                        episode_id=episode_id,
                        player=1,
                        obs=obs,
                        info=info,
                        state_encoder_fn=state_encoder_fn,
                        model_p=model_p,
                        model1=model1,
                        model2=model2,
                        other_skill_idx=pre_idx2,
                        selected_idx=selected_idx1,
                        decision_index=curr_rally_len,
                        physics_step=steps_in_episode,
                        selection_mode=selection_mode,
                        tau=tau,
                        confidence_margin=confidence_margin,
                        temperature=temperature,
                        epsilon=epsilon,
                    )
                )
            curr_idx1 = selected_idx1

            selected_idx2 = pick2(2, obs, curr_idx1, info)
            if decision_log_rows is not None:
                episode_id = (
                    int(active_replay_row["episode_id"])
                    if active_replay_row is not None
                    else attempted_episodes
                )
                _append_decision_log_row(
                    build_decision_diagnostic_row(
                        strategy=strategy2,
                        strategy1=strategy1,
                        strategy2=strategy2,
                        matchup_index=matchup_index,
                        episode_id=episode_id,
                        player=2,
                        obs=obs,
                        info=info,
                        state_encoder_fn=state_encoder_fn,
                        model_p=model_p,
                        model1=model1,
                        model2=model2,
                        other_skill_idx=curr_idx1,
                        selected_idx=selected_idx2,
                        decision_index=curr_rally_len,
                        physics_step=steps_in_episode,
                        selection_mode=selection_mode,
                        tau=tau,
                        confidence_margin=confidence_margin,
                        temperature=temperature,
                        epsilon=epsilon,
                    )
                )
            curr_idx2 = selected_idx2
            env.set_skills(skill_from_index(curr_idx1), skill_from_index(curr_idx2))
            if track_attempt_metadata:
                active_decisions.append({
                    "step": steps_in_episode,
                    "rally_length": curr_rally_len,
                    "p1_skill": skill_from_index(curr_idx1),
                    "p2_skill": skill_from_index(curr_idx2),
                })

            if strategy1 in _LEARNED_STRATEGIES:
                skill_usage[skill_from_index(curr_idx1)] += 1

        prev_ball_x = curr_ball_x

        if done:
            winner = _infer_winner(obs, info)
            termination_reason = (
                str(info.get("termination_reason"))
                if isinstance(info, dict) and info.get("termination_reason") is not None
                else "done"
            )
            if winner == "ego":
                ego_wins += 1
            else:
                opp_wins += 1

            rally_lengths.append(curr_rally_len)
            episode_steps.append(steps_in_episode)
            completed_episodes += 1
            _finish_decision_log_attempt(winner, False, termination_reason)
            _finish_replay_attempt(winner, False, termination_reason)

            curr_rally_len = 0
            steps_in_episode = 0

            env.set_skills(skill_from_index(curr_idx1), skill_from_index(curr_idx2))
            if track_attempt_metadata:
                if completed_episodes < n_episodes:
                    _begin_replay_attempt()
            else:
                obs, info = env.reset()
                prev_ball_x = float(obs[36])
                attempted_episodes += 1
                decision_rows_for_episode = []
                last_decision_row_by_player = {}
            continue

        if steps_in_episode >= max_steps_per_episode:
            truncated_episodes += 1
            _finish_decision_log_attempt("truncated", True, "max_steps")
            _finish_replay_attempt("truncated", True, "max_steps")
            curr_rally_len = 0
            steps_in_episode = 0

            env.set_skills(skill_from_index(curr_idx1), skill_from_index(curr_idx2))
            if track_attempt_metadata:
                _begin_replay_attempt()
            else:
                obs, info = env.reset()
                prev_ball_x = float(obs[36])
                attempted_episodes += 1
                decision_rows_for_episode = []
                last_decision_row_by_player = {}

    if active_recorder is not None:
        active_recorder.cleanup()

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


def replay_matchup_episode(env, bundle: Dict[str, Any], row: Dict[str, Any], recorder: EpisodeVideoRecorder) -> None:
    ppo = bundle["ppo"]
    model_p = bundle["model_p"]
    model1 = bundle["model1"]
    model2 = bundle["model2"]
    state_encoder_fn = bundle["state_encoder_fn"]
    settings = _json_loads(row.get("policy_selection_settings"), {})

    np.random.set_state(decode_np_random_state(str(row["np_random_state"])))
    rng1 = _decode_generator_state(row.get("rng1_state"))
    rng2 = _decode_generator_state(row.get("rng2_state"))
    pick1 = make_picker(
        str(row["strategy1"]),
        model_p,
        state_encoder_fn=state_encoder_fn,
        tau=float(settings.get("tau", 0.2)),
        confidence_margin=float(settings.get("confidence_margin", 0.05)),
        model1=model1,
        model2=model2,
        selection_mode=str(settings.get("selection_mode", "argmax")),
        temperature=float(settings.get("temperature", 1.0)),
        epsilon=float(settings.get("epsilon", 0.0)),
        rng=rng1,
    )
    pick2 = make_picker(
        str(row["strategy2"]),
        model_p,
        state_encoder_fn=state_encoder_fn,
        tau=float(settings.get("tau", 0.2)),
        confidence_margin=float(settings.get("confidence_margin", 0.05)),
        model1=model1,
        model2=model2,
        selection_mode=str(settings.get("selection_mode", "argmax")),
        temperature=float(settings.get("temperature", 1.0)),
        epsilon=float(settings.get("epsilon", 0.0)),
        rng=rng2,
    )

    curr_idx1 = skill_index(str(row["p1_initial_skill"]))
    curr_idx2 = skill_index(str(row["p2_initial_skill"]))
    env.set_skills(skill_from_index(curr_idx1), skill_from_index(curr_idx2))
    obs, info = env.reset()
    prev_ball_x = float(obs[36])
    max_steps = int(row["max_steps"])
    decisions = [{
        "step": 0,
        "rally_length": 0,
        "p1_skill": skill_from_index(curr_idx1),
        "p2_skill": skill_from_index(curr_idx2),
    }]
    rally_length = 0
    physics_steps = 0
    winner = "truncated"
    truncated = True
    termination_reason = "max_steps"

    for step in range(1, max_steps + 1):
        obs1 = _build_obs1(obs, info)
        obs2 = _build_obs2(obs, info)
        action1, _ = ppo.predict(obs1, deterministic=True)
        action2, _ = ppo.predict(obs2, deterministic=True)
        action = np.zeros(18)
        action[:9] = action1[:9]
        action[9:] = action2[:9]

        (obs, _, done, _, info), _lines = _capture_env_step(env, action)
        physics_steps = step
        recorder.capture(step)

        curr_ball_x = float(obs[36])
        if (prev_ball_x - TABLE_SHIFT) * (curr_ball_x - TABLE_SHIFT) < 0:
            rally_length += 1
            curr_idx1 = pick1(1, obs, curr_idx2, info)
            curr_idx2 = pick2(2, obs, curr_idx1, info)
            env.set_skills(skill_from_index(curr_idx1), skill_from_index(curr_idx2))
            decisions.append({
                "step": step,
                "rally_length": rally_length,
                "p1_skill": skill_from_index(curr_idx1),
                "p2_skill": skill_from_index(curr_idx2),
            })
        prev_ball_x = curr_ball_x

        if done:
            winner = _infer_winner(obs, info)
            termination_reason = (
                str(info.get("termination_reason"))
                if isinstance(info, dict) and info.get("termination_reason") is not None
                else "done"
            )
            truncated = False
            break

    replayed = {
        "winner": winner,
        "truncated": truncated,
        "physics_steps": physics_steps,
        "rally_length": rally_length,
        "termination_reason": termination_reason if not truncated else "max_steps",
        "final_ball_state": _final_ball_state(obs, info),
        "p1_skill_sequence": _skill_sequence_from_decisions(decisions, "p1_skill"),
        "p2_skill_sequence": _skill_sequence_from_decisions(decisions, "p2_skill"),
    }
    _warn_if_replay_differs(row, replayed)


def matchup_replay_stem(row: Dict[str, Any]) -> str:
    winner = str(row["winner"])
    outcome = "opp_win" if winner == "opp" else "ego_win" if winner == "ego" else "truncated"
    return (
        f"{row['strategy1']}_vs_{row['strategy2']}_"
        f"ep{row['episode_id']}_{outcome}_{row['physics_steps']}steps"
    )


def select_targeted_replays(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], str]:
    video_dir = args.video_dir
    if manual_render_requested(args):
        selected, manual_dir = prompt_manual_replays(rows)
        if manual_dir is not None:
            video_dir = manual_dir
        return selected, video_dir

    selected = rows
    if args.render_losses_only:
        selected = select_player1_loss_replays(selected)
    if args.render_truncated_only:
        selected = [row for row in selected if bool(row.get("truncated"))]

    limit = render_episode_limit(args)
    if limit is not None:
        selected = selected[:limit]
    return selected, video_dir


def run_targeted_replay(
    args: argparse.Namespace,
    ppo,
    model_p,
    model1,
    model2,
    state_encoder_fn,
    rows: List[Dict[str, Any]],
) -> None:
    if not _render_options_requested(args):
        return
    selected, video_dir = select_targeted_replays(args, rows)
    if not selected:
        print("No episodes selected for replay.", flush=True)
        return

    from nash_skills.env_wrapper import SkillEnv

    env = SkillEnv(
        proc_id=1,
        history=HISTORY,
        reset_mode=args.reset_mode,
        skill_profile=args.skill_profile,
        gantry_speed_scale=args.gantry_speed_scale,
    )
    bundle = {
        "ppo": ppo,
        "model_p": model_p,
        "model1": model1,
        "model2": model2,
        "state_encoder_fn": state_encoder_fn,
    }
    try:
        saved = replay_selected_episodes(
            env,
            bundle,
            selected,
            replay_one=replay_matchup_episode,
            filename_stem=matchup_replay_stem,
            video_dir=video_dir,
            fps=args.video_fps,
            capture_every=args.capture_every,
        )
    finally:
        env.close()
    for path in saved:
        print(f"Saved replay video: {path}", flush=True)


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


def save_decision_log(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    preferred = [
        "matchup_index",
        "strategy1",
        "strategy2",
        "strategy",
        "episode_id",
        "player",
        "player_label",
        "decision_index",
        "physics_step",
        "opponent_skill",
        "selected_skill",
        "selected_skill_idx",
        "score_type",
        "selected_probability",
        "final_winner",
        "player_won",
        "truncated",
        "termination_reason",
        "final_rally_length",
        "final_physics_steps",
        "ball_x",
        "ball_y",
        "ball_z",
        "ball_vx",
        "ball_vy",
        "ball_vz",
        "action_values_json",
        "q_values_json",
        "phi_values_json",
        "selection_probabilities_json",
        "state_json",
        "next_state_json",
        "next_action_values_json",
        "next_q_values_json",
        "next_phi_values_json",
    ]
    extra = sorted({key for row in rows for key in row if key not in preferred})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=preferred + extra, extrasaction="ignore")
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
        help=(
            "Optional safety cap on total simulator steps per matchup. "
            "Default None means keep running until --episodes done episodes are collected."
        ),
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
        help="Softmax temperature for flat-surface fallback in nash-p-hard/br/adaptive (default: 0.2)",
    )
    parser.add_argument(
        "--confidence-margin",
        type=float,
        default=0.05,
        dest="confidence_margin",
        help="Top-2 score gap below which nash-p-hard/br/adaptive use softmax instead of argmax (default: 0.05)",
    )
    parser.add_argument(
        "--selection-mode",
        default="argmax",
        dest="selection_mode",
        choices=["argmax", "softmax", "epsilon_argmax", "epsilon_softmax"],
        help=(
            "Skill-selection mode applied after computing action scores from Φ:\n"
            "  argmax          — deterministic argmax (default; preserves original behavior)\n"
            "  softmax         — sample from softmax(scores / --temperature)\n"
            "  epsilon_argmax  — argmax with ε-greedy uniform exploration\n"
            "  epsilon_softmax — softmax with ε-uniform mixing\n"
            "Note: 'argmax' still uses the existing confidence-margin softmax fallback."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for --selection-mode softmax/epsilon_softmax (default: 1.0)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Exploration rate in [0,1] for --selection-mode epsilon_argmax/epsilon_softmax (default: 0.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for probabilistic selection modes (default: None = non-reproducible)",
    )
    parser.add_argument(
        "--matchup",
        nargs=2,
        metavar=("STRATEGY1", "STRATEGY2"),
        default=None,
        help="Run only one targeted policy pair, e.g. --matchup nash-p-hard left",
    )
    parser.add_argument(
        "--render-losses-only",
        action="store_true",
        help="In targeted mode, save videos captured from original player-1 loss episodes.",
    )
    parser.add_argument(
        "--render-truncated-only",
        action="store_true",
        help="Restrict targeted rendering to truncated rallies.",
    )
    parser.add_argument(
        "--render-episodes",
        nargs="?",
        const="manual",
        default=None,
        help="Save at most N qualifying videos; with no N, prompt for episode IDs in experimental replay mode.",
    )
    parser.add_argument("--save-video", action="store_true", help="Save targeted diagnostic replay videos.")
    parser.add_argument("--video-dir", default="data/rendered_rallies", help="Directory for targeted replay videos.")
    parser.add_argument("--video-fps", type=int, default=60, help="Frames per second for replay videos.")
    parser.add_argument("--capture-every", type=int, default=1, help="Capture one video frame every N environment steps.")
    parser.add_argument(
        "--experimental-post-eval-replay",
        action="store_true",
        help="Use the older post-evaluation replay workflow instead of relying only on original episode capture.",
    )
    parser.add_argument("--reset-mode", choices=["clean", "ready", "carryover"], default="ready")
    parser.add_argument("--skill-profile", choices=SKILL_PROFILE_NAMES, default="aggressive")
    parser.add_argument("--gantry-speed-scale", type=float, default=1.0)
    parser.add_argument(
        "--model-dir",
        default=None,
        dest="model_dir",
        help=(
            "Override the directory from which model .pth files are loaded "
            "(default: 'models/'). Example: --model-dir models_new"
        ),
    )
    parser.add_argument(
        "--decision-log",
        default=None,
        help=(
            "Optional CSV path for per-player skill-decision diagnostics. "
            "Records P1 and P2 decisions separately at every counted net crossing."
        ),
    )
    args = parser.parse_args()

    if _render_options_requested(args) and args.matchup is None:
        raise SystemExit("Rendering options require --matchup STRATEGY1 STRATEGY2")
    if args.save_video and not (args.render_losses_only or args.experimental_post_eval_replay):
        raise SystemExit("--save-video requires --render-losses-only or --experimental-post-eval-replay")
    if args.render_truncated_only and not (args.render_losses_only or args.experimental_post_eval_replay):
        raise SystemExit("--render-truncated-only requires --render-losses-only or --experimental-post-eval-replay")
    if args.render_episodes == "manual" and not args.experimental_post_eval_replay:
        raise SystemExit("--render-episodes without N requires --experimental-post-eval-replay")
    if args.matchup is not None:
        unknown = [strategy for strategy in args.matchup if strategy not in VALID_STRATEGIES]
        if unknown:
            raise SystemExit(f"Unknown strategy in --matchup: {', '.join(unknown)}")
    if args.render_episodes not in (None, "manual"):
        try:
            args.render_episodes = int(args.render_episodes)
        except ValueError as exc:
            raise SystemExit("--render-episodes must be an integer or omitted for manual selection") from exc
        if args.render_episodes <= 0:
            raise SystemExit("--render-episodes must be positive")
    if args.video_fps <= 0:
        raise SystemExit("--video-fps must be positive")
    if args.capture_every <= 0:
        raise SystemExit("--capture-every must be positive")

    matchups = [tuple(args.matchup)] if args.matchup is not None else DEFAULT_MATCHUPS
    experimental_replay = bool(args.matchup is not None and args.experimental_post_eval_replay)
    original_capture = _original_capture_requested(args)

    from stable_baselines3 import PPO
    from model_arch import SimpleModel, FactoredModel

    print("Loading models...")
    ppo = PPO.load(PPO_MODEL_PATH)

    def _model_path(default_path: str) -> str:
        if args.model_dir is None:
            return default_path
        import os
        return os.path.join(args.model_dir, os.path.basename(default_path))

    if args.arch == "factored":
        # FactoredModel weights trained on 116-dim raw obs.
        # Pick v2 (minibatch-mean) or v3 (same-state per-sample) trained weights
        # based on the pipeline flag. One of --v2-5skill / --v3-5skill is required.
        if args.v3_5skill:
            model_p_path = _model_path(MODEL_P_5SK_V3_FACTORED_PATH)
            pipeline_tag = "v3-5skill-factored"
        elif args.v2_5skill:
            model_p_path = _model_path(MODEL_P_5SK_FACTORED_PATH)
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
        model_p_path = _model_path(MODEL_P_5SK_V3_PATH)
        model_p = SimpleModel(V2_STATE_DIM, [64, 32, 16], 1, last_layer_activation=None)
        pipeline_tag = "v3-5skill"
    elif args.v2_5skill:
        # 5-skill v2: 76-dim encoded states, all 5 skills, discounted-return training
        from nash_skills.v2.state_encoder import STATE_DIM as V2_STATE_DIM
        model_p_path = _model_path(MODEL_P_5SK_V2_PATH)
        model_p = SimpleModel(V2_STATE_DIM, [64, 32, 16], 1, last_layer_activation=None)
        pipeline_tag = "v2-5skill"
    elif args.v2:
        # 4-skill v2: 76-dim encoded states (original v2 diagnostic)
        from nash_skills.v2.state_encoder import STATE_DIM as V2_STATE_DIM
        model_p_path = _model_path(MODEL_P_V2_PATH)
        model_p = SimpleModel(V2_STATE_DIM, [64, 32, 16], 1, last_layer_activation=None)
        pipeline_tag = "v2-4skill"
    else:
        # v1: original 116-dim raw obs
        model_p_path = _model_path(MODEL_P_5SK_PATH)
        model_p = SimpleModel(116, [64, 32, 16], 1, last_layer_activation=None)
        pipeline_tag = "v1-5skill"

    model_p.load_state_dict(_safe_load_state_dict(model_p_path))
    model_p.eval()

    # Q-value models — needed for ibr / ibr-q
    needs_q = args.decision_log is not None or any(s in {"ibr", "ibr-q"} for s, _ in matchups) or any(
        s in {"ibr", "ibr-q"} for _, s in matchups
    )
    model1 = model2 = None
    if needs_q:
        # Architecture branch first: under --arch factored we construct
        # FactoredModel and skip the SimpleModel construction below.
        if args.arch == "factored":
            if args.v3_5skill:
                _q1_path = _model_path(MODEL1_5SK_V3_FACTORED_PATH)
                _q2_path = _model_path(MODEL2_5SK_V3_FACTORED_PATH)
            else:  # args.v2_5skill (the model_p branch above already enforced this)
                _q1_path = _model_path(MODEL1_5SK_FACTORED_PATH)
                _q2_path = _model_path(MODEL2_5SK_FACTORED_PATH)
            # 76-dim encoded data: 74 state dims + 2 skill dims (see model_p above).
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
                # _q1_path = MODEL1_5SK_V2_PATH
                # _q2_path = MODEL2_5SK_V2_PATH
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
                return encode_ego(obs, info)
                # return encode_opp(obs, info)

        state_encoder_fn = _v2_state_encoder
    else:
        state_encoder_fn = None

    print(f"  Loaded PPO:         {PPO_MODEL_PATH}")
    print(f"  Loaded potential:   {model_p_path}  ({pipeline_tag})")
    print(
        f"\nRunning {len(matchups)} matchups "
        f"to {args.episodes} completed episodes each "
        f"(warmup={args.warmup}, max_steps_per_episode={args.steps}) ...\n"
    )

    results: List[MatchupResult] = []
    replay_rows: List[Dict[str, Any]] = []
    saved_video_rows: List[Dict[str, Any]] = []
    decision_log_rows: List[Dict[str, Any]] = []

    for matchup_idx, (s1, s2) in enumerate(matchups):
        print(f"  [{s1} vs {s2}] ...")

        # Independent, per-matchup, per-player seeds: player 1 and player 2
        # must not share a generator (coupled stochastic draws), and each
        # matchup must not replay the same draws as the previous one.
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
            reset_mode=args.reset_mode,
            skill_profile=args.skill_profile,
            gantry_speed_scale=args.gantry_speed_scale,
            collect_replay_metadata=experimental_replay,
            replay_metadata=replay_rows,
            matchup_index=matchup_idx,
            capture_original_videos=original_capture,
            original_video_dir=args.video_dir,
            original_video_fps=args.video_fps,
            original_capture_every=args.capture_every,
            original_video_limit=render_episode_limit(args),
            original_render_truncated_only=args.render_truncated_only,
            saved_video_metadata=saved_video_rows,
            decision_log_rows=decision_log_rows if args.decision_log is not None else None,
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

    if args.decision_log is not None:
        save_decision_log(decision_log_rows, args.decision_log)
        print(f"Decision log saved to: {args.decision_log}")

    if experimental_replay:
        run_targeted_replay(
            args,
            ppo,
            model_p,
            model1,
            model2,
            state_encoder_fn,
            replay_rows,
        )


if __name__ == "__main__":
    main()
