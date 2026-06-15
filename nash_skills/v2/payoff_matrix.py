"""
Phase 2 diagnostic: empirical 5x5 fixed-skill payoff matrix.

Each of the 25 ego×opp skill pairs is run for a fixed number of episodes.
Results are saved as CSV and JSON so you can verify whether `right` is
genuinely dominant or whether the Nash-p policy is collapsing incorrectly.

Run from project root:
    MUJOCO_GL=cgl venv/bin/python nash_skills/v2/payoff_matrix.py
    MUJOCO_GL=cgl venv/bin/python nash_skills/v2/payoff_matrix.py \
        --episodes 30 --steps 600 \
        --output-csv  skill_eval/payoff_matrix.csv \
        --output-json skill_eval/payoff_matrix.json
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import csv
import dataclasses
import json
from typing import List, Optional

from nash_skills.skills import SKILL_NAMES


def parse_skills(raw: str | None) -> list:
    """
    Parse a comma-separated skill subset string.

    Parameters
    ----------
    raw : str or None
        e.g. "left_mid,right_short,right", or None to use all SKILL_NAMES.

    Returns
    -------
    list of validated skill name strings.

    Raises
    ------
    ValueError if any name is not in SKILL_NAMES.
    """
    if raw is None:
        return list(SKILL_NAMES)
    names = [s.strip() for s in raw.split(",") if s.strip()]
    unknown = [n for n in names if n not in SKILL_NAMES]
    if unknown:
        raise ValueError(
            f"Unknown skill(s): {unknown}. Valid skills: {SKILL_NAMES}"
        )
    return names


def build_payoff_matchups(skills_subset: list | None) -> list:
    """
    Build the list of (ego_skill, opp_skill) pairs for the payoff matrix.

    Parameters
    ----------
    skills_subset : list of skill names, or None to use all SKILL_NAMES.

    Returns
    -------
    list of (ego, opp) tuples covering every combination within the subset.
    """
    skills = list(skills_subset) if skills_subset is not None else list(SKILL_NAMES)
    return [(ego, opp) for ego in skills for opp in skills]


# Default matchups — all N×N pairs using the active skill set
PAYOFF_MATCHUPS = build_payoff_matchups(None)

DEFAULT_N_EPISODES = 30
DEFAULT_MAX_STEPS  = 600
DEFAULT_WARMUP     = 200


# --------------------------------------------------------------------------- #
# PayoffEntry dataclass                                                        #
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class PayoffEntry:
    ego_skill: str
    opp_skill: str
    ego_wins: int
    done_episodes: int
    attempted_episodes: int
    # Computed fields — populated in __post_init__ so dataclasses.fields() sees them
    win_rate: Optional[float] = dataclasses.field(init=False)
    done_fraction: Optional[float] = dataclasses.field(init=False)

    def __post_init__(self):
        self.win_rate = (
            self.ego_wins / self.done_episodes if self.done_episodes > 0 else None
        )
        self.done_fraction = (
            self.done_episodes / self.attempted_episodes
            if self.attempted_episodes > 0 else None
        )

    @property
    def reliability_label(self) -> str:
        if self.done_episodes < 10:
            return 'inconclusive'
        if self.done_fraction is not None and self.done_fraction < 0.30:
            return 'truncation-dominated'
        return 'reliable'


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #

def run_payoff_matrix(
    ppo,
    n_episodes: int = DEFAULT_N_EPISODES,
    max_steps: int = DEFAULT_MAX_STEPS,
    warmup_steps: int = DEFAULT_WARMUP,
    output_csv: str = "skill_eval/payoff_matrix.csv",
    output_json: str = "skill_eval/payoff_matrix.json",
    skills_subset: list | None = None,
) -> List[PayoffEntry]:
    """
    Run fixed-skill ego×opp matchups for every pair in skills_subset (default: all skills).

    Parameters
    ----------
    ppo           : stable_baselines3 PPO model (frozen)
    n_episodes    : target completed episodes per pair
    max_steps     : per-episode step cap before truncation
    warmup_steps  : one-time warmup steps before counting
    output_csv    : CSV output path
    output_json   : JSON output path
    skills_subset : list of skill names to include, or None for all skills
    """
    import io
    from contextlib import redirect_stdout
    import numpy as np
    from nash_skills.env_wrapper import SkillEnv
    from nash_skills.skills import skill_from_index, skill_index

    HISTORY = 4
    TABLE_SHIFT = 1.5  # net x-position; ball_x > 1.5 → opp's side

    def _build_obs1(obs, info):
        o = np.zeros(9 + 9 + 7 + 7 + 9 * HISTORY, dtype=np.float32)
        o[:9]   = obs[:9]
        o[9:18] = obs[18:27]
        o[18:21] = info["diff_pos"]
        o[21:25] = info["diff_quat"]
        o[25:32] = info["target"]
        o[32:]  = obs[42: 42 + HISTORY * 9]
        return o

    def _build_obs2(obs, info):
        o = np.zeros(9 + 9 + 7 + 7 + 9 * HISTORY, dtype=np.float32)
        o[:9]   = obs[9:18]
        o[9:18] = obs[27:36]
        o[18:21] = info["diff_pos_opp"]
        o[21:25] = info["diff_quat_opp"]
        o[25:32] = info["target_opp"]
        o[32:]  = obs[42 + HISTORY * 9: 42 + 2 * HISTORY * 9]
        return o

    def _infer_winner(obs, info):
        # obs layout: qpos[:18], qvel[:18], ball.xpos[36:39], ...
        # done fires when ball goes out of reach of one player.
        # ball_x > TABLE_SHIFT (1.5) → ball past net on opp side → opp missed → ego wins
        # ball_x < TABLE_SHIFT       → ball on ego side           → ego missed → opp wins
        ball_x = float(obs[36])
        return "ego" if ball_x > TABLE_SHIFT else "opp"

    active_matchups = build_payoff_matchups(skills_subset)
    entries: List[PayoffEntry] = []

    for ego_skill, opp_skill in active_matchups:
        print(f"  [{ego_skill} vs {opp_skill}] ...", flush=True)

        env = SkillEnv(proc_id=1, history=HISTORY)
        env.set_skills(ego_skill, opp_skill)
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
            buf = io.StringIO()
            with redirect_stdout(buf):
                obs, _, done, _, info = env.step(action)
            total_steps += 1
            if done:
                env.set_skills(ego_skill, opp_skill)
                obs, info = env.reset()
                prev_ball_x = float(obs[36])

        # Counted evaluation
        ego_wins = 0
        done_eps = 0
        trunc_eps = 0
        steps_in_ep = 0
        max_total = warmup_steps + n_episodes * max_steps * 5

        while done_eps < n_episodes and total_steps < max_total:
            obs1 = _build_obs1(obs, info)
            obs2 = _build_obs2(obs, info)
            action1, _ = ppo.predict(obs1, deterministic=True)
            action2, _ = ppo.predict(obs2, deterministic=True)
            action = np.zeros(18)
            action[:9] = action1[:9]
            action[9:] = action2[:9]
            buf = io.StringIO()
            with redirect_stdout(buf):
                obs, _, done, _, info = env.step(action)
            total_steps += 1
            steps_in_ep += 1

            if done:
                winner = _infer_winner(obs, info)
                if winner == "ego":
                    ego_wins += 1
                done_eps += 1
                steps_in_ep = 0
                env.set_skills(ego_skill, opp_skill)
                obs, info = env.reset()
            elif steps_in_ep >= max_steps:
                trunc_eps += 1
                steps_in_ep = 0
                env.set_skills(ego_skill, opp_skill)
                obs, info = env.reset()

        env.close()

        entry = PayoffEntry(
            ego_skill=ego_skill,
            opp_skill=opp_skill,
            ego_wins=ego_wins,
            done_episodes=done_eps,
            attempted_episodes=done_eps + trunc_eps,
        )
        entries.append(entry)

        wr = f"{entry.win_rate:.0%}" if entry.win_rate is not None else "---"
        df = f"{entry.done_fraction:.2f}" if entry.done_fraction is not None else "---"
        print(f"    done={done_eps}  trunc={trunc_eps}  win_rate={wr}  done_frac={df}",
              flush=True)

    save_payoff_csv(entries, output_csv)
    save_payoff_json(entries, output_json)
    return entries


# --------------------------------------------------------------------------- #
# Save helpers                                                                 #
# --------------------------------------------------------------------------- #

def save_payoff_csv(entries: List[PayoffEntry], path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    rows = []
    for e in entries:
        rows.append({
            "ego_skill": e.ego_skill,
            "opp_skill": e.opp_skill,
            "ego_wins": e.ego_wins,
            "done_episodes": e.done_episodes,
            "attempted_episodes": e.attempted_episodes,
            "win_rate": round(e.win_rate, 4) if e.win_rate is not None else "",
            "done_fraction": round(e.done_fraction, 4) if e.done_fraction is not None else "",
            "reliability_label": e.reliability_label,
        })
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_payoff_json(entries: List[PayoffEntry], path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    data = []
    for e in entries:
        d = dataclasses.asdict(e)
        d["win_rate"] = e.win_rate
        d["done_fraction"] = e.done_fraction
        d["reliability_label"] = e.reliability_label
        data.append(d)
    with open(path, "w") as f:
        json.dump({"payoff_matrix": data}, f, indent=2)


# --------------------------------------------------------------------------- #
# Analysis helpers                                                             #
# --------------------------------------------------------------------------- #

def print_payoff_table(entries: List[PayoffEntry]) -> None:
    """Print win-rate matrix: rows=ego, cols=opp."""
    from nash_skills.skills import SKILL_NAMES

    # Collect only the skills actually present in entries
    ego_skills = list(dict.fromkeys(e.ego_skill for e in entries))
    opp_skills = list(dict.fromkeys(e.opp_skill for e in entries))
    all_skills = ego_skills  # symmetric: ego_skills == opp_skills for full matrix

    col_w = 15
    header = f"{'ego \\ opp':<15}" + "".join(f"{s:>{col_w}}" for s in opp_skills)
    print(f"\n{'PAYOFF MATRIX (win rate)':^{len(header)}}")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    lookup = {(e.ego_skill, e.opp_skill): e for e in entries}
    for ego in ego_skills:
        row = f"{ego:<15}"
        for opp in opp_skills:
            e = lookup.get((ego, opp))
            if e is None or e.win_rate is None:
                row += f"{'---':>{col_w}}"
            else:
                row += f"{e.win_rate:>{col_w}.0%}"
        print(row)
    print("-" * len(header))


def iterated_best_response(entries: List[PayoffEntry], iterations: int = 20) -> dict:
    """
    Compute an approximate Nash equilibrium via Iterated Best Response (IBR)
    on the empirical payoff matrix.

    IBR: alternate between each player picking the pure best-response to the
    other's current mixed strategy, with smoothing (fictitious-play style).
    Each player's strategy is a probability distribution over skills.

    Parameters
    ----------
    entries    : list of PayoffEntry from run_payoff_matrix
    iterations : number of IBR rounds (more = closer to fixed point)

    Returns
    -------
    dict with keys:
        'ego_strategy'  : {skill: probability}
        'opp_strategy'  : {skill: probability}
        'ego_br_skill'  : pure best-response skill for ego
        'opp_br_skill'  : pure best-response skill for opp
        'matrix'        : {(ego, opp): win_rate}
    """
    import numpy as np

    ego_skills = list(dict.fromkeys(e.ego_skill for e in entries))
    opp_skills = list(dict.fromkeys(e.opp_skill for e in entries))
    n_ego = len(ego_skills)
    n_opp = len(opp_skills)

    # Build payoff matrix A[i,j] = ego win rate when ego plays i, opp plays j
    A = np.full((n_ego, n_opp), 0.5)
    for e in entries:
        if e.win_rate is not None and e.ego_skill in ego_skills and e.opp_skill in opp_skills:
            i = ego_skills.index(e.ego_skill)
            j = opp_skills.index(e.opp_skill)
            A[i, j] = e.win_rate

    # Opp payoff = 1 - ego win rate (zero-sum)
    B = 1.0 - A

    # Start uniform
    p = np.ones(n_ego) / n_ego   # ego mixed strategy
    q = np.ones(n_opp) / n_opp   # opp mixed strategy

    # Fictitious play accumulator
    p_sum = p.copy()
    q_sum = q.copy()

    for t in range(1, iterations + 1):
        # Ego best-responds to opp's current average strategy
        ego_values = A @ (q_sum / q_sum.sum())
        ego_br = int(np.argmax(ego_values))
        p_new = np.zeros(n_ego); p_new[ego_br] = 1.0

        # Opp best-responds to ego's current average strategy
        opp_values = B.T @ (p_sum / p_sum.sum())
        opp_br = int(np.argmax(opp_values))
        q_new = np.zeros(n_opp); q_new[opp_br] = 1.0

        p_sum += p_new
        q_sum += q_new

    p_avg = p_sum / p_sum.sum()
    q_avg = q_sum / q_sum.sum()

    # Final pure best responses against each other's average
    ego_br_idx = int(np.argmax(A @ q_avg))
    opp_br_idx = int(np.argmax(B.T @ p_avg))

    matrix = {(e.ego_skill, e.opp_skill): e.win_rate for e in entries if e.win_rate is not None}

    return {
        "ego_strategy": {ego_skills[i]: round(float(p_avg[i]), 3) for i in range(n_ego)},
        "opp_strategy": {opp_skills[j]: round(float(q_avg[j]), 3) for j in range(n_opp)},
        "ego_br_skill": ego_skills[ego_br_idx],
        "opp_br_skill": opp_skills[opp_br_idx],
        "matrix": matrix,
    }


def print_ibr_result(result: dict) -> None:
    """Print the IBR approximate Nash equilibrium result."""
    print("\n--- IBR Approximate Nash Equilibrium ---")
    print("Ego mixed strategy (fictitious-play average):")
    for skill, prob in result["ego_strategy"].items():
        bar = "#" * int(prob * 20)
        print(f"  {skill:<18} {prob:.3f}  {bar}")
    print(f"  → Pure best response: {result['ego_br_skill']}")
    print("\nOpp mixed strategy (fictitious-play average):")
    for skill, prob in result["opp_strategy"].items():
        bar = "#" * int(prob * 20)
        print(f"  {skill:<18} {prob:.3f}  {bar}")
    print(f"  → Pure best response: {result['opp_br_skill']}")
    print("----------------------------------------")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run empirical N×N fixed-skill payoff matrix with IBR Nash approximation"
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_N_EPISODES,
                        help=f"Target done episodes per pair (default: {DEFAULT_N_EPISODES})")
    parser.add_argument("--steps", type=int, default=DEFAULT_MAX_STEPS,
                        help=f"Max steps per episode (default: {DEFAULT_MAX_STEPS})")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help=f"Warmup steps (default: {DEFAULT_WARMUP})")
    parser.add_argument("--output-csv",  dest="output_csv",  default="skill_eval/payoff_matrix.csv",
                        help="CSV output path")
    parser.add_argument("--output-json", dest="output_json", default="skill_eval/payoff_matrix.json",
                        help="JSON output path")
    parser.add_argument("--skills", default=None,
                        help="Comma-separated subset of skills for the matrix "
                             f"(default: all {len(SKILL_NAMES)} skills). "
                             "Example: --skills left_mid,right_short,right")
    parser.add_argument("--ibr-iterations", type=int, default=100,
                        help="Fictitious-play IBR iterations for Nash approximation (default: 100)")
    args = parser.parse_args()

    # Validate --skills before loading PPO (fail fast on typos)
    active_skills = parse_skills(args.skills)
    active_matchups = build_payoff_matchups(active_skills)

    from stable_baselines3 import PPO
    PPO_MODEL_PATH = "logs/best_model_tracker1/best_model"
    print(f"Loading PPO from {PPO_MODEL_PATH} ...")
    ppo = PPO.load(PPO_MODEL_PATH)

    print(f"\nRunning {len(active_matchups)} matchups over skills {active_skills} "
          f"({args.episodes} done eps each, max_steps={args.steps}) ...\n")

    entries = run_payoff_matrix(
        ppo=ppo,
        n_episodes=args.episodes,
        max_steps=args.steps,
        warmup_steps=args.warmup,
        output_csv=args.output_csv,
        output_json=args.output_json,
        skills_subset=active_skills,
    )

    print_payoff_table(entries)

    # Always run IBR to show approximate Nash regardless of matrix quality
    ibr_result = iterated_best_response(entries, iterations=args.ibr_iterations)
    print_ibr_result(ibr_result)

    print(f"\nCSV  saved to: {args.output_csv}")
    print(f"JSON saved to: {args.output_json}")
