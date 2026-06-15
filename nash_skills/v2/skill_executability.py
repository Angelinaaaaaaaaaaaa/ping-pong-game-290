"""
Phase 2 diagnostic: per-skill executability under the frozen PPO.

For each of the 5 fixed skills, run a solo evaluation where the ego is locked
to that skill and the opponent plays a neutral (random) skill.  Measure:
  - done_fraction    : fraction of attempts that end with a real done signal
  - truncation rate  : complement of done_fraction
  - avg_rally_length : mean number of ball crossings per episode
  - is_feasible      : done_fraction >= FEASIBILITY_THRESHOLD

Skills with low done_fraction produce mostly truncated episodes, which means
the frozen PPO cannot execute them reliably — this inflates truncation counts
and degrades Q-model training quality.

Run from project root:
    MUJOCO_GL=cgl venv/bin/python nash_skills/v2/skill_executability.py
    MUJOCO_GL=cgl venv/bin/python nash_skills/v2/skill_executability.py \
        --episodes 30 --steps 600 \
        --output-csv  skill_eval/skill_executability.csv \
        --output-json skill_eval/skill_executability.json
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

FEASIBILITY_THRESHOLD = 0.30    # done_fraction below this → infeasible
DEFAULT_N_EPISODES    = 30
DEFAULT_MAX_STEPS     = 600
DEFAULT_WARMUP        = 200


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


# --------------------------------------------------------------------------- #
# ExecutabilityResult dataclass                                                #
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class ExecutabilityResult:
    skill: str
    done_episodes: int
    truncated_episodes: int
    avg_rally_length: Optional[float]
    # Computed fields — populated in __post_init__ so dataclasses.fields() sees them
    attempted_episodes: int = dataclasses.field(init=False)
    done_fraction: Optional[float] = dataclasses.field(init=False)

    def __post_init__(self):
        self.attempted_episodes = self.done_episodes + self.truncated_episodes
        if self.attempted_episodes == 0:
            self.done_fraction = None
        else:
            self.done_fraction = self.done_episodes / self.attempted_episodes

    @property
    def is_feasible(self) -> bool:
        if self.done_fraction is None:
            return False
        return self.done_fraction >= FEASIBILITY_THRESHOLD


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #

def run_executability(
    ppo,
    n_episodes: int = DEFAULT_N_EPISODES,
    max_steps: int = DEFAULT_MAX_STEPS,
    warmup_steps: int = DEFAULT_WARMUP,
    output_csv: str = "skill_eval/skill_executability.csv",
    output_json: str = "skill_eval/skill_executability.json",
    skills_subset: List[str] | None = None,
) -> List[ExecutabilityResult]:
    """
    For each skill in skills_subset (default: all SKILL_NAMES), run n_episodes
    with ego locked to that skill vs a random opponent skill.

    Parameters
    ----------
    ppo           : stable_baselines3 PPO model (frozen)
    n_episodes    : target completed episodes per skill
    max_steps     : per-episode step cap
    warmup_steps  : one-time warmup steps before counting
    output_csv    : CSV output path
    output_json   : JSON output path
    skills_subset : list of skill names to evaluate, or None for all skills
    """
    import io
    from contextlib import redirect_stdout
    import numpy as np
    from nash_skills.env_wrapper import SkillEnv
    from nash_skills.skills import skill_from_index

    HISTORY = 4
    TABLE_SHIFT = 1.5

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

    active_skills = list(skills_subset) if skills_subset is not None else list(SKILL_NAMES)
    results: List[ExecutabilityResult] = []

    for skill in active_skills:
        print(f"  Evaluating skill '{skill}' ...", flush=True)

        # Opponent plays random skill each rally
        opp_skill = skill_from_index(np.random.randint(len(SKILL_NAMES)))
        env = SkillEnv(proc_id=1, history=HISTORY)
        env.set_skills(skill, opp_skill)
        obs, info = env.reset()
        total_steps = 0

        # One-time warmup
        while total_steps < warmup_steps:
            obs1 = _build_obs1(obs, info)
            obs2 = _build_obs2(obs, info)
            a1, _ = ppo.predict(obs1, deterministic=True)
            a2, _ = ppo.predict(obs2, deterministic=True)
            action = np.zeros(18)
            action[:9] = a1[:9]
            action[9:] = a2[:9]
            buf = io.StringIO()
            with redirect_stdout(buf):
                obs, _, done, _, info = env.step(action)
            total_steps += 1
            if done:
                # Randomise opp skill each episode
                opp_skill = skill_from_index(np.random.randint(len(SKILL_NAMES)))
                env.set_skills(skill, opp_skill)
                obs, info = env.reset()

        # Counted evaluation
        done_eps  = 0
        trunc_eps = 0
        rally_lengths: List[int] = []
        steps_in_ep = 0
        curr_rally_len = 0
        prev_ball_x = float(obs[36])
        max_total = warmup_steps + n_episodes * max_steps * 5

        while done_eps < n_episodes and total_steps < max_total:
            obs1 = _build_obs1(obs, info)
            obs2 = _build_obs2(obs, info)
            a1, _ = ppo.predict(obs1, deterministic=True)
            a2, _ = ppo.predict(obs2, deterministic=True)
            action = np.zeros(18)
            action[:9] = a1[:9]
            action[9:] = a2[:9]
            buf = io.StringIO()
            with redirect_stdout(buf):
                obs, _, done, _, info = env.step(action)
            total_steps += 1
            steps_in_ep += 1

            curr_ball_x = float(obs[36])
            if (prev_ball_x - TABLE_SHIFT) * (curr_ball_x - TABLE_SHIFT) < 0:
                curr_rally_len += 1
                # Randomise opp skill at each crossing
                opp_skill = skill_from_index(np.random.randint(len(SKILL_NAMES)))
                env.set_skills(skill, opp_skill)
            prev_ball_x = curr_ball_x

            if done:
                done_eps += 1
                rally_lengths.append(curr_rally_len)
                curr_rally_len = 0
                steps_in_ep = 0
                opp_skill = skill_from_index(np.random.randint(len(SKILL_NAMES)))
                env.set_skills(skill, opp_skill)
                obs, info = env.reset()
                prev_ball_x = float(obs[36])
            elif steps_in_ep >= max_steps:
                trunc_eps += 1
                rally_lengths.append(curr_rally_len)
                curr_rally_len = 0
                steps_in_ep = 0
                opp_skill = skill_from_index(np.random.randint(len(SKILL_NAMES)))
                env.set_skills(skill, opp_skill)
                obs, info = env.reset()
                prev_ball_x = float(obs[36])

        env.close()

        avg_rl = float(sum(rally_lengths) / len(rally_lengths)) if rally_lengths else None
        r = ExecutabilityResult(
            skill=skill,
            done_episodes=done_eps,
            truncated_episodes=trunc_eps,
            avg_rally_length=avg_rl,
        )
        results.append(r)

        df = f"{r.done_fraction:.2f}" if r.done_fraction is not None else "---"
        arl = f"{avg_rl:.1f}" if avg_rl is not None else "---"
        print(f"    done={done_eps}  trunc={trunc_eps}  "
              f"done_frac={df}  avg_rally={arl}  "
              f"feasible={r.is_feasible}", flush=True)

    save_executability_csv(results, output_csv)
    save_executability_json(results, output_json)
    return results


# --------------------------------------------------------------------------- #
# Save helpers                                                                 #
# --------------------------------------------------------------------------- #

def save_executability_csv(results: List[ExecutabilityResult], path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    rows = []
    for r in results:
        rows.append({
            "skill": r.skill,
            "done_episodes": r.done_episodes,
            "truncated_episodes": r.truncated_episodes,
            "attempted_episodes": r.attempted_episodes,
            "done_fraction": round(r.done_fraction, 4) if r.done_fraction is not None else "",
            "avg_rally_length": round(r.avg_rally_length, 2) if r.avg_rally_length is not None else "",
            "is_feasible": r.is_feasible,
        })
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_executability_json(results: List[ExecutabilityResult], path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    data = []
    for r in results:
        d = dataclasses.asdict(r)
        d["attempted_episodes"] = r.attempted_episodes
        d["done_fraction"] = r.done_fraction
        d["is_feasible"] = r.is_feasible
        data.append(d)
    with open(path, "w") as f:
        json.dump({"executability": data}, f, indent=2)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-skill executability diagnostic under frozen PPO"
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_N_EPISODES,
                        help=f"Target done episodes per skill (default: {DEFAULT_N_EPISODES})")
    parser.add_argument("--steps", type=int, default=DEFAULT_MAX_STEPS,
                        help=f"Max steps per episode (default: {DEFAULT_MAX_STEPS})")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help=f"Warmup steps (default: {DEFAULT_WARMUP})")
    parser.add_argument("--output-csv",  dest="output_csv",  default="skill_eval/skill_executability.csv",
                        help="CSV output path")
    parser.add_argument("--output-json", dest="output_json", default="skill_eval/skill_executability.json",
                        help="JSON output path")
    parser.add_argument("--output", dest="output_csv",
                        help="Shorthand alias for --output-csv")
    parser.add_argument("--skills", default=None,
                        help="Comma-separated subset of skills to evaluate "
                             f"(default: all {len(SKILL_NAMES)} skills). "
                             f"Example: --skills left_mid,right_short,right")
    args = parser.parse_args()

    # Validate --skills before loading PPO (fail fast on typos)
    active_skills = parse_skills(args.skills)

    from stable_baselines3 import PPO
    PPO_MODEL_PATH = "logs/best_model_tracker1/best_model"
    print(f"Loading PPO from {PPO_MODEL_PATH} ...")
    ppo = PPO.load(PPO_MODEL_PATH)

    print(f"\nRunning executability for {len(active_skills)} skill(s): {active_skills} "
          f"({args.episodes} done eps each, max_steps={args.steps}) ...\n")

    results = run_executability(
        ppo=ppo,
        n_episodes=args.episodes,
        max_steps=args.steps,
        warmup_steps=args.warmup,
        output_csv=args.output_csv,
        output_json=args.output_json,
        skills_subset=active_skills,
    )

    print(f"\n{'Skill':<15} {'DoneFrac':>10} {'AvgRally':>10} {'Feasible':>10}")
    print("-" * 50)
    for r in results:
        df  = f"{r.done_fraction:.2f}" if r.done_fraction is not None else "---"
        arl = f"{r.avg_rally_length:.1f}" if r.avg_rally_length is not None else "---"
        print(f"{r.skill:<15} {df:>10} {arl:>10} {str(r.is_feasible):>10}")

    print(f"\nCSV  saved to: {args.output_csv}")
    print(f"JSON saved to: {args.output_json}")
