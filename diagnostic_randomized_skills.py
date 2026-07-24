#!/usr/bin/env python3
"""Randomized skill diagnostics for the 5-skill setup."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from diagnostic_rendering import (
    EpisodeVideoRecorder,
    add_render_args,
    decode_np_random_state,
    encode_np_random_state,
    json_safe,
    manual_render_requested,
    outcome_label,
    post_replay_requested,
    prompt_manual_replays,
    replay_selected_episodes,
    render_episode_limit,
    select_truncated_replays,
    should_render_live,
    validate_render_args,
)
from diagnostic_fixed_skill import build_ppo_obs
from nash_skills.skills import SKILL_NAMES, SKILL_PROFILE_NAMES, world_target_xy

RALLY_FIELDS = [
    "episode_id",
    "rally_id",
    "seed",
    "mode",
    "setting_index",
    "setting",
    "episode_index",
    "fixed_player",
    "fixed_skill",
    "p1_initial_skill",
    "p2_initial_skill",
    "p1_policy_type",
    "p2_policy_type",
    "winner",
    "truncated",
    "rally_length",
    "num_decisions",
    "max_steps",
    "reset_mode",
    "skill_profile",
    "gantry_speed_scale",
    "initial_state",
    "np_random_state",
]

DECISION_FIELDS = [
    "rally_id",
    "decision_t",
    "player",
    "chosen_skill",
    "winner",
    "truncated",
    "rally_length",
    "mode",
    "fixed_player",
    "fixed_skill",
]
CONTACT_FIELDS = [
    "rally_id",
    "seed",
    "mode",
    "fixed_player",
    "fixed_skill",
    "step",
    "player",
    "chosen_skill",
    "x_land",
    "y_land",
    "expected_x",
    "expected_y",
    "error_dx",
    "error_dy",
    "error_dist",
    "winner",
    "truncated",
    "rally_length",
]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def capture_step(env, action):
    buffer = StringIO()
    with redirect_stdout(buffer):
        result = env.step(action)
    return result, buffer.getvalue().splitlines()


def parse_contacts(lines: list[str], step: int, p1_skill: str, p2_skill: str, skill_profile: str) -> list[dict[str, Any]]:
    contacts = []
    for line in lines:
        if "Returned successfully by ego" in line:
            player = 1
            skill = p1_skill
        elif "Returned successfully by opp" in line:
            player = 2
            skill = p2_skill
        else:
            continue
        parts = line.split()
        try:
            x_land = float(parts[-2])
            y_land = float(parts[-1])
        except (ValueError, IndexError):
            x_land = ""
            y_land = ""
        expected_x, expected_y = world_target_xy(player, skill, profile=skill_profile)
        if x_land == "" or y_land == "":
            dx = dy = dist = ""
        else:
            dx = float(x_land) - expected_x
            dy = float(y_land) - expected_y
            dist = float(np.hypot(dx, dy))
        contacts.append({
            "step": step,
            "player": player,
            "chosen_skill": skill,
            "x_land": x_land,
            "y_land": y_land,
            "expected_x": expected_x,
            "expected_y": expected_y,
            "error_dx": dx,
            "error_dy": dy,
            "error_dist": dist,
        })
    return contacts


def policy_types(mode: str, fixed_player: int | None) -> tuple[str, str]:
    if mode == "random_vs_random":
        return "random", "random"
    if fixed_player == 1:
        return "fixed", "random"
    if fixed_player == 2:
        return "random", "fixed"
    raise ValueError(f"Invalid fixed_player for {mode}: {fixed_player}")


def choose_skill(policy_type: str, fixed_skill: str | None, rng: np.random.Generator) -> str:
    if policy_type == "fixed":
        if fixed_skill is None:
            raise ValueError("fixed policy requires fixed_skill")
        return fixed_skill
    if policy_type == "random":
        return str(rng.choice(SKILL_NAMES))
    raise ValueError(f"Unknown policy_type: {policy_type}")


def make_rally_id(seed: int, mode: str, setting: str, episode_index: int) -> str:
    return f"{seed}:{mode}:{setting}:{episode_index}"


def randomized_video_stem(
    *,
    mode: str,
    fixed_player: int | None,
    fixed_skill: str | None,
    episode_index: int,
    winner: str,
    truncated: bool,
    steps: int,
) -> str:
    if mode == "random_vs_random":
        p1_label = "random"
        p2_label = "random"
    elif fixed_player == 1:
        p1_label = f"fixed_{fixed_skill}"
        p2_label = "random"
    elif fixed_player == 2:
        p1_label = "random"
        p2_label = f"fixed_{fixed_skill}"
    else:
        raise ValueError(f"Invalid video label setting: mode={mode} fixed_player={fixed_player}")
    return f"{mode}_{p1_label}_vs_{p2_label}_ep{episode_index}_{outcome_label(winner, truncated)}_{steps}steps"


def episode_rng(seed: int, mode_index: int, setting_index: int, episode_index: int) -> np.random.Generator:
    value = (
        (seed + 1_000_003) * 1_000_003
        + mode_index * 100_003
        + setting_index * 10_007
        + episode_index
    ) % (2**32 - 1)
    return np.random.default_rng(value)


def infer_winner(obs, info) -> str:
    from nash_skills.winner_inference import infer_terminal_winner

    inferred = infer_terminal_winner(obs, info, fallback=None)
    if inferred == "ego":
        return "player1"
    if inferred == "opp":
        return "player2"
    return "truncated"


def run_rally(
    env,
    model,
    *,
    rally_id: str,
    seed: int,
    mode: str,
    fixed_player: int | None,
    fixed_skill: str | None,
    max_steps: int,
    rng: np.random.Generator,
    skill_profile: str,
    render_live: bool = False,
    video_recorder: EpisodeVideoRecorder | None = None,
    episode_id: int | None = None,
    setting_index: int | None = None,
    setting: str | None = None,
    episode_index: int | None = None,
    reset_mode: str | None = None,
    gantry_speed_scale: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    p1_policy_type, p2_policy_type = policy_types(mode, fixed_player)
    p1_skill = choose_skill(p1_policy_type, fixed_skill, rng)
    p2_skill = choose_skill(p2_policy_type, fixed_skill, rng)

    env.set_skills(p1_skill, p2_skill)
    np_random_state = encode_np_random_state(np.random.get_state())
    obs, info = env.reset()
    initial_state = json.dumps(json_safe(info.get("initial_state", {})))
    prev_ball_x = float(obs[36])
    decisions: list[dict[str, Any]] = []
    contacts: list[dict[str, Any]] = []
    winner = "truncated"
    truncated = True
    done = False
    step = 0
    decision_t = 0

    def log_current_decision() -> None:
        decisions.append({
            "rally_id": rally_id,
            "decision_t": decision_t,
            "player": 1,
            "chosen_skill": p1_skill,
            "mode": mode,
            "fixed_player": "" if fixed_player is None else fixed_player,
            "fixed_skill": "" if fixed_skill is None else fixed_skill,
        })
        decisions.append({
            "rally_id": rally_id,
            "decision_t": decision_t,
            "player": 2,
            "chosen_skill": p2_skill,
            "mode": mode,
            "fixed_player": "" if fixed_player is None else fixed_player,
            "fixed_skill": "" if fixed_skill is None else fixed_skill,
        })

    log_current_decision()

    for step in range(1, max_steps + 1):
        ppo1 = build_ppo_obs(obs, info, player=1)
        ppo2 = build_ppo_obs(obs, info, player=2)
        a1, _ = model.predict(ppo1, deterministic=True)
        a2, _ = model.predict(ppo2, deterministic=True)
        action = np.zeros(18, dtype=np.float32)
        action[:9] = a1[:9]
        action[9:] = a2[:9]

        (obs, _reward, done, _truncated, info), lines = capture_step(env, action)
        if render_live:
            env.render()
        if video_recorder is not None:
            video_recorder.capture(step)
        contacts.extend(parse_contacts(lines, step, p1_skill, p2_skill, skill_profile))
        curr_ball_x = float(obs[36])

        if done:
            winner = infer_winner(obs, info)
            truncated = winner == "truncated"
            break

        if (prev_ball_x - 1.5) * (curr_ball_x - 1.5) < 0:
            decision_t += 1
            p1_skill = choose_skill(p1_policy_type, fixed_skill, rng)
            p2_skill = choose_skill(p2_policy_type, fixed_skill, rng)
            env.set_skills(p1_skill, p2_skill)
            log_current_decision()

        prev_ball_x = curr_ball_x

    rally_length = step
    if not done:
        winner = "truncated"
        truncated = True

    for decision in decisions:
        decision["winner"] = winner
        decision["truncated"] = truncated
        decision["rally_length"] = rally_length
    for contact in contacts:
        contact.update({
            "rally_id": rally_id,
            "seed": seed,
            "mode": mode,
            "fixed_player": "" if fixed_player is None else fixed_player,
            "fixed_skill": "" if fixed_skill is None else fixed_skill,
            "winner": winner,
            "truncated": truncated,
            "rally_length": rally_length,
        })

    row = {
        "episode_id": "" if episode_id is None else episode_id,
        "rally_id": rally_id,
        "seed": seed,
        "mode": mode,
        "setting_index": "" if setting_index is None else setting_index,
        "setting": "" if setting is None else setting,
        "episode_index": "" if episode_index is None else episode_index,
        "fixed_player": "" if fixed_player is None else fixed_player,
        "fixed_skill": "" if fixed_skill is None else fixed_skill,
        "p1_initial_skill": decisions[0]["chosen_skill"],
        "p2_initial_skill": decisions[1]["chosen_skill"],
        "p1_policy_type": p1_policy_type,
        "p2_policy_type": p2_policy_type,
        "winner": winner,
        "truncated": truncated,
        "rally_length": rally_length,
        "num_decisions": decision_t + 1,
        "max_steps": max_steps,
        "reset_mode": "" if reset_mode is None else reset_mode,
        "skill_profile": skill_profile,
        "gantry_speed_scale": "" if gantry_speed_scale is None else gantry_speed_scale,
        "initial_state": initial_state,
        "np_random_state": np_random_state,
    }
    return row, decisions, contacts


def settings_for_mode(mode: str, episodes_per_setting: int, episodes: int) -> list[dict[str, Any]]:
    if mode == "fixed_vs_random":
        settings = []
        for fixed_player in (1, 2):
            for fixed_skill in SKILL_NAMES:
                settings.append({
                    "fixed_player": fixed_player,
                    "fixed_skill": fixed_skill,
                    "episodes": episodes_per_setting,
                    "setting": f"p{fixed_player}_fixed_{fixed_skill}",
                })
        return settings
    if mode == "random_vs_random":
        return [{"fixed_player": None, "fixed_skill": None, "episodes": episodes, "setting": "all_random"}]
    raise ValueError(f"Unsupported mode: {mode}")


def randomized_replay_stem(row: dict[str, Any]) -> str:
    return randomized_video_stem(
        mode=str(row["mode"]),
        fixed_player=None if row["fixed_player"] in ("", None) else int(row["fixed_player"]),
        fixed_skill=None if row["fixed_skill"] in ("", None) else str(row["fixed_skill"]),
        episode_index=int(row["episode_index"]),
        winner=str(row["winner"]),
        truncated=str(row["truncated"]).lower() in {"true", "1", "yes"},
        steps=int(row["rally_length"]),
    )


def replay_randomized_rally(env, model, row: dict[str, Any], recorder: EpisodeVideoRecorder) -> None:
    seed = int(row["seed"])
    mode = str(row["mode"])
    mode_index = 0 if mode == "fixed_vs_random" else 1
    setting_index = int(row["setting_index"])
    episode_index = int(row["episode_index"])
    fixed_player = None if row["fixed_player"] in ("", None) else int(row["fixed_player"])
    fixed_skill = None if row["fixed_skill"] in ("", None) else str(row["fixed_skill"])
    np.random.set_state(decode_np_random_state(str(row["np_random_state"])))
    run_rally(
        env,
        model,
        rally_id=str(row["rally_id"]),
        seed=seed,
        mode=mode,
        fixed_player=fixed_player,
        fixed_skill=fixed_skill,
        max_steps=int(row["max_steps"]),
        rng=episode_rng(seed, mode_index, setting_index, episode_index),
        skill_profile=str(row["skill_profile"]),
        video_recorder=recorder,
    )


def run_post_eval_replay(args, model, rows: list[dict[str, Any]]) -> None:
    if not post_replay_requested(args):
        return
    selected: list[dict[str, Any]]
    video_dir = args.video_dir
    limit = render_episode_limit(args)
    if args.render_truncated_only:
        selected = select_truncated_replays(rows, limit)
    elif manual_render_requested(args):
        selected, manual_dir = prompt_manual_replays(rows)
        if manual_dir is not None:
            video_dir = manual_dir
    else:
        selected = []
    if not selected:
        print("No episodes selected for replay.", flush=True)
        return

    from nash_skills.env_wrapper import SkillEnv

    env = SkillEnv(
        proc_id=1,
        history=4,
        reset_mode=args.reset_mode,
        skill_profile=args.skill_profile,
        gantry_speed_scale=args.gantry_speed_scale,
    )
    try:
        saved = replay_selected_episodes(
            env,
            model,
            selected,
            replay_one=replay_randomized_rally,
            filename_stem=randomized_replay_stem,
            video_dir=video_dir,
            fps=args.video_fps,
            capture_every=args.capture_every,
        )
    finally:
        env.close()
    for path in saved:
        print(f"Saved replay video: {path}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect randomized 5-skill diagnostics.")
    parser.add_argument("--mode", choices=["fixed_vs_random", "random_vs_random"], required=True)
    parser.add_argument("--episodes-per-setting", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ppo", default="logs/best_model_tracker1/best_model")
    parser.add_argument("--reset-mode", choices=["clean", "ready", "carryover"], default="ready")
    parser.add_argument("--gantry-speed-scale", type=float, default=1.0)
    parser.add_argument("--skill-profile", choices=SKILL_PROFILE_NAMES, default="current")
    add_render_args(parser)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.episodes_per_setting <= 0:
        raise ValueError("--episodes-per-setting must be positive")
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    validate_render_args(args)

    from stable_baselines3 import PPO
    from nash_skills.env_wrapper import SkillEnv

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    settings = settings_for_mode(args.mode, args.episodes_per_setting, args.episodes)
    mode_index = 0 if args.mode == "fixed_vs_random" else 1

    model = PPO.load(args.ppo)
    env = SkillEnv(
        proc_id=1,
        history=4,
        reset_mode=args.reset_mode,
        skill_profile=args.skill_profile,
        gantry_speed_scale=args.gantry_speed_scale,
    )
    rally_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    contact_rows: list[dict[str, Any]] = []
    start = time.monotonic()
    try:
        for setting_index, setting in enumerate(settings):
            print(f"Running {args.mode} {setting['setting']}", flush=True)
            for episode_index in range(setting["episodes"]):
                episode_id = len(rally_rows)
                rally_id = make_rally_id(args.seed, args.mode, setting["setting"], episode_index)
                rng = episode_rng(args.seed, mode_index, setting_index, episode_index)
                rally, decisions, contacts = run_rally(
                    env,
                    model,
                    rally_id=rally_id,
                    seed=args.seed,
                    mode=args.mode,
                    fixed_player=setting["fixed_player"],
                    fixed_skill=setting["fixed_skill"],
                    max_steps=args.steps,
                    rng=rng,
                    skill_profile=args.skill_profile,
                    render_live=should_render_live(args),
                    episode_id=episode_id,
                    setting_index=setting_index,
                    setting=setting["setting"],
                    episode_index=episode_index,
                    reset_mode=args.reset_mode,
                    gantry_speed_scale=args.gantry_speed_scale,
                )
                rally_rows.append(rally)
                decision_rows.extend(decisions)
                contact_rows.extend(contacts)
                print(
                    f"  {rally_id} winner={rally['winner']} steps={rally['rally_length']} "
                    f"decisions={rally['num_decisions']}",
                    flush=True,
                )
    finally:
        env.close()

    write_csv(out / "rallies.csv", rally_rows, RALLY_FIELDS)
    write_csv(out / "decisions.csv", decision_rows, DECISION_FIELDS)
    write_csv(out / "contacts.csv", contact_rows, CONTACT_FIELDS)
    metadata = {
        "args": vars(args),
        "skills": SKILL_NAMES,
        "rallies": len(rally_rows),
        "decisions": len(decision_rows),
        "contacts": len(contact_rows),
        "elapsed_seconds": time.monotonic() - start,
        "interpretation_note": (
            "Skill distribution in wins counts every skill chosen during rallies that eventually "
            "ended in a win; last-skill-before-terminal is reported separately."
        ),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote randomized diagnostics to {out}", flush=True)
    run_post_eval_replay(args, model, rally_rows)


if __name__ == "__main__":
    main()
