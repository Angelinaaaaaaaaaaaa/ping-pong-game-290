"""Shared rendering helpers for MuJoCo diagnostic scripts."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


def add_render_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--render", action="store_true", help="Enable live rendering with env.render().")
    parser.add_argument(
        "--render-episodes",
        nargs="?",
        const="manual",
        default=None,
        help=(
            "After evaluation, replay selected episode IDs. With no value, prompt for IDs. "
            "With an integer, cap post-evaluation replay videos to N episodes."
        ),
    )
    parser.add_argument(
        "--render-truncated-only",
        action="store_true",
        help="Only render/save episodes that reach the maximum step limit.",
    )
    parser.add_argument("--save-video", action="store_true", help="Save rendered rally videos when rgb_array is supported.")
    parser.add_argument("--video-dir", default="data/rendered_rallies", help="Directory for saved rally videos.")
    parser.add_argument("--video-fps", type=int, default=60, help="Frames per second for saved rally videos.")
    parser.add_argument("--capture-every", type=int, default=1, help="Capture one video frame every N environment steps.")


def validate_render_args(args: argparse.Namespace) -> None:
    limit = render_episode_limit(args)
    if limit is not None and limit <= 0:
        raise ValueError("--render-episodes must be positive when an integer is provided")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")
    if args.capture_every <= 0:
        raise ValueError("--capture-every must be positive")


def selected_episode_count_allows(count: int, limit: int) -> bool:
    return count < limit


def should_render_live(args: argparse.Namespace, _selected_count: int = 0) -> bool:
    return bool(args.render)


def render_episode_limit(args: argparse.Namespace) -> int | None:
    value = args.render_episodes
    if value in (None, "manual"):
        return None
    return int(value)


def manual_render_requested(args: argparse.Namespace) -> bool:
    return args.render_episodes == "manual"


def post_replay_requested(args: argparse.Namespace) -> bool:
    return bool(args.render_truncated_only or manual_render_requested(args))


def should_save_video(args: argparse.Namespace, selected_count: int, truncated: bool) -> bool:
    limit = render_episode_limit(args)
    if not (args.save_video or post_replay_requested(args)):
        return False
    if limit is not None and not selected_episode_count_allows(selected_count, limit):
        return False
    return truncated or not args.render_truncated_only


def should_attempt_video(args: argparse.Namespace, saved_count: int) -> bool:
    limit = render_episode_limit(args)
    return bool((args.save_video or post_replay_requested(args)) and (limit is None or selected_episode_count_allows(saved_count, limit)))


class EpisodeVideoRecorder:
    def __init__(self, env: Any, video_dir: str | Path, fps: int, capture_every: int):
        self.env = env
        self.video_dir = Path(video_dir)
        self.fps = fps
        self.capture_every = capture_every
        self.video_dir.mkdir(parents=True, exist_ok=True)
        temp = tempfile.NamedTemporaryFile(prefix=".tmp_rally_", suffix=".mp4", dir=self.video_dir, delete=False)
        temp.close()
        self.temp_path = Path(temp.name)
        self._writer = None
        self.frame_count = 0

    def _ensure_writer(self):
        if self._writer is None:
            try:
                import imageio.v2 as imageio
            except ImportError as exc:
                raise RuntimeError("Saving video requires imageio") from exc
            self._writer = imageio.get_writer(self.temp_path, fps=self.fps)
        return self._writer

    def capture(self, step: int) -> None:
        if step % self.capture_every != 0:
            return
        frame = self.env.render(mode="rgb_array")
        if frame is None:
            return
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        self._ensure_writer().append_data(frame)
        self.frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def finish(self, keep: bool, final_path: Path) -> Path | None:
        self.close()
        if keep and self.frame_count:
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(self.temp_path), final_path)
            return final_path
        self.cleanup()
        return None

    def cleanup(self) -> None:
        self.close()
        self.temp_path.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.cleanup()
        return False


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def outcome_label(winner: str, truncated: bool) -> str:
    return "truncated" if truncated else winner


def video_path(video_dir: str | Path, stem: str) -> Path:
    return Path(video_dir) / f"{safe_filename(stem)}.mp4"


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def encode_np_random_state(state: tuple[Any, ...]) -> str:
    name, keys, pos, has_gauss, cached_gaussian = state
    return json.dumps({
        "name": name,
        "keys": keys.tolist(),
        "pos": pos,
        "has_gauss": has_gauss,
        "cached_gaussian": cached_gaussian,
    })


def decode_np_random_state(value: str) -> tuple[Any, ...]:
    data = json.loads(value)
    return (
        data["name"],
        np.asarray(data["keys"], dtype=np.uint32),
        int(data["pos"]),
        int(data["has_gauss"]),
        float(data["cached_gaussian"]),
    )


def summarize_episode(row: dict[str, Any]) -> str:
    episode_id = row.get("episode_id", row.get("rally_id", row.get("key", "")))
    winner = row.get("winner", "")
    steps = row.get("rally_length", row.get("physics_steps", ""))
    truncated = row.get("truncated", row.get("reached_step_limit", ""))
    return f"{episode_id}: winner={winner} steps={steps} truncated={truncated}"


def row_is_truncated(row: dict[str, Any]) -> bool:
    value = row.get("truncated", row.get("reached_step_limit", False))
    return str(value).lower() in {"true", "1", "yes"}


def select_truncated_replays(rows: Iterable[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    selected = [row for row in rows if row_is_truncated(row)]
    if limit is not None:
        return selected[:limit]
    return selected


def select_manual_replays(rows: Iterable[dict[str, Any]], requested_ids: Iterable[str]) -> list[dict[str, Any]]:
    by_id = {str(row.get("episode_id", row.get("rally_id", row.get("key", "")))): row for row in rows}
    selected = []
    for episode_id in requested_ids:
        row = by_id.get(str(episode_id).strip())
        if row is not None:
            selected.append(row)
    return selected


def prompt_manual_replays(
    rows: list[dict[str, Any]],
    *,
    input_func: Callable[[str], str] = input,
    print_func: Callable[[str], None] = print,
) -> tuple[list[dict[str, Any]], str | None]:
    for row in rows:
        print_func(summarize_episode(row))
    raw_ids = input_func("Episode IDs to replay (comma-separated): ").strip()
    if not raw_ids:
        return [], None
    out_dir = input_func("Video output directory: ").strip()
    ids = [item.strip() for item in raw_ids.split(",") if item.strip()]
    return select_manual_replays(rows, ids), out_dir or None


def replay_selected_episodes(
    env: Any,
    model: Any,
    rows: Iterable[dict[str, Any]],
    *,
    replay_one: Callable[[Any, Any, dict[str, Any], EpisodeVideoRecorder], None],
    filename_stem: Callable[[dict[str, Any]], str],
    video_dir: str | Path,
    fps: int,
    capture_every: int,
) -> list[Path]:
    saved = []
    for row in rows:
        recorder = EpisodeVideoRecorder(env, video_dir, fps, capture_every)
        try:
            replay_one(env, model, row, recorder)
            saved_path = recorder.finish(True, video_path(video_dir, filename_stem(row)))
            if saved_path is not None:
                saved.append(saved_path)
        except Exception:
            recorder.cleanup()
            raise
    return saved
