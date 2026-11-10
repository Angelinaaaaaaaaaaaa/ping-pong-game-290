"""
Calibrate the preparation poses (nash_skills.defend_target_pose) for one
low-level controller and one skill profile.

A preparation position should be where the receiver's racket actually meets
the corresponding attack. So: both players use the same attack skill for a
whole rally (left / center_safe / right), no defend mechanism is active, and
at every racket contact we record the hitter's racket target (own frame,
the frame defend_target_pose uses) and its tracking error. The first contact
of each rally is skipped (that ball was not aimed by a skill). Contacts the
controller was not tracking (error > --max-track-err) are dropped.

Output: per position the median x/y/z, plus x/z medians shared across
positions (the current pose model uses one x and one z), as JSON.

    MUJOCO_GL=egl PYTHONPATH=. python scripts/calibrate_prep_poses.py \\
        --ppo logs/best_model_tracker1/best_model --profile aggressive \\
        --rallies 40 --output analysis/prep_pose_calibration/aggressive.json
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import contextlib
import hashlib
import io
import json

import numpy as np

ATTACK_FOR_POSITION = {"left": "left", "center": "center_safe", "right": "right"}


def _mean_orientation(quats):
    """Mean racket orientation (scalar-last quaternion, the env's convention) and its median spread."""
    from scipy.spatial.transform import Rotation as R
    rots = R.from_quat(np.asarray(quats, dtype=float))
    mean = rots.mean()
    spread = np.degrees((rots * mean.inv()).magnitude())
    env_form = R.from_matrix(mean.as_matrix()).as_quat()  # the env's sign convention (see prep_pose_tables)
    return {"quat": [float(v) for v in env_form], "quat_spread_deg": float(np.median(spread))}


def summarize_contacts(records, max_track_err):
    """records: dicts with attack, target [x, y, z], track_err. Returns per-position medians."""
    kept = [r for r in records if r["track_err"] <= max_track_err]
    out = {}
    for position, attack in ATTACK_FOR_POSITION.items():
        pts = np.array([r["target"] for r in kept if r["attack"] == attack], dtype=float)
        if len(pts) == 0:
            raise ValueError(f"no converged contacts for position {position!r} (attack {attack!r})")
        med = np.median(pts, axis=0)
        q25, q75 = np.percentile(pts[:, 1], [25, 75])
        out[position] = {"x": float(med[0]), "y": float(med[1]), "z": float(med[2]), "n": int(len(pts)),
                         "y_iqr": [float(q25), float(q75)]}
        quats = [r["quat"] for r in kept if r["attack"] == attack and "quat" in r]
        if quats:
            out[position].update(_mean_orientation(quats))
    all_pts = np.array([r["target"] for r in kept], dtype=float)
    out["shared"] = {"x": float(np.median(all_pts[:, 0])), "z": float(np.median(all_pts[:, 2]))}
    out["n_dropped"] = len(records) - len(kept)
    return out


def _step(env, ppo, obs, info):
    from nash_skills.v2.selfplay_attack_defend import _build_ppo_obs
    a1, _ = ppo.predict(_build_ppo_obs(obs, info, 1), deterministic=True)
    a2, _ = ppo.predict(_build_ppo_obs(obs, info, 2), deterministic=True)
    act = np.zeros(18, dtype=np.float32)
    act[:9], act[9:] = a1[:9], a2[:9]
    with contextlib.redirect_stdout(io.StringIO()):
        return env.step(act)


def collect_contacts(env, ppo, attack, n_rallies, max_steps, seed0):
    records = []
    for ep in range(n_rallies):
        np.random.seed(seed0 + ep)
        env.set_skills(attack, attack)
        obs, info = env.reset()
        prev, n_contacts = env._env.rally_state.get("last_hitter"), 0
        for _ in range(max_steps):
            obs, _r, done, trunc, info = _step(env, ppo, obs, info)
            hitter = env._env.rally_state.get("last_hitter")
            if hitter is not None and hitter != prev:
                n_contacts += 1
                if n_contacts > 1:
                    own = env._env.curr_target if hitter == "ego" else env._env.curr_target_opp
                    err = info["diff_pos"] if hitter == "ego" else info["diff_pos_opp"]
                    records.append({"attack": attack, "target": [float(v) for v in own[:3]],
                                    "quat": [float(v) for v in own[3:7]],
                                    "track_err": float(np.linalg.norm(err))})
            prev = hitter
            if done or trunc:
                break
    return records


def _md5(path):
    with open(path if path.endswith(".zip") else path + ".zip", "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    from stable_baselines3 import PPO
    from nash_skills.env_wrapper import SkillEnv
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--rallies", type=int, default=40, help="rallies per attack")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--max-track-err", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    env = SkillEnv(proc_id=1, history=4, reset_mode="ready", skill_profile=args.profile, reset_targets=True)
    ppo = PPO.load(args.ppo, device="cpu")
    records = []
    for attack in ATTACK_FOR_POSITION.values():
        records += collect_contacts(env, ppo, attack, args.rallies, args.max_steps, args.seed * 10000)
    env.close()
    summary = summarize_contacts(records, args.max_track_err)
    result = {"profile": args.profile, "ppo": args.ppo, "ppo_md5": _md5(args.ppo), "args": vars(args),
              "summary": summary}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    for k in ("left", "center", "right"):
        s = summary[k]
        print(f"{args.profile:<20} {k:<7} y={s['y']:+.3f} IQR [{s['y_iqr'][0]:+.3f},{s['y_iqr'][1]:+.3f}] "
              f"x={s['x']:+.3f} z={s['z']:.3f} n={s['n']} quat={np.round(s.get('quat', []), 4).tolist()} "
              f"spread={s.get('quat_spread_deg', float('nan')):.1f} deg")
    print(f"{args.profile:<20} shared  x={summary['shared']['x']:+.3f} z={summary['shared']['z']:.3f} "
          f"(dropped {summary['n_dropped']} unconverged)  ppo md5 {result['ppo_md5']}")


if __name__ == "__main__":
    main()
