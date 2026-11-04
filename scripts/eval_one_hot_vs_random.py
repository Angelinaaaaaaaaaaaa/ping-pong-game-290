"""
Win rate vs random for the one-hot (84-dim) main-pipeline model --
the comparison point to the already-run legacy-encoder result
(nash-p-hard 58%, nash-p-br 60%, nash-p-minimax 72%, nash-p-adaptive 55%,
ibr 68%, ibr-q 55%, all via `eval_matchup.py --v3-5skill`).

`eval_matchup.py`'s --v3-5skill path hardcodes the legacy encoder (a
closure named _v2_state_encoder wrapping nash_skills.v2.state_encoder.
encode_ego/encode_opp) with no flag to switch encoders. Rather than
modifying that already-complex, working, shared file, this script
reuses its exported run_matchup() / make_picker() functions as a
library and supplies a one-hot-aware state_encoder_fn closure instead --
the same injection point _v2_state_encoder itself uses internally.

IMPORTANT, replicated deliberately for comparability: _v2_state_encoder
has a quirk -- for player==2 it still calls encode_ego(obs, info)
(encode_opp is present but commented out), meaning the legacy "vs
random" numbers above were computed with player 2's Q/Φ evaluation fed
player 1's own spatial state (gantry/arm/history/diff_pos), not player
2's -- only the skill-identity slots are player-symmetric and therefore
unaffected. This looks like a latent bug, not something introduced
here. It is replicated in the one-hot closure below ON PURPOSE so this
script's number is comparable to the already-reported legacy one (same
methodology, different encoder) rather than introducing a second,
uncontrolled difference on top of the encoder change. Flagged here and
in the script's printed output -- not silently fixed, not silently
ignored.

Run:
    MUJOCO_GL=egl PYTHONPATH=. venv/bin/python \\
        scripts/eval_one_hot_vs_random.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse

import numpy as np
import torch
from stable_baselines3 import PPO

from model_arch import SimpleModel
from nash_skills.eval_matchup import run_matchup, PPO_MODEL_PATH
from nash_skills.v2.state_encoder_one_hot import encode_ego, encode_opp, STATE_DIM_ONE_HOT

STRATEGIES = ["nash-p-hard", "nash-p-br", "nash-p-minimax", "nash-p-adaptive", "ibr", "ibr-q"]
MODEL1_PATH = "models/model1_5skill_v3_one_hot.pth"
MODEL2_PATH = "models/model2_5skill_v3_one_hot.pth"
MODEL_P_PATH = "models/model_p_5skill_v3_one_hot.pth"


_USE_CORRECTED_ENCODER = os.environ.get("USE_ENCODE_OPP") == "1"


def _one_hot_state_encoder(obs, info, player):
    # Replicates _v2_state_encoder's player==2 quirk by default -- see
    # module docstring. Set USE_ENCODE_OPP=1 to use the "corrected"
    # encode_opp call instead, for a follow-up comparison isolating how
    # much of the one-hot-vs-random gap is this quirk vs the encoding
    # change itself.
    if player == 1:
        return encode_ego(obs, info)
    elif _USE_CORRECTED_ENCODER:
        return encode_opp(obs, info)
    else:
        return encode_ego(obs, info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=300)
    args = ap.parse_args()

    print(f"Loading PPO from {PPO_MODEL_PATH} ...")
    ppo = PPO.load(PPO_MODEL_PATH, device="cpu")

    model_p = SimpleModel(STATE_DIM_ONE_HOT, [64, 32, 16], 1, last_layer_activation=None)
    model_p.load_state_dict(torch.load(MODEL_P_PATH, map_location="cpu", weights_only=True))
    model_p.eval()

    model1 = SimpleModel(STATE_DIM_ONE_HOT, [64, 32, 16], 1)
    model1.load_state_dict(torch.load(MODEL1_PATH, map_location="cpu", weights_only=True))
    model1.eval()
    model2 = SimpleModel(STATE_DIM_ONE_HOT, [64, 32, 16], 1)
    model2.load_state_dict(torch.load(MODEL2_PATH, map_location="cpu", weights_only=True))
    model2.eval()

    print(f"Loaded one-hot models: {MODEL1_PATH}, {MODEL2_PATH}, {MODEL_P_PATH}\n")
    print(f"encoder mode for player 2: {'encode_opp (corrected)' if _USE_CORRECTED_ENCODER else 'encode_ego (replicated quirk, default)'}")
    print(f"Running {len(STRATEGIES)} matchups vs random, {args.episodes} episodes each\n")

    print(f"{'strategy':<16}{'win_rate':<10}{'completed':<11}{'truncated':<11}{'avg_rally':<10}")
    rows = []
    for strategy in STRATEGIES:
        result = run_matchup(
            strategy, "random", ppo, model_p,
            n_episodes=args.episodes,
            max_steps_per_episode=args.steps,
            warmup_steps=args.warmup,
            state_encoder_fn=_one_hot_state_encoder,
            model1=model1, model2=model2,
        )
        wr = result.win_rate
        rows.append((strategy, wr, result.episodes, result.truncated_episodes, result.avg_rally_length))
        print(f"{strategy:<16}{wr:<10.2f}{result.episodes:<11}{result.truncated_episodes:<11}"
              f"{result.avg_rally_length:<10.1f}")

    print("\n(player==2's Q/Phi evaluation uses encode_ego, not encode_opp -- a pre-existing "
          "quirk in eval_matchup.py's legacy encoder wiring, replicated here on purpose for "
          "comparability with the already-reported legacy-encoder vs-random numbers. See "
          "module docstring.)")


if __name__ == "__main__":
    main()
