"""
Gantry-only state encoder (v3.1) — drops joint angles, adds opp gantry.

Designed in response to meeting feedback after diagnostics showed the v2/v3
state encoder (76-dim) produced Q models that completely ignored opp_skill
(column-bias: ego always picks 'left' ~56% regardless of opp choice).

Root cause:
  - encode_ego only included EGO's own body state (gantry + arm joints),
    so the model never saw where the opponent was.
  - 76 dims diluted the 2-dim skill signal (2/76 = 2.6% of input).

Fix:
  - Include BOTH players' gantry positions (key spatial info)
  - Drop arm joint angles (high-dim, low-signal)
  - Keep ball position / velocity (shared game state)
  - 12-dim total → skill signal is now 2/12 = 16.7%

State layout (12 dims):
  [0:2]   ego gantry    (qpos[0:2])
  [2:4]   opp gantry    (qpos[18:20])
  [4:7]   ball position (obs[36:39])
  [7:10]  ball velocity (obs[39:42])
  [10]    ego skill (normalised, 0..1)
  [11]    opp skill (normalised, 0..1)

This encoder does NOT need the info dict — it only uses raw obs values.
That makes it possible to re-encode existing rally pickles (which store
raw_obs but not info) without re-running data collection.

STALE MODEL WARNING (post skill-slot-swap fix):
encode_opp_gantry previously placed skill indices in the wrong slots when
building the opponent's perspective (ego/opp skill were not swapped to
match the ego/opp gantry swap). Any model trained on data produced with
the old encode_opp_gantry -- including models_new/model_p_5skill_v3_gantry.pth
and models_new/model_p_5skill_v3_gantry_sym.pth -- learned from mislabeled
skill-to-outcome pairs for the opp-perspective half of the data and should
be considered STALE. They require retraining on data re-encoded with the
fixed encode_opp_gantry before being used for evaluation or diagnostics.
"""

import numpy as np

STATE_DIM = 12


def encode_ego_gantry(obs: np.ndarray) -> np.ndarray:
    """
    Encode raw 116-dim env obs into a 12-dim ego-perspective state.

    Parameters
    ----------
    obs : (116,) float32 — raw KukaTennisEnv observation

    Returns
    -------
    (12,) float32
    """
    out = np.zeros(STATE_DIM, dtype=np.float32)
    out[0:2]   = obs[0:2]      # ego gantry
    out[2:4]   = obs[18:20]    # opp gantry
    out[4:7]   = obs[36:39]    # ball position
    out[7:10]  = obs[39:42]    # ball velocity
    # Skill indices always occupy the last 2 dims of the state vector
    # (shared convention with the 76-dim encoder in augment.py).
    out[10]    = obs[-2]       # ego skill index (normalised)
    out[11]    = obs[-1]       # opp skill index (normalised)
    return out


def encode_opp_gantry(obs: np.ndarray) -> np.ndarray:
    """
    Encode raw 116-dim env obs into a 12-dim opp-perspective state.

    From opp's perspective, swap ego/opp gantry AND swap the skill indices:
    obs[-1] (original opp's own skill) becomes "their ego skill" (slot 10),
    obs[-2] (original ego's skill) becomes "their opp skill" (slot 11).
    Skill indices always occupy the last 2 dims of the state vector
    (shared convention with the 76-dim encoder in augment.py); the ego/opp
    slot swap must mirror the gantry swap above or the model is trained on
    mislabeled skill-to-outcome pairs. Ball position/velocity are shared
    physical state and do not need swapping.
    """
    out = np.zeros(STATE_DIM, dtype=np.float32)
    out[0:2]   = obs[18:20]    # opp gantry (as their "ego")
    out[2:4]   = obs[0:2]      # ego gantry (as their "opp")
    out[4:7]   = obs[36:39]    # ball position
    out[7:10]  = obs[39:42]    # ball velocity
    out[10]    = obs[-1]       # opp's own skill -> their "ego skill" slot
    out[11]    = obs[-2]       # original ego's skill -> their "opp skill" slot
    return out
