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
    out[10]    = obs[-2]       # ego skill index (normalised)
    out[11]    = obs[-1]       # opp skill index (normalised)
    return out


def encode_opp_gantry(obs: np.ndarray) -> np.ndarray:
    """
    Encode raw 116-dim env obs into a 12-dim opp-perspective state.

    From opp's perspective, swap ego/opp gantry AND swap the skill slots to
    match: slot [10] is always "this perspective's own skill", slot [11] is
    always "the other player's skill". obs[-2]/obs[-1] are stored in the
    ego-perspective convention (ego skill, opp skill), so the opp-perspective
    encoder must swap them too -- ported from main (nash_skills/v2/
    state_encoder_gantry.py), where a diagnostic run showed the un-swapped
    version fed the opp's own skill into the "opp" slot from its own point
    of view, silently corrupting every opp-side Phi(s, ego, opp) lookup.
    """
    out = np.zeros(STATE_DIM, dtype=np.float32)
    out[0:2]   = obs[18:20]    # opp gantry (as their "ego")
    out[2:4]   = obs[0:2]      # ego gantry (as their "opp")
    out[4:7]   = obs[36:39]    # ball position
    out[7:10]  = obs[39:42]    # ball velocity
    out[10]    = obs[-1]       # opp's own skill -> their "ego" slot
    out[11]    = obs[-2]       # ego's skill (as seen from opp) -> their "opp" slot
    return out
