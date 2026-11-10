"""
The pieces of Kexin's V3 game a self-play rally needs, taken from her v3 package (the functions her
v3/eval_q_policy.run_episode uses), plus her environment with her evaluation settings. Tests pass a fake V3Game.
"""
from dataclasses import dataclass
from typing import Callable

__all__ = ["V3Game", "her_game", "make_her_env", "HER_MAX_STEPS"]

HER_MAX_STEPS = 600          # her --steps in every V3 command (v3/README.md, eval_policy_fixed_grid.py)


@dataclass(frozen=True)
class V3Game:
    build_ppo_action: Callable        # (ppo, obs, info) -> 18-dim low-level action
    step: Callable                    # (env, action) -> (obs, reward, done, truncated, info)
    receiver_from_crossing: Callable  # (previous ball x, current ball x) -> receiving player or None
    state_for_attacker: Callable      # (obs, info, attacker) -> her 74-dim attacker-centred state
    infer_winner: Callable            # (obs, info) -> "ego" (P1) or "opp" (P2)
    shots: tuple                      # her shot names, in index order
    positions: tuple                  # her pre-position names, in index order


def her_game() -> V3Game:
    from nash_skills.skills import SKILL_NAMES
    from v3.constants import DEFENSE_POSITIONS
    from v3.eval_fixed_preposition import build_ppo_action, receiver_from_crossing
    from v3.eval_q_policy import _capture_env_step, _infer_winner
    from v3.state_encoder import state_for_attacker

    def step(env, action):
        transition, _lines = _capture_env_step(env, action)
        return transition

    return V3Game(build_ppo_action=build_ppo_action, step=step, receiver_from_crossing=receiver_from_crossing,
                  state_for_attacker=state_for_attacker, infer_winner=_infer_winner,
                  shots=tuple(SKILL_NAMES), positions=tuple(DEFENSE_POSITIONS))


def make_her_env(skill_profile="aggressive", reset_mode="ready", gantry_speed_scale=1.0, planner="legacy"):
    """Her environment as v3/eval_policy_fixed_grid.py builds it (defaults = her evaluation defaults)."""
    from v3.env_wrapper import V3PrePositionEnv
    from v3.eval_fixed_preposition import HISTORY
    return V3PrePositionEnv(proc_id=1, history=HISTORY, reset_mode=reset_mode, skill_profile=skill_profile,
                            gantry_speed_scale=gantry_speed_scale, planner=planner)
