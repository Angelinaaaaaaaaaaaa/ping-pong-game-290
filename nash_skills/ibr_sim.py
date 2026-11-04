"""
Simulation-based iterated best response (paper Sec. 5.1).

At each of its decisions the IBR player alternates best responses -- its own,
then the opponent's, then its own, ... (2*rounds - 1 updates, ending with its
own) -- starting from both players at `init`. Each response is searched over
the 15 pure (attack target, preparation position) pairs, i.e. the vertices of
the compact family Delta_5 x Delta_3, and scored by a simulated continuation
from the current state: both players hold their candidate pair fixed for
H_IBR exchanges (or until the point ends), the low-level controller runs as
usual, and the return is +-gamma^k for a point won/lost after k exchanges,
0 if unresolved. The IBR player then plays its last response; the opponent
it modelled is only its belief, the real opponent acts on its own.

Stepping is deterministic, so with both players fixed one rollout per pair
is exact; rollouts are cached per pair within a decision, and one rollout
scores the pair for both players (zero-sum). Searching vertices only is a
local response search: finite rounds need not reach an equilibrium.
"""
import time

import numpy as np

from nash_skills.decision_clock import DecisionClock
from nash_skills.defend_mechanisms import set_defend_choices
from nash_skills.defend_positions import defend_position_from_index, defend_position_index
from nash_skills.sim_clone import restore_env, snapshot_env
from nash_skills.skills import skill_from_index, skill_index
from nash_skills.v2.joint_action import N_JOINT_ACTIONS, joint_action_from_index, joint_action_index
from nash_skills.winner_inference import infer_terminal_winner

__all__ = ["IBR_DEFAULTS", "iterated_best_response", "SimRolloutEvaluator", "apply_joint_pair",
           "make_ibr_sim_picker"]

IBR_DEFAULTS = {"rounds": 2, "horizon_exchanges": 4, "max_steps": 1500, "gamma": 0.99,
                "init": ("center_safe", "center")}
_PICK_INFO_KEYS = ("env", "current_joints", "attacker")


def iterated_best_response(player, evaluate, n_actions, rounds, init, rng):
    """evaluate(p, own, other) -> p's return. Returns (planner's response, trace)."""
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    strategy = {1: init, 2: init}
    trace = []
    for k in range(2 * rounds - 1):
        p = player if k % 2 == 0 else 3 - player
        values = np.array([evaluate(p, c, strategy[3 - p]) for c in range(n_actions)])
        ties = np.flatnonzero(values >= values.max() - 1e-12)
        strategy[p] = int(rng.choice(ties))
        trace.append({"player": p, "best": strategy[p], "values": values.tolist()})
    return strategy[player], trace


def apply_joint_pair(env, mechanism, joint1, joint2):
    (a1, d1), (a2, d2) = joint_action_from_index(joint1), joint_action_from_index(joint2)
    env.set_skills(skill_from_index(a1), skill_from_index(a2))
    set_defend_choices(env, mechanism, defend_position_from_index(d1), defend_position_from_index(d2))


class SimRolloutEvaluator:
    """Scores fixed joint pairs by simulated continuation from a bound state."""

    def __init__(self, sim_env, ppo, defend_mechanism, horizon_exchanges, max_steps, gamma):
        self.sim, self.ppo, self.mechanism = sim_env, ppo, defend_mechanism
        self.horizon, self.max_steps, self.gamma = horizon_exchanges, max_steps, gamma
        self.rollouts, self._snap, self._cache = 0, None, {}

    def bind(self, real_env, obs, info):
        clean = {k: v for k, v in info.items() if k not in _PICK_INFO_KEYS}
        self._snap, self._cache = snapshot_env(real_env, obs, clean), {}

    def __call__(self, p, own, other):
        pair = (own, other) if p == 1 else (other, own)
        if pair not in self._cache:
            self._cache[pair] = self._rollout(*pair)
        v1 = self._cache[pair]
        return v1 if p == 1 else -v1

    def _rollout(self, joint1, joint2):
        from nash_skills.v2.selfplay_rollout import _ppo_step
        self.rollouts += 1
        obs, info = restore_env(self.sim, self._snap)
        apply_joint_pair(self.sim, self.mechanism, joint1, joint2)
        clock = DecisionClock("contact")
        clock.update(self.sim)  # latch the exchange already in progress
        exchanges = 0
        for _ in range(self.max_steps):
            obs, _r, done, _t, info = _ppo_step(self.sim, self.ppo, obs, info)
            if done:
                won = infer_terminal_winner(obs, info, fallback="position") == "ego"
                return (1.0 if won else -1.0) * self.gamma ** exchanges
            if clock.update(self.sim) is not None:
                exchanges += 1
                if exchanges >= self.horizon:
                    break
        return 0.0


def make_ibr_sim_picker(ppo, skill_profile, defend_mechanism, seed=0, **overrides):
    """Picker for run_matchup_joint: needs info["env"] (the live env) at each call."""
    cfg = {**IBR_DEFAULTS, **overrides}
    init = joint_action_index(skill_index(cfg["init"][0]), defend_position_index(cfg["init"][1]))
    rng = np.random.default_rng(seed)
    state = {"evaluator": None}
    stats = {"decisions": 0, "rollouts": 0, "seconds": 0.0, "config": {k: cfg[k] for k in cfg}}

    def pick_ibr_sim(player, obs, _other_joint_idx, info=None):
        if info is None or "env" not in info:
            raise ValueError("ibr-sim needs the live env in info['env'] (run_matchup_joint supplies it)")
        if state["evaluator"] is None:
            from nash_skills.env_wrapper import SkillEnv
            sim = SkillEnv(proc_id=1, history=4, reset_mode="ready", skill_profile=skill_profile,
                           reset_targets=True)
            state["evaluator"] = SimRolloutEvaluator(sim, ppo, defend_mechanism, cfg["horizon_exchanges"],
                                                     cfg["max_steps"], cfg["gamma"])
        ev = state["evaluator"]
        t0, before = time.time(), ev.rollouts
        ev.bind(info["env"], obs, info)
        joint, _trace = iterated_best_response(player, ev, N_JOINT_ACTIONS, cfg["rounds"], init, rng)
        stats["decisions"] += 1
        stats["rollouts"] += ev.rollouts - before
        stats["seconds"] += time.time() - t0
        return joint

    pick_ibr_sim.stats = stats
    return pick_ibr_sim
