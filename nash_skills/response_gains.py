"""
Unilateral response gains (paper Sec. 5.1, Eq. 15), estimated with exact simulated continuations.

For a frozen profile pi = (pi^1, pi^2) -- a method playing both seats -- decision points x are
sampled from its own rallies (the simulator state right after an exchange starts, before either
player has chosen). For player i, the rest of the rally is replayed from x with player i following
each candidate response and player -i following the method:

    R_i(x; pi) = max_{candidate} V^i(x; candidate, pi^-i) - V^i(x; pi),

where the candidates are the method itself and the 15 fixed (target, preparation) strategies, and
V^i is the discounted point outcome (+-gamma^k after k more exchanges, 0 if unresolved). Because
the original policy is a candidate the gain is >= 0, and a finite candidate set makes it a lower
bound on the true regret. Physics is deterministic, so deterministic methods need one rollout per
candidate; stochastic methods (sampled self-play, Nash-Q) average `repeats` rollouts.

Method pickers are created fresh for every rollout, so stateful rules (e.g. Potential Adaptive
Match's opponent counts) restart their memory at the decision point.
"""
import dataclasses

import numpy as np

from nash_skills.decision_clock import DecisionClock, validate_decision_timing

DEFAULT_GAIN_TIMING = "contact"
from nash_skills.defend_positions import DEFEND_POSITION_NAMES, defend_position_index
from nash_skills.ibr_sim import apply_joint_pair
from nash_skills.sim_clone import EnvSnapshot, restore_env, snapshot_env
from nash_skills.skills import SKILL_NAMES, skill_index
from nash_skills.v2.joint_action import joint_action_index
from nash_skills.winner_inference import infer_terminal_winner

__all__ = ["STATIC_RESPONSES", "DecisionPoint", "gain_from_values", "policy_level_gain",
           "collect_decision_points", "ResponseGainEvaluator"]

STATIC_RESPONSES = [joint_action_index(skill_index(a), defend_position_index(d))
                    for a in SKILL_NAMES for d in DEFEND_POSITION_NAMES]


@dataclasses.dataclass
class DecisionPoint:
    snap: EnvSnapshot
    attacker: int
    joints: tuple
    decision_index: int


def gain_from_values(v_original, candidate_values):
    """Gain over the original policy; the original is itself a candidate, so the gain is >= 0."""
    best = max(candidate_values, key=candidate_values.get) if candidate_values else None
    if best is None or candidate_values[best] <= v_original:
        return {"gain": 0.0, "best_response": None, "original": v_original}
    return {"gain": float(candidate_values[best] - v_original), "best_response": int(best), "original": v_original}


def policy_level_gain(per_point_values, rng=None, n_boot=2000):
    """Gain of the best response *policy* over the evaluation states: each candidate is scored by its
    mean value over all points (common points for every candidate), then the best mean is compared
    with the original's mean. per_point_values: list of {candidate: value} dicts with key "orig".

    Taking the expectation before the max keeps the estimate from rewarding a different candidate at
    each state: with a deterministic simulator, some fixed strategy happens to win almost any single
    exact continuation, so a per-state max measures chaos rather than exploitability."""
    cands = list(per_point_values[0].keys())
    table = np.array([[v[c] for c in cands] for v in per_point_values], dtype=float)
    orig = cands.index("orig")

    def gain(rows):
        means = table[rows].mean(axis=0)
        return float(means.max() - means[orig]), cands[int(means.argmax())]

    g, best = gain(np.arange(len(table)))
    rng = rng or np.random.default_rng(0)
    boots = [gain(rng.integers(0, len(table), len(table)))[0] for _ in range(n_boot)] if len(table) > 1 else [g]
    return {"gain": g, "best_response": None if best == "orig" else int(best),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "original_mean": float(table[:, orig].mean())}


def _ppo_step(env, ppo, obs, info):
    from nash_skills.v2.selfplay_rollout import _ppo_step as step
    return step(env, ppo, obs, info)


def _decide(pickers, obs, info, env, attacker, joints, n_dec, mechanism):
    pi = {**info, "current_joints": joints, "attacker": attacker, "env": env, "decision_index": n_dec}
    new = (pickers[1](1, obs, joints[1], pi), pickers[2](2, obs, joints[0], pi))
    apply_joint_pair(env, mechanism, new[0], new[1])
    return new


def collect_decision_points(env, ppo, mechanism, make_method_picker, n_points, rng, max_steps, take_prob=0.5,
                            decision_timing=DEFAULT_GAIN_TIMING):
    """Play the method against itself and snapshot the state at sampled decision points."""
    validate_decision_timing(decision_timing)
    points = []
    while len(points) < n_points:
        pickers = {1: make_method_picker(), 2: make_method_picker()}
        apply_joint_pair(env, mechanism, 0, 0)
        obs, info = env.reset()
        clock = DecisionClock(decision_timing)
        clock.reset()
        joints, n_dec = (0, 0), 0
        for _ in range(max_steps):
            obs, _r, done, _t, info = _ppo_step(env, ppo, obs, info)
            if done:
                break
            attacker = clock.update(env)
            if attacker is None:
                continue
            if rng.random() < take_prob and len(points) < n_points:
                points.append(DecisionPoint(snapshot_env(env, obs, info), attacker, joints, n_dec))
            joints = _decide(pickers, obs, info, env, attacker, joints, n_dec, mechanism)
            n_dec += 1
    return points


class ResponseGainEvaluator:
    def __init__(self, sim_env, ppo, mechanism, gamma, max_steps, make_method_picker, repeats=1,
                 decision_timing=DEFAULT_GAIN_TIMING):
        self.timing = validate_decision_timing(decision_timing)
        self.sim, self.ppo, self.mechanism = sim_env, ppo, mechanism
        self.gamma, self.max_steps, self.repeats = gamma, max_steps, repeats
        self.make_method_picker = make_method_picker

    def _rollout(self, point, player, response):
        obs, info = restore_env(self.sim, point.snap)
        fixed = (lambda _p, _o, _x, _i=None: response) if response is not None else None
        pickers = {p: self.make_method_picker() for p in (1, 2)}
        if fixed is not None:
            pickers[player] = fixed
        clock = DecisionClock(self.timing)
        clock.update(self.sim)  # latch the exchange that is starting
        joints = _decide(pickers, obs, info, self.sim, point.attacker, point.joints, point.decision_index,
                         self.mechanism)
        n_dec, k = point.decision_index + 1, 0
        for _ in range(self.max_steps):
            obs, _r, done, _t, info = _ppo_step(self.sim, self.ppo, obs, info)
            if done:
                won = infer_terminal_winner(obs, info, fallback="position") == ("ego" if player == 1 else "opp")
                return (1.0 if won else -1.0) * self.gamma ** k
            attacker = clock.update(self.sim)
            if attacker is not None:
                k += 1
                joints = _decide(pickers, obs, info, self.sim, attacker, joints, n_dec, self.mechanism)
                n_dec += 1
        return 0.0

    def value(self, point, player, response=None):
        """Mean discounted outcome for `player` following `response` (None = the method itself)."""
        return float(np.mean([self._rollout(point, player, response) for _ in range(self.repeats)]))

    def candidate_values(self, point, player):
        """{"orig": method's value, joint index: value} for player at the point."""
        out = {"orig": self.value(point, player, None)}
        out.update({c: self.value(point, player, c) for c in STATIC_RESPONSES})
        return out

    def gains(self, point, player):
        values = self.candidate_values(point, player)
        v_original = values.pop("orig")
        return {**gain_from_values(v_original, values), "values": values}
