"""v3_selfplay.rollout: one self-play rally in Kexin's V3 game, decisions at net crossings as in her run_episode."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from v3_selfplay import policy as sp
from v3_selfplay.game import V3Game
from v3_selfplay.rollout import play_rally

SHOTS = ("left", "left_short", "center_safe", "right_short", "right")
POSITIONS = ("left", "center", "right")


class FakeEnv:
    """Ball x follows a script (net at x = 1.5); the rally ends after the script with `winner`."""

    def __init__(self, xs, winner="ego"):
        self.xs, self.winner, self.calls = list(xs), winner, []
        self._skill1, self._skill2 = "left", "right"

    def set_skills(self, s1, s2):
        self._skill1, self._skill2 = s1, s2
        self.calls.append(("skills", s1, s2))

    def clear_defense_position(self):
        self.calls.append(("clear",))

    def set_defense_position(self, defender, position):
        self.calls.append(("defense", defender, position))

    def reset(self, seed=None):
        self.calls.append(("reset", seed))
        self.t = 0
        return self._obs(), {}

    def _obs(self):
        obs = np.zeros(80, dtype=np.float32)
        obs[36] = self.xs[min(self.t, len(self.xs) - 1)]
        return obs

    def advance(self):
        self.t += 1
        done = self.t >= len(self.xs) - 1
        return self._obs(), 0.0, done, False, ({"winner": self.winner} if done else {})


def _game():
    def receiver(prev, cur):
        if (prev - 1.5) * (cur - 1.5) >= 0:
            return None
        return 2 if cur > 1.5 else 1

    return V3Game(build_ppo_action=lambda ppo, obs, info: np.zeros(18), step=lambda env, action: env.advance(),
                  receiver_from_crossing=receiver,
                  state_for_attacker=lambda obs, info, attacker: np.full(74, float(attacker), dtype=np.float32),
                  infer_winner=lambda obs, info: "opp", shots=SHOTS, positions=POSITIONS)


def _deciding(log, shot=4, position=0, value=0.5):
    """decide stub: player tag -> fixed shot / position; logs (tag, role, input)."""
    def decide(policy, x, role):
        log.append((policy, role, x.copy()))
        return (shot if role == sp.ATTACKER else position), -0.1, value
    return decide


# x: 1.0 -> 2.0 (P2 receives) -> 1.0 (P1 receives) -> 2.0 (P2 receives) -> rally ends
XS = [1.0, 1.2, 2.0, 1.0, 2.0, 2.0]


def test_decisions_happen_at_net_crossings_with_the_receiver_attacking():
    log = []
    out = play_rally(FakeEnv(XS), None, {1: "p1", 2: "p2"}, learner=1, game=_game(), max_steps=100,
                     decide_fn=_deciding(log))
    assert out["exchanges"] == 3
    assert [(tag, role) for tag, role, _x in log] == [
        ("p2", "attacker"), ("p1", "defender"), ("p1", "attacker"), ("p2", "defender"),
        ("p2", "attacker"), ("p1", "defender")]
    assert out["roles"] == ["defender", "attacker", "defender"]             # only the learner's decisions


def test_both_players_see_her_attacker_centred_state_with_their_role_flag():
    log = []
    play_rally(FakeEnv(XS), None, {1: "p1", 2: "p2"}, learner=2, game=_game(), max_steps=100,
               decide_fn=_deciding(log))
    _tag, _role, x_att = log[0]
    _tag, _role, x_def = log[1]
    assert np.all(x_att[:74] == 2.0) and np.all(x_def[:74] == 2.0)       # P2 attacks first: P2-centred state
    assert x_att[74] == 1.0 and x_def[74] == 0.0


def test_choices_reach_the_env_like_her_evaluator_sets_them():
    env = FakeEnv(XS)
    play_rally(env, None, {1: "p1", 2: "p2"}, learner=1, game=_game(), max_steps=100,
               decide_fn=_deciding([], shot=1, position=2))
    assert env.calls[:3] == [("skills", "left", "right"), ("clear",), ("reset", None)]
    assert ("skills", "left", "left_short") in env.calls                      # P2 attacks: its own slot
    assert ("defense", 1, "right") in env.calls                               # P1 defends
    assert ("skills", "left_short", "left_short") in env.calls                # then P1 attacks


@pytest.mark.parametrize("winner, learner, reward", [("ego", 1, 1.0), ("ego", 2, -1.0), ("opp", 2, 1.0)])
def test_reward_is_the_rally_outcome_for_the_learner(winner, learner, reward):
    out = play_rally(FakeEnv(XS, winner=winner), None, {1: "p1", 2: "p2"}, learner=learner, game=_game(),
                     max_steps=100, decide_fn=_deciding([]))
    assert out["done"] and out["reward"] == reward and out["bootstrap_value"] == 0.0
    assert out["winner"] == (1 if winner == "ego" else 2)


def test_a_rally_cut_by_the_step_cap_scores_nothing_and_bootstraps_the_critic():
    out = play_rally(FakeEnv(XS), None, {1: "p1", 2: "p2"}, learner=1, game=_game(), max_steps=3,
                     decide_fn=_deciding([], value=0.25))
    assert not out["done"] and out["reward"] == 0.0 and out["winner"] is None
    assert out["bootstrap_value"] == 0.25 and out["steps"] == 3


def test_records_line_up_for_ppo():
    out = play_rally(FakeEnv(XS), None, {1: "p1", 2: "p2"}, learner=1, game=_game(), max_steps=100,
                     decide_fn=_deciding([], shot=4, position=0))
    n = len(out["actions"])
    assert n == len(out["states"]) == len(out["old_log_probs"]) == len(out["old_values"]) == 3
    assert out["actions"] == [0, 4, 0]
    assert out["usage"]["attack"].tolist() == [0, 0, 0, 0, 1] and out["usage"]["defend"].tolist() == [2, 0, 0]
