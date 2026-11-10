"""
One self-play rally in Kexin's V3 game, following her v3/eval_q_policy.run_episode: rally start with shots
(left, right) and no pre-position, the low-level PPO controller moves both robots every step, and at each net
crossing the receiver (attacker) picks a shot and the other player (defender) a gantry pre-position, both from
the same pre-decision state. Both choices are applied the way her evaluator applies them. The learner's
decisions are recorded for PPO; the reward is +1 / -1 for the learner winning / losing the rally, 0 if the rally
is cut at max_steps (then the critic's value at the cut is the bootstrap).
"""
import numpy as np

from v3_selfplay.policy import ATTACKER, DEFENDER, N_POSITIONS, N_SHOTS, decide, policy_input

__all__ = ["play_rally"]


def _winner_player(obs, info, game):
    winner = str(info.get("winner") or game.infer_winner(obs, info))
    return 1 if winner == "ego" else 2 if winner == "opp" else None


def _apply(env, attacker, defender, shot, position):
    if attacker == 1:
        env.set_skills(shot, env._skill2)
    else:
        env.set_skills(env._skill1, shot)
    env.set_defense_position(defender, position)


def play_rally(env, ppo, policies, learner, game, max_steps, seed=None, decide_fn=decide):
    """policies: {1: psi_1, 2: psi_2}; game: a v3_selfplay.game.V3Game. Returns the learner's decisions and
    the rally outcome."""
    env.set_skills("left", "right")
    env.clear_defense_position()
    obs, info = env.reset() if seed is None else env.reset(seed=seed)
    previous_x = float(obs[36])
    rec = {"states": [], "roles": [], "actions": [], "old_log_probs": [], "old_values": []}
    usage = {"attack": np.zeros(N_SHOTS, int), "defend": np.zeros(N_POSITIONS, int)}
    steps, crossings, done, last = 0, 0, False, None
    while steps < max_steps and not done:
        obs, _reward, done, _truncated, info = game.step(env, game.build_ppo_action(ppo, obs, info))
        steps += 1
        current_x = float(obs[36])
        attacker = game.receiver_from_crossing(previous_x, current_x)
        previous_x = current_x
        if attacker is None:
            continue
        crossings += 1
        defender = 3 - attacker
        state = game.state_for_attacker(obs, info, attacker)
        last = (state, attacker)
        chosen = {}
        for player, role in ((attacker, ATTACKER), (defender, DEFENDER)):
            x = policy_input(state, role)
            action, log_prob, value = decide_fn(policies[player], x, role)
            chosen[role] = action
            if player == learner:
                rec = {"states": rec["states"] + [x], "roles": rec["roles"] + [role],
                       "actions": rec["actions"] + [action], "old_log_probs": rec["old_log_probs"] + [log_prob],
                       "old_values": rec["old_values"] + [value]}
                usage[{ATTACKER: "attack", DEFENDER: "defend"}[role]][action] += 1
        _apply(env, attacker, defender, game.shots[chosen[ATTACKER]], game.positions[chosen[DEFENDER]])

    winner = _winner_player(obs, info, game) if done else None
    reward = 0.0 if winner is None else (1.0 if winner == learner else -1.0)
    bootstrap = 0.0
    if not done and last is not None:
        state, attacker = last
        role = ATTACKER if learner == attacker else DEFENDER
        bootstrap = decide_fn(policies[learner], policy_input(state, role), role)[2]
    return {**rec, "reward": reward, "done": done, "winner": winner, "bootstrap_value": bootstrap,
            "steps": steps, "exchanges": crossings, "usage": usage}
