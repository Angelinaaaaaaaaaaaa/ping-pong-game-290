"""
PPO for the self-play actor-critic (v3_selfplay.policy): GAE over each player's decisions in a rally, the
clipped surrogate on the head that acted, value regression and an entropy bonus. Same algorithm and defaults
as our Design B self-play (nash_skills/v2/selfplay_attack_defend.py, selfplay_per_player.py); copied so the
package does not depend on our nash_skills, which Kexin's checkout predates.
"""
import numpy as np
import torch
import torch.nn.functional as F

from v3_selfplay.policy import ATTACKER

__all__ = ["compute_gae", "ppo_clipped_loss", "role_log_prob_entropy", "build_batch", "ppo_update"]

MAX_GRAD_NORM = 1.0


def compute_gae(rewards, values, gamma, gae_lambda, bootstrap_value=0.0):
    """(advantages, returns) for one trajectory; bootstrap_value stands in for what follows a cut rally."""
    advantages, gae = [0.0] * len(rewards), 0.0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(rewards) else bootstrap_value
        gae = rewards[t] + gamma * next_value - values[t] + gamma * gae_lambda * gae
        advantages[t] = gae
    return advantages, [a + v for a, v in zip(advantages, values)]


def ppo_clipped_loss(new_log_probs, old_log_probs, advantages, clip_epsilon):
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    return -torch.min(ratio * advantages, clipped).mean()


def role_log_prob_entropy(shot_logits, position_logits, actions, is_attacker):
    """Log-prob and entropy of the head that acted at each decision."""
    shots = torch.distributions.Categorical(logits=shot_logits)
    positions = torch.distributions.Categorical(logits=position_logits)
    shot_lp = shots.log_prob(actions.clamp(max=shot_logits.shape[-1] - 1))
    position_lp = positions.log_prob(actions.clamp(max=position_logits.shape[-1] - 1))
    log_prob = torch.where(is_attacker, shot_lp, position_lp)
    entropy = torch.where(is_attacker, shots.entropy(), positions.entropy())
    return log_prob, entropy


def build_batch(rallies, gamma, gae_lambda):
    """The learner's decisions of all rallies with their advantages and returns; the terminal reward sits on
    each rally's last decision."""
    batch = {k: [] for k in ("states", "roles", "actions", "old_log_probs", "advantages", "returns")}
    for rally in rallies:
        n = len(rally["old_values"])
        if n == 0:
            continue
        rewards = [0.0] * (n - 1) + [rally["reward"]]
        adv, ret = compute_gae(rewards, rally["old_values"], gamma, gae_lambda, rally["bootstrap_value"])
        for key in ("states", "roles", "actions", "old_log_probs"):
            batch[key] = batch[key] + list(rally[key])
        batch["advantages"] = batch["advantages"] + adv
        batch["returns"] = batch["returns"] + ret
    return batch


def ppo_update(policy, optim, batch, cfg):
    """cfg: ppo_epochs, minibatch_size, clip_epsilon, value_coef, entropy_coef. Returns mean loss stats."""
    n = len(batch["advantages"])
    if n == 0:
        return {"policy_loss": float("nan"), "value_loss": float("nan"), "entropy": float("nan")}
    states = torch.as_tensor(np.stack(batch["states"]), dtype=torch.float32)
    actions = torch.tensor(batch["actions"], dtype=torch.long)
    is_attacker = torch.tensor([role == ATTACKER for role in batch["roles"]])
    old_lp = torch.tensor(batch["old_log_probs"], dtype=torch.float32)
    returns = torch.tensor(batch["returns"], dtype=torch.float32)
    adv = torch.tensor(batch["advantages"], dtype=torch.float32)
    if n > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = {"policy_loss": [], "value_loss": [], "entropy": []}
    for _ in range(cfg.ppo_epochs):
        perm = torch.randperm(n)
        for start in range(0, n, cfg.minibatch_size):
            idx = perm[start:start + cfg.minibatch_size]
            shot_logits, position_logits, values = policy(states[idx])
            lp, ent = role_log_prob_entropy(shot_logits, position_logits, actions[idx], is_attacker[idx])
            policy_loss = ppo_clipped_loss(lp, old_lp[idx], adv[idx], cfg.clip_epsilon)
            value_loss = F.mse_loss(values, returns[idx])
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * ent.mean()
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optim.step()
            for key, value in (("policy_loss", policy_loss), ("value_loss", value_loss), ("entropy", ent.mean())):
                stats[key].append(float(value.item()))
    return {k: float(np.mean(v)) for k, v in stats.items()}
