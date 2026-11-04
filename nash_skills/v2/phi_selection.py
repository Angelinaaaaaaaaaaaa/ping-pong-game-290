"""
Phi checkpoint selection and dataset metadata for the Design B trainer.

Selection modes:
  train_loss    legacy: keep the epoch with the lowest sampled training loss.
                The loss is computed on a fresh random sample every epoch, so
                the minimum largely picks a lucky sample.
  val_residual  keep the epoch whose unilateral residuals |e| on a validation
                split (rallies carved out of the training set, disjoint from
                the held-out set used to report alpha) have the lowest
                statistic. The validation deviations and states are drawn
                from a fixed seed, so every epoch is scored on the same set.
"""
import numpy as np

__all__ = ["PHI_SELECT_MODES", "PHI_SELECT_STATS", "PhiValidator", "rally_metadata"]

PHI_SELECT_MODES = ("train_loss", "val_residual")
PHI_SELECT_STATS = ("p95", "max", "mean")

# What untagged rallies were collected with (the collector's defaults before
# these tags existed; every launch script used the aggressive profile).
_METADATA_DEFAULTS = {"defend_mechanism": "reactive", "decision_timing": "crossing",
                      "skill_profile": "aggressive"}


def rally_metadata(rallies) -> dict:
    """The collection settings shared by every rally; raises if rallies disagree."""
    out = {}
    for key, default in _METADATA_DEFAULTS.items():
        values = {r.get(key, default) for r in rallies}
        if len(values) > 1:
            raise ValueError(f"rallies mix several {key} values: {sorted(values)}; train on one setting")
        out[key] = values.pop() if values else default
    return out


class PhiValidator:
    """Scores a Phi model by a statistic of |e| over fixed validation deviations."""

    def __init__(self, x_val, constraints, theta_mode: str, seed: int, n_constraints: int,
                 batch_size: int, stat: str = "p95", p_vertex: float = 0.3):
        if stat not in PHI_SELECT_STATS:
            raise ValueError(f"Unknown selection stat '{stat}'. Choose from: {PHI_SELECT_STATS}")
        self.x_val, self.constraints, self.theta_mode = x_val, constraints, theta_mode
        self.seed, self.n_constraints, self.batch_size = seed, n_constraints, batch_size
        self.stat, self.p_vertex = stat, p_vertex

    def residuals(self, model_p, model1, model2) -> np.ndarray:
        from nash_skills.v2.train_q_model_5skill_v3_attack_defend import (
            theta_unilateral_residuals, unilateral_residuals,
        )
        rng = np.random.default_rng(self.seed)
        if self.theta_mode == "pure":
            return unilateral_residuals(model_p, model1, model2, self.x_val, self.constraints, rng,
                                        n_constraints=self.n_constraints, batch_size=self.batch_size)
        return theta_unilateral_residuals(model_p, model1, model2, self.x_val, rng,
                                          n_samples=self.n_constraints, batch_size=self.batch_size,
                                          p_vertex=self.p_vertex)

    def score(self, model_p, model1, model2) -> float:
        a = np.abs(self.residuals(model_p, model1, model2))
        if self.stat == "p95":
            return float(np.percentile(a, 95))
        return float(a.max() if self.stat == "max" else a.mean())
