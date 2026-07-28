"""
Probabilistic skill selection for the Nash potential pipeline.

Provides three composable building blocks:

    softmax_probs(values, temperature)
        Convert potential values to a probability distribution.

    epsilon_mix_probs(base_probs, epsilon, num_skills)
        Mix any distribution with the uniform distribution.

    select_skill_from_values(values, mode, temperature, epsilon, rng)
        End-to-end: potential values → chosen skill index.

Supported modes
---------------
'argmax'          Deterministic argmax — preserves existing eval behavior.
'softmax'         Sample from softmax(values / temperature).
'epsilon_argmax'  Argmax with ε-greedy uniform exploration.
'epsilon_softmax' Softmax with ε-uniform mixing.

Default is 'argmax' so callers that omit the argument are unaffected.
"""

import numpy as np
from typing import Optional

__all__ = ["softmax_probs", "epsilon_mix_probs", "select_skill_from_values"]

_VALID_MODES = ("argmax", "softmax", "epsilon_argmax", "epsilon_softmax")


def softmax_probs(values, temperature: float = 1.0) -> np.ndarray:
    """
    Convert an array of potential values to a probability distribution.

    Parameters
    ----------
    values      : array-like of shape (N,)
    temperature : positive float; lower → sharper (closer to argmax),
                  higher → flatter (closer to uniform)

    Returns
    -------
    probs : np.ndarray of shape (N,), dtype float64, sums to 1.0
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    v = v / temperature
    v = v - v.max()  # shift for numerical stability
    exp = np.exp(v)
    return exp / exp.sum()


def epsilon_mix_probs(
    base_probs: np.ndarray,
    epsilon: float,
    num_skills: int,
) -> np.ndarray:
    """
    Mix `base_probs` with the uniform distribution.

    result = (1 - epsilon) * base_probs + epsilon * uniform

    Parameters
    ----------
    base_probs  : array-like of shape (num_skills,), should sum to 1
    epsilon     : float in [0, 1]; 0 → pure base_probs, 1 → pure uniform
    num_skills  : number of skills (uniform denominator)

    Returns
    -------
    mixed : np.ndarray of shape (num_skills,), dtype float64
    """
    if not (0.0 <= epsilon <= 1.0):
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
    base = np.asarray(base_probs, dtype=np.float64).reshape(-1)
    uniform = np.ones(num_skills, dtype=np.float64) / num_skills
    return (1.0 - epsilon) * base + epsilon * uniform


def select_skill_from_values(
    values,
    mode: str = "argmax",
    temperature: float = 1.0,
    epsilon: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """
    Select a skill index from potential/Q values.

    Parameters
    ----------
    values      : array-like of shape (N,) — potential or Q values
    mode        : one of 'argmax', 'softmax', 'epsilon_argmax', 'epsilon_softmax'
    temperature : softmax temperature (used by 'softmax' and 'epsilon_softmax')
    epsilon     : exploration rate in [0, 1] (used by 'epsilon_argmax' and
                  'epsilon_softmax')
    rng         : numpy Generator for reproducibility; if None, a fresh default
                  Generator is created (non-reproducible)

    Returns
    -------
    skill_idx : int in [0, N)
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Unknown mode {mode!r}. "
            f"Expected one of: {', '.join(_VALID_MODES)}"
        )

    v = np.asarray(values, dtype=np.float64).reshape(-1)
    n = len(v)

    if rng is None:
        rng = np.random.default_rng()

    if mode == "argmax":
        return int(np.argmax(v))

    if mode == "softmax":
        probs = softmax_probs(v, temperature)
        return int(rng.choice(n, p=probs))

    if mode == "epsilon_argmax":
        if rng.random() < epsilon:
            return int(rng.integers(n))
        return int(np.argmax(v))

    # epsilon_softmax
    probs = softmax_probs(v, temperature)
    probs = epsilon_mix_probs(probs, epsilon, n)
    return int(rng.choice(n, p=probs))
