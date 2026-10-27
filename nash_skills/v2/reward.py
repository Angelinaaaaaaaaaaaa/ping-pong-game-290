"""
Configurable rally-level reward for the self-play trainers.

One definition shared by all four self-play variants, so the reward can't
drift between files again. The parameters span every design the project has
used, which makes them directly comparable in an ablation:

    pure zero-sum (current default)
        shaping_coef=0.0, trunc_shaping_coef=0.0, trunc_penalty=-0.5
        Terminal +/-1 sums to zero. Truncated rallies pay a symmetric penalty,
        which is *not* zero-sum (sums to 2*trunc_penalty) but discourages
        stalling.

    strictly zero-sum on every episode
        shaping_coef=0.0, trunc_shaping_coef=0.0, trunc_penalty=0.0
        Matches `labeling.py`'s convention for the offline Q/Phi pipeline,
        which scores truncated rallies 0.0 and enforces G2 = -G1.

    crossing shaping (the pre-2026-07 design)
        shaping_coef=0.005, trunc_shaping_coef=0.001, trunc_penalty=0.0
        A per-crossing bonus paid to BOTH players. Note this breaks zero-sum
        on every rally (sums to 2*shaped) and rewards keeping the rally alive
        -- the older 0.05 setting let a 50-crossing rally out-earn a win.

Shaping is a *common* term by construction: both players receive the same
bonus. That is what the original design did, and reproducing it faithfully is
the point of keeping it configurable.
"""

__all__ = ["rally_rewards"]


def rally_rewards(
    done: bool,
    ego_won: bool,
    crossings: int,
    *,
    trunc_penalty: float = -0.0,
    shaping_coef: float = 0.0,
    trunc_shaping_coef: float = 0.0,
):
    """
    Compute (ego_reward, opp_reward) for one rally.

    Parameters
    ----------
    done               : whether the rally reached a decisive terminal state
    ego_won            : winner was ego (ignored when `done` is False)
    crossings          : number of net crossings, used only by the shaping terms
    trunc_penalty      : reward given to BOTH players when the rally is truncated
    shaping_coef       : per-crossing bonus to BOTH players on decisive rallies
    trunc_shaping_coef : per-crossing bonus to BOTH players on truncated rallies

    Returns
    -------
    (ego_reward, opp_reward) : tuple of float
    """
    if done:
        ego_terminal = 1.0 if ego_won else -1.0
        shaped = shaping_coef * crossings
        return ego_terminal + shaped, -ego_terminal + shaped

    shaped = trunc_shaping_coef * crossings
    return trunc_penalty + shaped, trunc_penalty + shaped
