"""
The paper's rally status ell_t (Sec. 3.2: "the last contact, bounce information, and terminal
outcome"), read from KukaTennisEnv's rally tracker (mujoco_env_comp.new_rally_state):

    last_hitter          "ego" (player 1), "opp" (player 2) or None before the first racket contact
    bounces_since_hit    table bounces since that contact
    receiver_had_chance  the receiver has had a legal chance to return (a bounce on its side)

The terminal outcome is not part of it: a decision state is never terminal. At contact-timed
decisions (right after a racket contact) the status is always (the player who just hit, 0, False).
"""

__all__ = ["rally_status"]


def rally_status(rally_state: dict) -> dict:
    last = rally_state.get("last_hitter")
    since = 0
    for bounce in reversed(rally_state.get("bounce_history", [])):
        if bounce.get("last_hitter") != last:
            break
        since += 1
    return {"last_hitter": last, "bounces_since_hit": since,
            "receiver_had_chance": bool(rally_state.get("receiver_had_return_opportunity", False))}
