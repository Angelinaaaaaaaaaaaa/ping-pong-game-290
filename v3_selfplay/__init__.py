"""
Self-play RL in Kexin's V3 game (her v3/ package), so it can be compared with her alpha-potential players on her
own protocol (v3/eval_policy_fixed_grid.py). Her game: at each net crossing the receiver is the attacker and picks
one of 5 shots, the other player is the defender and moves its gantry to one of 3 pre-positions; rallies are cut
at 600 steps.

The method is our Design B self-play (nash_skills/v2/selfplay_per_player.py: one actor-critic per player,
role-gated heads, PPO, alternating update rounds with a snapshot pool); only the game and the state change. The
state is her 74-dim attacker-centred encoding (v3.state_encoder) plus a role flag, the same input her
alpha-potential sees. The package is self-contained apart from her v3 package, so it can run inside a snapshot
of her checkout (her simulator files included).
"""
