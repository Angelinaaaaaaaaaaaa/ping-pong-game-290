"""
Design B decision rules: generalizes eval_matchup.py's Phi-based and
IBR/Q-based picker strategies from a 5-way single-skill action space to
the 15-way joint (attack,defend) action space
(nash_skills.v2.joint_action.N_JOINT_ACTIONS).

Legacy eval_matchup.py is completely untouched -- this is a parallel
module, not a modification, consistent with every other Design-B
versioned-pipeline decision this session.

Direct translations of eval_matchup.py's existing rules (read in full
this session before writing this module) onto a 15x15 table instead of
5x5:
  nash-p-hard:     row/col max   (phi.max(dim=1) / phi.max(dim=0))
  nash-p-br:       conditional best response, opponent's CURRENT joint
                   action fixed  (phi[:, other] / phi[other, :])
  nash-p-minimax:  row/col min, then dispatch picks the argmax row/col
                   of those minima (maximin)
  nash-p-adaptive: expected value against an empirical opponent-joint-
                   action distribution  (phi @ probs / probs @ phi)
  ibr:             Q-based alternating best response, IBR_STEPS=4 rounds
  ibr-q:           Q-based empirical-mix best response
  nash-p-pga:      NEW -- the paper's own online phase (Eq. 18):
                   projected gradient ascent on Phi(x, theta) over the
                   compact mixed-strategy family theta in Delta_5 x
                   Delta_3 per player, warm-started from the previous
                   solution plus random restarts (local-optima check);
                   the player then SAMPLES its own (attack, defend) from
                   its component of the solution. Only meaningful with a
                   Phi trained on mixed-theta data (collector --sampling
                   mixed, trainer --theta-mode mixed): on a pure-trained
                   Phi, interior theta is extrapolation, and the
                   multilinear mixed extension of a pure table has its
                   max at a vertex anyway (see nash_skills.v2.mixed_strategy).

Also supports a FIXED strategy: pass a (attack_name, defend_name) tuple
to always play that one joint action -- these are Table 1's 15 static
baselines (5 attack targets x 3 preparation positions).

NOT reused from eval_matchup.py: run_matchup and make_picker are
hardcoded to env.set_skills only (no defend-position calls at all) and
use a cruder raw ball_x sign-change crossing check rather than the
role-tracker crossing detection Design B needs. run_matchup_joint
(below) is the two-axis replacement, modeled directly on
collect_data_attack_defend.py's loop (which already drives both
set_skills and set_defend_reactive_positions correctly via
env._role_tracker transitions) rather than on run_matchup's warmup/
total-step-budget scheme -- living alongside make_picker_joint here,
mirroring how run_matchup lives alongside make_picker in eval_matchup.py.

Reused from eval_matchup.py (genuinely action-space-agnostic):
  _pick_with_softmax_fallback -- generic over action_scores length.
  _capture_env_step / _parse_contact_lines -- per-step contact/return-
      success parsing, independent of action space.
"""
import dataclasses
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from nash_skills.skills import SKILL_NAMES, skill_index, skill_from_index
from nash_skills.defend_positions import (
    DEFEND_POSITION_NAMES, defend_position_index, defend_position_from_index,
)
from nash_skills.v2.joint_action import (
    N_JOINT_ACTIONS, joint_action_index, joint_action_from_index, write_joint_pair_by_index,
    write_mixed_strategy_pair,
)
from nash_skills.v2.mixed_strategy import pga_maximize, sample_action_from_theta
from nash_skills.defend_mechanisms import DEFAULT_DEFEND_MECHANISM, set_defend_choices, validate_defend_mechanism
from nash_skills.decision_clock import DEFAULT_DECISION_TIMING, DecisionClock
from nash_skills.eval_matchup import _pick_with_softmax_fallback, _capture_env_step, _parse_contact_lines
from nash_skills.winner_inference import infer_terminal_winner
from nash_skills.v2.collect_data import _build_ppo_obs

SkillEnv = None  # lazy-imported in run_matchup_joint, like collect_data_attack_defend.py's SkillEnv

__all__ = [
    "LEARNED_STRATEGIES_JOINT",
    "VALID_STRATEGIES_JOINT",
    "MatchupResultJoint",
    "build_phi_table_joint",
    "q_values_for_player_joint",
    "phi_values_for_player_joint",
    "run_matchup_joint",
    "make_picker_joint",
]

LEARNED_STRATEGIES_JOINT = (
    "nash-p-hard", "nash-p-br", "nash-p-minimax", "nash-p-adaptive", "nash-p-pga", "ibr", "ibr-q",
)
VALID_STRATEGIES_JOINT = LEARNED_STRATEGIES_JOINT + ("random", "ibr-sim")

SELFPLAY_STRATEGY_KINDS = ("selfplay", "selfplay-argmax")

_PHI_STRATEGIES = {"nash-p-hard", "nash-p-br", "nash-p-minimax", "nash-p-adaptive", "nash-p-pga"}

DEFAULT_PGA_STEPS = 60
DEFAULT_PGA_LR = 0.2
DEFAULT_PGA_RESTARTS = 3

FixedJointStrategy = Tuple[str, str]


def _resolve_fixed_joint_index(strategy: FixedJointStrategy) -> int:
    attack_name, defend_name = strategy
    if attack_name not in SKILL_NAMES:
        raise ValueError(f"Unknown attack skill '{attack_name}'. Choose from: {SKILL_NAMES}")
    if defend_name not in DEFEND_POSITION_NAMES:
        raise ValueError(f"Unknown defend position '{defend_name}'. Choose from: {DEFEND_POSITION_NAMES}")
    return joint_action_index(skill_index(attack_name), defend_position_index(defend_name))


def build_phi_table_joint(base_vec, model_p) -> torch.Tensor:
    """
    Evaluate Phi(s, p1_joint, p2_joint) for all N_JOINT_ACTIONS^2 pairs.
    Direct generalization of eval_matchup._build_phi_table (5x5 -> 15x15),
    same batched-single-forward-pass pattern: encode the state once,
    build all N*N joint-action variants, one forward pass.
    """
    rows = []
    for ego_j in range(N_JOINT_ACTIONS):
        for opp_j in range(N_JOINT_ACTIONS):
            base = torch.tensor(base_vec, dtype=torch.float32)
            rows.append(write_joint_pair_by_index(base, ego_j, opp_j))
    batch = torch.stack(rows)
    with torch.no_grad():
        vals = model_p(batch)[:, 0]
    return vals.reshape(N_JOINT_ACTIONS, N_JOINT_ACTIONS)


def q_values_for_player_joint(base_enc, player: int, other_joint_idx: int, model1, model2) -> List[float]:
    """Value of each of `player`'s N_JOINT_ACTIONS candidate joint
    actions, with the other player's joint action fixed. Direct
    generalization of eval_matchup._q_values_for_player."""
    if model1 is None or model2 is None:
        return []
    model = model1 if player == 1 else model2
    values = []
    for candidate in range(N_JOINT_ACTIONS):
        x = torch.tensor(base_enc, dtype=torch.float32).unsqueeze(0)
        if player == 1:
            x = write_joint_pair_by_index(x, candidate, other_joint_idx)
        else:
            x = write_joint_pair_by_index(x, other_joint_idx, candidate)
        with torch.no_grad():
            values.append(float(model(x).item()))
    return values


def phi_values_for_player_joint(base_enc, player: int, other_joint_idx: int, model_p) -> List[float]:
    """Batched analogue of q_values_for_player_joint using the potential
    model -- direct generalization of eval_matchup._phi_values_for_player."""
    if model_p is None:
        return []
    rows = []
    for candidate in range(N_JOINT_ACTIONS):
        x = torch.tensor(base_enc, dtype=torch.float32)
        if player == 1:
            x = write_joint_pair_by_index(x, candidate, other_joint_idx)
        else:
            x = write_joint_pair_by_index(x, other_joint_idx, candidate)
        rows.append(x)
    with torch.no_grad():
        return model_p(torch.stack(rows))[:, 0].tolist()


def _is_per_player_checkpoint(path: str) -> bool:
    import json
    try:
        with open(path + ".json") as f:
            return bool(json.load(f).get("per_player"))
    except FileNotFoundError:
        return False


def _make_per_player_selfplay_picker(checkpoint_path: str, argmax: bool):
    """Per-player, role-gated actor (nash_skills.v2.selfplay_per_player): player p uses
    psi_p, and only the head of its role this exchange (info['attacker']) changes."""
    from nash_skills.v2.selfplay_rollout import _decide, encode_for_player, load_per_player
    policies, _meta = load_per_player(checkpoint_path)
    device = torch.device("cpu")

    def pick_selfplay(player, obs_vec, _other_joint_idx, info=None):
        if info is None or "attacker" not in info or "current_joints" not in info:
            raise ValueError("per-player self-play needs info['attacker'] and info['current_joints']")
        choice = {p: list(joint_action_from_index(info["current_joints"][p - 1])) for p in (1, 2)}
        role = "attack" if info["attacker"] == player else "defend"
        action, _lp, _v = _decide(policies[player], encode_for_player(player, obs_vec, info, choice), role,
                                  device, argmax=argmax)
        choice[player][0 if role == "attack" else 1] = action
        return joint_action_index(*choice[player])

    return pick_selfplay


def _make_selfplay_picker(checkpoint_path: str, argmax: bool):
    """The state is encoded exactly as selfplay_attack_defend.play_one_rally
    does: encode_ego for player 1, encode_opp for player 2, with both players'
    currently active defend positions (own last pick, other's previous pick)."""
    from nash_skills.v2.selfplay_attack_defend import MetaPolicy, check_checkpoint_encoder
    from nash_skills.v2.state_encoder_attack_defend import encode_ego, encode_opp

    if _is_per_player_checkpoint(checkpoint_path):
        return _make_per_player_selfplay_picker(checkpoint_path, argmax)
    check_checkpoint_encoder(checkpoint_path)
    policy = MetaPolicy()
    policy.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    policy.eval()
    state = {"own": 0}

    def pick_selfplay(player, obs_vec, other_joint_idx, info=None):
        own = info["current_joints"][player - 1] if info and "current_joints" in info else state["own"]
        own_defend = defend_position_from_index(joint_action_from_index(own)[1])
        other_defend = defend_position_from_index(joint_action_from_index(other_joint_idx)[1])
        d1, d2 = (own_defend, other_defend) if player == 1 else (other_defend, own_defend)
        encode = encode_ego if player == 1 else encode_opp
        x = torch.from_numpy(encode(obs_vec, info, d1, d2)).float().unsqueeze(0)
        with torch.no_grad():
            attack_logits, defend_logits, _value = policy(x)
        if argmax:
            attack, defend = int(attack_logits[0].argmax()), int(defend_logits[0].argmax())
        else:
            attack = int(torch.distributions.Categorical(logits=attack_logits[0]).sample())
            defend = int(torch.distributions.Categorical(logits=defend_logits[0]).sample())
        state["own"] = joint_action_index(attack, defend)
        return state["own"]

    return pick_selfplay


def make_picker_joint(
    strategy: Union[str, FixedJointStrategy],
    model_p,
    state_encoder_fn=None,
    tau: float = 0.2,
    confidence_margin: float = 0.05,
    model1=None,
    model2=None,
    pga_steps: int = DEFAULT_PGA_STEPS,
    pga_lr: float = DEFAULT_PGA_LR,
    pga_restarts: int = DEFAULT_PGA_RESTARTS,
    pga_seed: int = 0,
    ppo=None,
    skill_profile: str = "aggressive",
    defend_mechanism: str = DEFAULT_DEFEND_MECHANISM,
    ibr_sim_config: Optional[dict] = None,
):
    """
    Return pick_fn(player, obs_vec, other_joint_idx, info=None) -> joint_idx.

    `strategy` is one of LEARNED_STRATEGIES_JOINT, "random", a
    (attack_name, defend_name) tuple -- a FIXED strategy (one of Table
    1's 15 static baselines: always play that one joint action) -- or
    "selfplay:<checkpoint>" / "selfplay-argmax:<checkpoint>" for a trained
    self-play actor, sampled or at its modes (paper Sec. 4.2).
    """
    if isinstance(strategy, tuple):
        fixed_idx = _resolve_fixed_joint_index(strategy)
        return lambda _player, _obs, _other, _info=None: fixed_idx

    if strategy == "random":
        return lambda _player, _obs, _other, _info=None: np.random.randint(N_JOINT_ACTIONS)

    if strategy == "ibr-sim":
        if ppo is None:
            raise ValueError("ibr-sim simulates rollouts and needs the low-level controller: pass ppo=...")
        from nash_skills.ibr_sim import make_ibr_sim_picker
        return make_ibr_sim_picker(ppo, skill_profile, defend_mechanism, seed=pga_seed, **(ibr_sim_config or {}))

    if isinstance(strategy, str) and strategy.split(":", 1)[0] in SELFPLAY_STRATEGY_KINDS:
        kind, path = strategy.split(":", 1)
        return _make_selfplay_picker(path, argmax=(kind == "selfplay-argmax"))

    if strategy in _PHI_STRATEGIES and model_p is None:
        raise ValueError(
            f"Strategy '{strategy}' requires a loaded potential model. "
            "Pass it to make_picker_joint(model_p=...)."
        )

    def _dispatch(action_scores: torch.Tensor) -> int:
        return _pick_with_softmax_fallback(action_scores, tau, confidence_margin)

    def _base_vec(obs_vec, info, player):
        if state_encoder_fn is not None and info is not None:
            return state_encoder_fn(obs_vec, info, player)
        return obs_vec

    # ------------------------------------------------------------------ #
    # nash-p-pga: projected gradient ascent over mixed theta (paper Eq.18) #
    # ------------------------------------------------------------------ #
    if strategy == "nash-p-pga":
        pga_rng = np.random.default_rng(pga_seed)
        pga_state: Dict[str, Any] = {"theta": None, "last_result": None}

        def pick_pga(player, obs_vec, _other_joint_idx, info=None):
            base = torch.as_tensor(_base_vec(obs_vec, info, player), dtype=torch.float32)

            def phi_fn(wA1, wD1, wA2, wD2):
                x = write_mixed_strategy_pair(base, wA1, wD1, wA2, wD2).unsqueeze(0)
                # (1, 90) on purpose: SimpleModel.forward SKIPS its
                # BatchNorm on 1-D input, which would silently change Phi.
                return model_p(x)[0, 0]

            result = pga_maximize(phi_fn, init=pga_state["theta"], steps=pga_steps, lr=pga_lr,
                                  n_restarts=pga_restarts, rng=pga_rng)
            pga_state["theta"] = result.theta
            pga_state["last_result"] = result
            wA, wD = (result.theta[0], result.theta[1]) if player == 1 else (result.theta[2], result.theta[3])
            attack_idx, defend_idx = sample_action_from_theta(pga_rng, wA.numpy(), wD.numpy())
            return joint_action_index(attack_idx, defend_idx)

        pick_pga.pga_state = pga_state
        return pick_pga

    # ------------------------------------------------------------------ #
    # nash-p-hard: optimistic heuristic over Phi rows/columns              #
    # ------------------------------------------------------------------ #
    if strategy == "nash-p-hard":
        def pick_hard(player, obs_vec, _other_joint_idx, info=None):
            phi = build_phi_table_joint(_base_vec(obs_vec, info, player), model_p)
            action_scores = phi.max(dim=1).values if player == 1 else phi.max(dim=0).values
            return _dispatch(action_scores)
        return pick_hard

    # ------------------------------------------------------------------ #
    # nash-p-br: best response fixing opp's current joint action           #
    # ------------------------------------------------------------------ #
    if strategy == "nash-p-br":
        def pick_br(player, obs_vec, other_joint_idx, info=None):
            phi = build_phi_table_joint(_base_vec(obs_vec, info, player), model_p)
            action_scores = phi[:, other_joint_idx] if player == 1 else phi[other_joint_idx, :]
            return _dispatch(action_scores)
        return pick_br

    # ------------------------------------------------------------------ #
    # nash-p-minimax: maximin heuristic                                    #
    # ------------------------------------------------------------------ #
    if strategy == "nash-p-minimax":
        def pick_minimax(player, obs_vec, _other_joint_idx, info=None):
            phi = build_phi_table_joint(_base_vec(obs_vec, info, player), model_p)
            action_scores = phi.min(dim=1).values if player == 1 else phi.min(dim=0).values
            return _dispatch(action_scores)
        return pick_minimax

    # ------------------------------------------------------------------ #
    # nash-p-adaptive: expected value against empirical opponent mix       #
    # ------------------------------------------------------------------ #
    if strategy == "nash-p-adaptive":
        opponent_counts = {
            1: np.full(N_JOINT_ACTIONS, 0.1, dtype=np.float64),
            2: np.full(N_JOINT_ACTIONS, 0.1, dtype=np.float64),
        }

        def pick_adaptive(player, obs_vec, other_joint_idx, info=None):
            phi = build_phi_table_joint(_base_vec(obs_vec, info, player), model_p)

            opponent_counts[player][other_joint_idx] += 1.0
            probs_np = opponent_counts[player] / opponent_counts[player].sum()
            probs = torch.tensor(probs_np, dtype=phi.dtype, device=phi.device)

            action_scores = phi @ probs if player == 1 else probs @ phi
            return _dispatch(action_scores)
        return pick_adaptive

    # ------------------------------------------------------------------ #
    # ibr: Q-based alternating best response (Phi-independent)             #
    # ------------------------------------------------------------------ #
    if strategy == "ibr":
        if model1 is None or model2 is None:
            raise ValueError(
                "ibr requires model1 and model2 Q-value models. "
                "Pass them to make_picker_joint(model1=..., model2=...)."
            )

        IBR_STEPS = 4

        def pick_ibr(player, obs_vec, other_joint_idx, info=None):
            base_enc = torch.tensor(_base_vec(obs_vec, info, player), dtype=torch.float32)

            s1 = other_joint_idx
            s2 = other_joint_idx

            for _ in range(IBR_STEPS):
                q1_vals = []
                for ego_j in range(N_JOINT_ACTIONS):
                    x = write_joint_pair_by_index(base_enc.unsqueeze(0), ego_j, s2)
                    with torch.no_grad():
                        q1_vals.append(model1(x).item())
                s1 = int(np.argmax(q1_vals))

                q2_vals = []
                for opp_j in range(N_JOINT_ACTIONS):
                    x = write_joint_pair_by_index(base_enc.unsqueeze(0), s1, opp_j)
                    with torch.no_grad():
                        q2_vals.append(model2(x).item())
                s2 = int(np.argmax(q2_vals))

            return s1 if player == 1 else s2
        return pick_ibr

    # ------------------------------------------------------------------ #
    # ibr-q: Q-based empirical-mix best response                           #
    # ------------------------------------------------------------------ #
    if strategy == "ibr-q":
        if model1 is None or model2 is None:
            raise ValueError(
                "ibr-q requires model1 and model2 Q-value models. "
                "Pass them to make_picker_joint(model1=..., model2=...)."
            )

        opp_counts = [1.0] * N_JOINT_ACTIONS

        def pick_ibr_q(player, obs_vec, other_joint_idx, info=None):
            opp_counts[other_joint_idx] += 1.0
            total = sum(opp_counts)
            opp_mix = [c / total for c in opp_counts]

            base_enc = torch.tensor(_base_vec(obs_vec, info, player), dtype=torch.float32)
            q_model = model1 if player == 1 else model2

            ego_vals = []
            for ego_j in range(N_JOINT_ACTIONS):
                val = 0.0
                for opp_j in range(N_JOINT_ACTIONS):
                    base = base_enc.unsqueeze(0)
                    if player == 1:
                        x = write_joint_pair_by_index(base, ego_j, opp_j)
                    else:
                        x = write_joint_pair_by_index(base, opp_j, ego_j)
                    with torch.no_grad():
                        val += opp_mix[opp_j] * q_model(x).item()
                ego_vals.append(val)
            return int(np.argmax(ego_vals))
        return pick_ibr_q

    raise ValueError(
        f"Unknown strategy '{strategy}'. Choose from: {VALID_STRATEGIES_JOINT} "
        f"or a (attack_name, defend_name) tuple."
    )


@dataclasses.dataclass
class MatchupResultJoint:
    """Two-axis analog of eval_matchup.MatchupResult."""
    strategy1: Union[str, Tuple[str, str]]
    strategy2: Union[str, Tuple[str, str]]
    episodes: int
    ego_wins: int
    opp_wins: int
    ego_contacts: int
    opp_contacts: int
    ego_successes: int
    opp_successes: int
    rally_lengths: List[int]
    truncated_episodes: int = 0
    decision_seconds: Dict[int, List[float]] = dataclasses.field(default_factory=lambda: {1: [], 2: []})
    exchanges: List[int] = dataclasses.field(default_factory=list)
    picker_stats: Dict[int, Any] = dataclasses.field(default_factory=dict)

    @property
    def win_rate(self) -> Optional[float]:
        if self.episodes == 0:
            return None
        return self.ego_wins / self.episodes

    @property
    def avg_rally_length(self) -> Optional[float]:
        if not self.rally_lengths:
            return None
        return float(np.mean(self.rally_lengths))

    @property
    def ego_success_rate(self) -> Optional[float]:
        if self.ego_contacts == 0:
            return None
        return self.ego_successes / self.ego_contacts


def run_matchup_joint(
    strategy1: Union[str, Tuple[str, str]],
    strategy2: Union[str, Tuple[str, str]],
    ppo,
    model_p,
    n_episodes: int,
    max_steps_per_episode: int,
    max_attempts: Optional[int] = None,
    state_encoder_fn=None,
    tau: float = 0.2,
    confidence_margin: float = 0.05,
    model1=None,
    model2=None,
    reset_mode: str = "ready",
    skill_profile: str = "aggressive",
    gantry_speed_scale: float = 1.0,
    defend_mechanism: str = DEFAULT_DEFEND_MECHANISM,
    reset_targets: bool = False,
    decision_timing: str = DEFAULT_DECISION_TIMING,
    ibr_sim_config: Optional[dict] = None,
    picker_seed: int = 0,
) -> MatchupResultJoint:
    """
    Run one Design B matchup headlessly -- the two-axis analog of
    eval_matchup.run_matchup. Drives BOTH env.set_skills and
    env.set_defend_reactive_positions via env._role_tracker transitions
    (SkillEnv already drives this tracker internally once
    set_defend_reactive_positions has real values for both players),
    exactly like collect_data_attack_defend.py's loop, rather than
    run_matchup's env.set_skills-only, raw-ball_x-sign-change scheme.
    Uses the velocity/trajectory-informed reactive mechanism, not the
    static pose-redirect "Option B" (set_defend_target_positions) --
    the latter was found net-harmful (mechanism_validation_target_pose.py),
    the former is the fix (mechanism_validation_reactive_pose.py).

    Attempts rallies until `n_episodes` DECISIVE (non-truncated) ones
    are completed, or `max_attempts` total attempts are used (default:
    10x n_episodes) -- same safety-cap shape as
    collect_data_attack_defend.py's max_rallies, so an unlucky streak of
    truncations can't hang forever.

    `defend_mechanism` selects the reactive (receiver) or preparation
    (player who just hit) defend mechanism; see nash_skills.defend_mechanisms.
    """
    validate_defend_mechanism(defend_mechanism)
    clock = DecisionClock(decision_timing)
    global SkillEnv
    if SkillEnv is None:
        from nash_skills.env_wrapper import SkillEnv as _SkillEnv
        SkillEnv = _SkillEnv

    if max_attempts is None:
        max_attempts = max(10 * n_episodes, 10)

    picker_kw = dict(state_encoder_fn=state_encoder_fn, tau=tau, confidence_margin=confidence_margin,
                     model1=model1, model2=model2, ppo=ppo, skill_profile=skill_profile,
                     defend_mechanism=defend_mechanism, ibr_sim_config=ibr_sim_config)
    pick1 = make_picker_joint(strategy1, model_p, pga_seed=picker_seed, **picker_kw)
    pick2 = make_picker_joint(strategy2, model_p, pga_seed=picker_seed + 1, **picker_kw)
    decision_seconds = {1: [], 2: []}
    exchanges_per_rally: List[int] = []

    env = SkillEnv(
        proc_id=1,
        reset_mode=reset_mode,
        skill_profile=skill_profile,
        gantry_speed_scale=gantry_speed_scale,
        reset_targets=reset_targets,
    )

    def _apply_joint(ego_joint: int, opp_joint: int):
        ego_attack, ego_defend = joint_action_from_index(ego_joint)
        opp_attack, opp_defend = joint_action_from_index(opp_joint)
        env.set_skills(skill_from_index(ego_attack), skill_from_index(opp_attack))
        set_defend_choices(
            env, defend_mechanism,
            defend_position_from_index(ego_defend), defend_position_from_index(opp_defend),
        )

    episodes = 0
    truncated_episodes = 0
    ego_wins = opp_wins = 0
    ego_contacts = opp_contacts = ego_successes = opp_successes = 0
    rally_lengths: List[int] = []

    attempt = 0
    while episodes < n_episodes and attempt < max_attempts:
        attempt += 1

        # Bootstrap: an arbitrary initial joint action (index 0 for
        # both) until the first role-tracker transition lets the
        # pickers see a real state.
        _apply_joint(0, 0)
        obs, info = env.reset()

        ego_joint = opp_joint = 0
        clock.reset()
        n_decisions = 0
        steps = 0
        done = False
        rally_ego_contacts = rally_opp_contacts = rally_ego_successes = rally_opp_successes = 0

        while not done and steps < max_steps_per_episode:
            ppo_obs1 = _build_ppo_obs(obs, info, player=1)
            ppo_obs2 = _build_ppo_obs(obs, info, player=2)
            a1, _ = ppo.predict(ppo_obs1, deterministic=True)
            a2, _ = ppo.predict(ppo_obs2, deterministic=True)
            action = np.zeros(18, dtype=np.float32)
            action[:9] = a1[:9]
            action[9:] = a2[:9]

            (obs, _reward, done, _trunc, info), lines = _capture_env_step(env, action)
            ec, oc, es, osu = _parse_contact_lines(lines)
            rally_ego_contacts += ec
            rally_opp_contacts += oc
            rally_ego_successes += es
            rally_opp_successes += osu
            steps += 1

            attacker = clock.update(env)
            if attacker is not None:
                # Simultaneous move (paper Sec. 3.2): neither picker sees the
                # other's newly sampled action, only the one it played last.
                # current_joints tells stateful pickers what is actually in
                # force (each rally restarts both players at joint 0);
                # attacker is this exchange's receiver; env is the live env
                # (simulation-based pickers snapshot it, nobody steps it).
                pick_info = {**info, "current_joints": (ego_joint, opp_joint), "attacker": attacker, "env": env}
                t0 = time.perf_counter()
                new_ego = pick1(1, obs, opp_joint, pick_info)
                t1 = time.perf_counter()
                new_opp = pick2(2, obs, ego_joint, pick_info)
                decision_seconds[1].append(t1 - t0)
                decision_seconds[2].append(time.perf_counter() - t1)
                ego_joint, opp_joint = new_ego, new_opp
                _apply_joint(ego_joint, opp_joint)
                n_decisions += 1

        if done:
            winner = infer_terminal_winner(obs, info, fallback="position")
            if winner == "ego":
                ego_wins += 1
            elif winner == "opp":
                opp_wins += 1
            episodes += 1
            rally_lengths.append(steps)
            exchanges_per_rally.append(n_decisions)
            ego_contacts += rally_ego_contacts
            opp_contacts += rally_opp_contacts
            ego_successes += rally_ego_successes
            opp_successes += rally_opp_successes
        else:
            truncated_episodes += 1

    env.close()

    return MatchupResultJoint(
        strategy1=strategy1, strategy2=strategy2, episodes=episodes,
        ego_wins=ego_wins, opp_wins=opp_wins,
        ego_contacts=ego_contacts, opp_contacts=opp_contacts,
        ego_successes=ego_successes, opp_successes=opp_successes,
        rally_lengths=rally_lengths, truncated_episodes=truncated_episodes,
        decision_seconds=decision_seconds, exchanges=exchanges_per_rally,
        picker_stats={p: getattr(f, "stats", None) for p, f in ((1, pick1), (2, pick2))},
    )
