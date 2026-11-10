"""
Design B data collector: joint-action collection for the
attacker/defender extension.

Collects rally data under Design B's interface: both players always hold
a (attack_skill, defend_position) joint choice; the environment applies
whichever half is physically relevant based on current ball direction
(nash_skills.env_wrapper.SkillEnv.set_skills / set_defend_target_positions,
already fully general for independent per-player values -- no new
mechanism built here).

Uses set_defend_reactive_positions (nash_skills.defend_reactive_pose), NOT
set_defend_positions (hard-override) or set_defend_target_positions
(the static pose-redirect "Option B", also found net-harmful: its target
was ball-independent, creating a 0.5+ unit discontinuity at handoff that
the remaining flight-time couldn't close -- avg_delta_tested_wr stayed in
-0.27..-0.31 across three separate fix attempts). The reactive mechanism
instead recomputes a real ball-trajectory interception prediction every
step and adds a lateral bias that fades to zero as the ball approaches --
no freeze, no handoff, no discontinuity by construction. Matched-pair
validation (scripts/mechanism_validation_reactive_pose.py): contacts
positive in every tested cell (vs negative for every prior mechanism),
avg_delta_tested_wr improved from -0.243 (hard-override, the previous
best) to roughly -0.04..-0.15 depending on evaluation step budget (the
residual is substantially a step-cap artifact -- rallies run much longer
now that both players return far more often, so a fixed cap increasingly
truncates them before a winner is decided -- not a real defensive
deficit; see that script's own docstring and this session's analysis).
This collector must use the fixed mechanism so downstream training data
reflects it, not either harmful predecessor.

Fixed-per-rally sampling (changed this round, root-caused via
scripts/diagnose_collector_stall.py): the original design resampled a
fresh (ego_attack, ego_defend, opp_attack, opp_defend) at EVERY
decision (net crossing), not just once per rally. Under the reactive
mechanism this collapsed throughput to near zero -- traced directly:
10/10 sampled rallies hit the full step cap with ~41 decisions each at
an almost perfectly regular ~65-step cadence that never broke, i.e. a
sustained, effectively non-terminating rally, not a fluke. The same
symptom (trunc=1.00 every iteration) reproduced independently in
nash_skills/v2/selfplay_attack_defend.py's smoke test, which resamples
the same way -- this is a structural property of fully re-randomizing
BOTH axes for BOTH players every exchange, not a collector-specific
bug: with no consistent strategy on either side, the reactive
mechanism's much-improved defense dominates regardless of which named
choice is nominally active, and rallies rarely resolve within any
practical step budget. mechanism_validation_reactive_pose.py's own
full-scale numbers (truncation 0.13-0.93 depending on cell, never
~100%) already showed fixed-for-the-whole-rally choices terminate far
more reliably than this -- consistent with that being the dominant
factor, not the mechanism being unresolvable outright. Fix: sample
ONCE per rally (matching the validation harness's own design), still
recording one state per decision as before (multiple decisions per
rally, crediting whichever physical player was defending at each one),
just with identical (ego_attack, ego_defend, opp_attack, opp_defend)
values across every decision in a given rally -- still valuable
training data (distinct physical states per decision even when the
label repeats), just coarser-grained coverage than per-decision
resampling would have given if it had worked.

This is a NEW, separate collector -- not a mode added to collect_data.py.
Design B's loop is structurally different enough (recording two choices
per player instead of one, tracking which physical player is currently
defending) that extending the existing grid/retry/balance machinery in
place risks destabilizing the working legacy collector. Legacy
collect_data.py, its 76-dim encoder, and the one-hot 84-dim encoder are
all completely untouched by this module.

Collection mode: `sampling="random"` (default, backward compatible) draws
each joint action uniformly, independent of history -- the original
first-pass validation mode. `sampling="grid"` round-robins the full
15x15=225 joint-action space via GridJointActionSampler, giving every
combination roughly equal coverage instead of random variance (a random
pilot run left coverage uneven: 116/225 combos hit across 150 decisions,
mean 1.3/combo) -- use this for real training-data collection. At each
net crossing -- detected via SkillEnv's own DefenderRoleTracker
(env._role_tracker.defending_player changing, including its initial
None -> X latch; SkillEnv already drives this tracker internally every
step once set_defend_target_positions has been called with real values
for both players, so nothing new needs to run it here) -- the state is
recorded against the (attack, defend) choices that were active during
the segment that just concluded, then a fresh (ego_attack, ego_defend,
opp_attack, opp_defend) is sampled for the next segment. This reuses the
production, contact-noise-resistant role detection rather than
collect_data.py's cruder raw ball_x sign-change check.

Reuses, does not reimplement:
  nash_skills.env_wrapper.SkillEnv (set_skills, set_defend_target_positions,
      and its internal _role_tracker)
  nash_skills.v2.collect_data._build_ppo_obs
  nash_skills.v2.state_encoder_attack_defend.encode_ego
  nash_skills.v2.labeling.detect_winner
  nash_skills.eval_matchup._capture_env_step / _parse_contact_lines --
      per-rally contact/return-success counts, for the mechanism-health
      check (contact rate, rally length) directly comparable to the
      none/p2_only/symmetric numbers from ab_defend_symmetry.py, rather
      than trusting the pipeline "ran end to end" as proof the mechanism
      is healthy

nash_skills.v2.labeling.check_balance/summarise_balance are NOT reused
here -- they key on one skill-pair per whole rally (rally["skill1"]/
["skill2"]), which doesn't fit Design B's per-DECISION joint actions
(multiple different choices can occur within one rally). A small,
decision-level balance helper is defined locally instead
(_summarize_decision_balance).

sampling="mixed" (new, paper Sec 4.1): one MIXED strategy theta per
player per rally (nash_skills.v2.mixed_strategy.sample_theta -- a pure
target-preparation pairing with probability --p-vertex, otherwise uniform
over Delta_5 x Delta_3), held fixed for the rally; a fresh (attack,
defend) is DRAWN from theta at every decision. The recorded state carries
theta itself in the four action slots (write_mixed_strategy_pair), not
the realized action's one-hot -- the label is a sample of the return
under theta, which is exactly what V^i(x, theta) and Phi(x, theta) need.
Train on it with train_q_model_5skill_v3_attack_defend.py --theta-mode
mixed. Per-decision draws from a FIXED theta are not the per-decision
re-randomization that once looked like a stall cause: that stall was the
neutral-aim orientation bug (see nash_skills.defend_reactive_pose).

Run:
    MUJOCO_GL=egl PYTHONPATH=. venv/bin/python \\
        nash_skills/v2/collect_data_attack_defend.py --decisions 100 --sampling grid
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import pickle as pkl
from collections import Counter

import numpy as np
import torch

from nash_skills.skills import SKILL_NAMES, skill_from_index
from nash_skills.defend_positions import DEFEND_POSITION_NAMES, defend_position_from_index
from nash_skills.defend_mechanisms import (
    DEFEND_MECHANISMS, DEFAULT_DEFEND_MECHANISM, set_defend_choices, validate_defend_mechanism,
)
from nash_skills.decision_clock import DECISION_TIMINGS, DEFAULT_DECISION_TIMING, DecisionClock
from nash_skills.v2.collect_data import _build_ppo_obs
from nash_skills.v2.state_encoder_attack_defend import (
    ENCODER_DIMS, STATE_DIM_ATTACK_DEFEND, STATE_ENCODER_NAME, encode_ego, state_dim,
)
from nash_skills.v2.labeling import detect_winner
from nash_skills.v2.joint_action import write_mixed_strategy_pair
from nash_skills.v2.mixed_strategy import DEFAULT_P_VERTEX, sample_theta, sample_action_from_theta
from nash_skills.eval_matchup import _capture_env_step, _parse_contact_lines

PPO = None
SkillEnv = None

PPO_MODEL_PATH = "logs/best_model_tracker1/best_model"
DEFAULT_OUTPUT = "data/rallies_attack_defend_random.pkl"
TARGET_DECISIONS = 100
MAX_RALLIES = 500  # safety cap -- decisions trigger far more often than
                    # this in practice (multiple per rally), this just
                    # guards against an unlucky streak of all-truncated
                    # attempts running forever
MAX_STEPS_PER_EPISODE = 1000


def _sample_joint_action(rng: np.random.Generator):
    """One (ego_attack, ego_defend, opp_attack, opp_defend) sample,
    uniform over the full 5x3 per-player space -- the "random
    joint-action collection" mode from the original plan, deliberately
    simpler than a full grid sweep for this first validation pass."""
    return (
        str(rng.choice(SKILL_NAMES)),
        str(rng.choice(DEFEND_POSITION_NAMES)),
        str(rng.choice(SKILL_NAMES)),
        str(rng.choice(DEFEND_POSITION_NAMES)),
    )


class GridJointActionSampler:
    """
    Round-robin sampler over the full 225-combination joint-action space
    (5 ego_attack x 3 ego_defend x 5 opp_attack x 3 opp_defend),
    reshuffled only once exhausted -- every combination is drawn before
    any repeats, unlike _sample_joint_action's pure-random draw (which
    left the pilot run's coverage uneven: 116/225 combos hit across 150
    decisions, mean 1.3/combo, min 1, max 4). A collection run targeting
    N decisions under this sampler gets every combination hit
    floor(N/225) or ceil(N/225) times instead.

    Indices are shuffled via the passed rng (not the combo tuples
    themselves) to avoid numpy coercing a list of string-tuples into an
    array when shuffled directly.
    """

    def __init__(self, rng: np.random.Generator):
        self._rng = rng
        self.all_combos = [
            (ea, ed, oa, od)
            for ea in SKILL_NAMES
            for ed in DEFEND_POSITION_NAMES
            for oa in SKILL_NAMES
            for od in DEFEND_POSITION_NAMES
        ]
        self._queue = []

    def sample(self):
        if not self._queue:
            order = self._rng.permutation(len(self.all_combos))
            self._queue = [self.all_combos[i] for i in order]
        return self._queue.pop()


SAMPLING_MODES = ("random", "grid", "mixed")


def _make_sampler(sampling: str, rng: np.random.Generator):
    """Pure-joint-action samplers only -- "mixed" is handled in collect()
    itself, since it needs a per-rally theta AND per-decision draws."""
    if sampling == "random":
        return lambda: _sample_joint_action(rng)
    if sampling == "grid":
        grid = GridJointActionSampler(rng)
        return grid.sample
    raise ValueError(f"Unknown sampling mode '{sampling}'. Choose from: {SAMPLING_MODES}")


def _draw_from_thetas(rng: np.random.Generator, ego_theta, opp_theta):
    """One (ego_attack, ego_defend, opp_attack, opp_defend) name tuple
    drawn from each player's mixed strategy."""
    ea, ed = sample_action_from_theta(rng, *ego_theta)
    oa, od = sample_action_from_theta(rng, *opp_theta)
    return (skill_from_index(ea), defend_position_from_index(ed),
            skill_from_index(oa), defend_position_from_index(od))


def _summarize_decision_balance(rallies: list) -> Counter:
    """Count occurrences of each (ego_attack, ego_defend, opp_attack,
    opp_defend) combination across all recorded decisions -- the
    Design-B analogue of labeling.summarise_balance, but at the decision
    level rather than one-pair-per-rally."""
    counts = Counter()
    for r in rallies:
        for ea, ed, oa, od in zip(
            r["ego_attack_choices"], r["ego_defend_choices"],
            r["opp_attack_choices"], r["opp_defend_choices"],
        ):
            counts[(ea, ed, oa, od)] += 1
    return counts


def collect(
    target_decisions: int = TARGET_DECISIONS,
    output_path: str = DEFAULT_OUTPUT,
    ppo_path: str = PPO_MODEL_PATH,
    max_steps_per_episode: int = MAX_STEPS_PER_EPISODE,
    max_rallies: int = MAX_RALLIES,
    progress_every: int = 10,
    reset_mode: str = "ready",
    skill_profile: str = "aggressive",
    gantry_speed_scale: float = 1.0,
    seed: int = 0,
    quiet: bool = False,
    sampling: str = "random",
    p_vertex: float = DEFAULT_P_VERTEX,
    defend_mechanism: str = DEFAULT_DEFEND_MECHANISM,
    reset_targets: bool = False,
    decision_timing: str = DEFAULT_DECISION_TIMING,
    encoder: str = STATE_ENCODER_NAME,
) -> list:
    """
    `encoder` names the state encoder (nash_skills.v2.state_encoder_attack_defend.ENCODER_DIMS);
    attack_defend_joint_v3 adds the ball's spin and the rally status to every state.

    Collect rallies via joint-action sampling until `target_decisions`
    total decisions (net crossings, from accepted rallies only) have
    been recorded, or `max_rallies` rallies have been attempted,
    whichever comes first.

    `sampling`: "random" (default, backward-compatible) draws each joint
    action uniformly, independent of history. "grid" round-robins the
    full 225-combination space via GridJointActionSampler, giving every
    combination roughly equal coverage instead of random variance --
    see that class's docstring. "mixed": one mixed strategy theta per
    player per rally (pure pairing with probability p_vertex), actions
    drawn from theta each decision, theta recorded in the state's action
    slots -- see the module docstring.

    `defend_mechanism`: "reactive" (receiver's approach bias) or
    "preparation" (the player who just hit prepares, as in the paper) --
    see nash_skills.defend_mechanisms. Recorded on every rally.

    `reset_targets`: restore both racket targets before every rally so a
    rally depends only on its seed (SkillEnv's reset_targets).

    `decision_timing`: "crossing" (net crossings, legacy) or "contact" (each
    racket contact starts an exchange, paper Sec. 3.2) -- see
    nash_skills.decision_clock. Recorded on every rally, with the attacker
    at each decision.

    Returns the full list of accepted rally dicts (also saved to
    output_path).
    """
    if sampling not in SAMPLING_MODES:
        raise ValueError(f"Unknown sampling mode '{sampling}'. Choose from: {SAMPLING_MODES}")
    validate_defend_mechanism(defend_mechanism)
    state_dim(encoder)  # fails fast on an unknown encoder name
    clock = DecisionClock(decision_timing)
    mixed = sampling == "mixed"
    theta_rng = np.random.default_rng(seed) if mixed else None
    sample_joint_action = None if mixed else _make_sampler(sampling, np.random.default_rng(seed))
    global SkillEnv
    if SkillEnv is None:
        from nash_skills.env_wrapper import SkillEnv as _SkillEnv
        SkillEnv = _SkillEnv

    env = SkillEnv(
        proc_id=1,
        reset_mode=reset_mode,
        skill_profile=skill_profile,
        gantry_speed_scale=gantry_speed_scale,
        reset_targets=reset_targets,
    )
    global PPO
    if PPO is None:
        from stable_baselines3 import PPO as _PPO
        PPO = _PPO
    model = PPO.load(ppo_path)

    all_rallies = []
    total_decisions = 0
    accepted = 0
    discarded = 0
    rally_index = 0

    def _log(msg):
        if not quiet:
            print(msg, flush=True)

    while total_decisions < target_decisions and rally_index < max_rallies:
        rally_index += 1
        if mixed:
            ego_theta = sample_theta(theta_rng, p_vertex)
            opp_theta = sample_theta(theta_rng, p_vertex)
            ego_attack, ego_defend, opp_attack, opp_defend = _draw_from_thetas(theta_rng, ego_theta, opp_theta)
        else:
            ego_attack, ego_defend, opp_attack, opp_defend = sample_joint_action()
        env.set_skills(ego_attack, opp_attack)
        set_defend_choices(env, defend_mechanism, ego_defend, opp_defend)
        obs, info = env.reset()

        curr_states = []
        curr_ego_attack, curr_ego_defend = [], []
        curr_opp_attack, curr_opp_defend = [], []
        curr_defending_player = []

        curr_attacker = []
        clock.reset(env, obs)
        prev_defending_player = env._role_tracker.defending_player
        steps = 0
        done = False
        ego_contacts = opp_contacts = ego_successes = opp_successes = 0

        while not done and steps < max_steps_per_episode:
            ppo1 = _build_ppo_obs(obs, info, player=1)
            ppo2 = _build_ppo_obs(obs, info, player=2)
            a1, _ = model.predict(ppo1, deterministic=True)
            a2, _ = model.predict(ppo2, deterministic=True)
            action = np.zeros(18, dtype=np.float32)
            action[:9] = a1[:9]
            action[9:] = a2[:9]

            (obs, _reward, done, _trunc, info), lines = _capture_env_step(env, action)
            ec, oc, es, osu = _parse_contact_lines(lines)
            ego_contacts += ec
            opp_contacts += oc
            ego_successes += es
            opp_successes += osu
            steps += 1

            defending_player = env._role_tracker.defending_player
            attacker = clock.update(env, obs) if not (done and decision_timing == "crossing") else None
            if attacker is not None:
                # A decision just concluded: record the state against the
                # choices that were active during it, then resample fresh
                # choices for the upcoming segment. The role credited is
                # whoever was defending during the segment that just
                # ended (prev_defending_player) -- or, if this is the
                # very first latch of the rally (prev was None), the
                # newly-latched role, since there was no prior segment.
                # Contact timing: credited as the new exchange's attacker.
                if decision_timing == "crossing":
                    credited_role = prev_defending_player if prev_defending_player is not None else defending_player
                else:
                    credited_role = attacker
                curr_attacker.append(attacker)
                state = encode_ego(obs, info, ego_defend, opp_defend, encoder=encoder)
                if mixed:
                    # The label is a return under theta, so the state's
                    # action slots carry theta, not this segment's draw.
                    state = write_mixed_strategy_pair(
                        torch.from_numpy(state), ego_theta[0], ego_theta[1],
                        opp_theta[0], opp_theta[1]).numpy()
                curr_states.append(state)
                curr_ego_attack.append(ego_attack)
                curr_ego_defend.append(ego_defend)
                curr_opp_attack.append(opp_attack)
                curr_opp_defend.append(opp_defend)
                curr_defending_player.append(credited_role)
                if mixed:
                    # theta stays fixed for the rally; a fresh action is
                    # DRAWN from it for the next segment (paper Sec 4.1).
                    ego_attack, ego_defend, opp_attack, opp_defend = _draw_from_thetas(
                        theta_rng, ego_theta, opp_theta)
                    env.set_skills(ego_attack, opp_attack)
                    set_defend_choices(env, defend_mechanism, ego_defend, opp_defend)
                # In random/grid modes, NOT resampled here -- see "Fixed-
                # per-rally sampling" in the module docstring.

            prev_defending_player = defending_player

        winner = detect_winner([obs], done=done, info=info)

        if done and len(curr_states) > 0 and winner in (1, 2):
            rally = {
                "states": curr_states,
                "ego_attack_choices": curr_ego_attack,
                "ego_defend_choices": curr_ego_defend,
                "opp_attack_choices": curr_opp_attack,
                "opp_defend_choices": curr_opp_defend,
                "defending_player": curr_defending_player,
                "attacker": curr_attacker,
                "winner": winner,
                "encoder": encoder,
                "defend_mechanism": defend_mechanism,
                "decision_timing": decision_timing,
                "skill_profile": skill_profile,
                "steps": steps,
                "ego_contacts": ego_contacts,
                "opp_contacts": opp_contacts,
                "ego_successes": ego_successes,
                "opp_successes": opp_successes,
            }
            if mixed:
                rally["ego_theta"] = [ego_theta[0].tolist(), ego_theta[1].tolist()]
                rally["opp_theta"] = [opp_theta[0].tolist(), opp_theta[1].tolist()]
                rally["theta_mode"] = "mixed"
            all_rallies.append(rally)
            accepted += 1
            total_decisions += len(curr_states)
        else:
            discarded += 1

        if progress_every > 0 and rally_index % progress_every == 0:
            _log(f"  rally {rally_index}: accepted={accepted} discarded={discarded} "
                 f"total_decisions={total_decisions}/{target_decisions}")

    env.close()

    _log(f"\nDone: {rally_index} rallies attempted, {accepted} accepted, "
         f"{discarded} discarded, {total_decisions} decisions recorded")

    if all_rallies:
        avg_steps = sum(r["steps"] for r in all_rallies) / len(all_rallies)
        avg_decisions_per_rally = total_decisions / len(all_rallies)
        total_ego_contacts = sum(r["ego_contacts"] for r in all_rallies)
        total_opp_contacts = sum(r["opp_contacts"] for r in all_rallies)
        _log(f"Mechanism-health check (compare against none/p2_only/symmetric "
             f"from ab_defend_symmetry.py): avg_steps/rally={avg_steps:.1f}  "
             f"avg_decisions/rally={avg_decisions_per_rally:.2f}  "
             f"avg_ego_contacts/rally={total_ego_contacts/len(all_rallies):.2f}  "
             f"avg_opp_contacts/rally={total_opp_contacts/len(all_rallies):.2f}")

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            pkl.dump(all_rallies, f)
        _log(f"Saved {len(all_rallies)} rallies to {output_path}")

    balance = _summarize_decision_balance(all_rallies)
    _log(f"\nDecision-level joint-action coverage: {len(balance)} distinct "
         f"combinations out of {len(SKILL_NAMES) * len(DEFEND_POSITION_NAMES) * len(SKILL_NAMES) * len(DEFEND_POSITION_NAMES)} possible")
    if balance:
        counts = list(balance.values())
        _log(f"  per-combination count: min={min(counts)} max={max(counts)} "
             f"mean={sum(counts)/len(counts):.2f}")

    return all_rallies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect random joint-action rally data for the Design B attacker/defender extension."
    )
    parser.add_argument("--decisions", type=int, default=TARGET_DECISIONS,
                        help=f"Target number of recorded decisions, from accepted rallies only (default: {TARGET_DECISIONS})")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"Output pickle path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--ppo", type=str, default=PPO_MODEL_PATH,
                        help="Path to PPO model checkpoint")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_EPISODE,
                        help=f"Maximum steps per rally attempt (default: {MAX_STEPS_PER_EPISODE})")
    parser.add_argument("--max-rallies", type=int, default=MAX_RALLIES,
                        help=f"Safety cap on rally attempts (default: {MAX_RALLIES})")
    parser.add_argument("--progress-every", type=int, default=10,
                        help="Print progress every N rally attempts; <=0 disables")
    parser.add_argument("--reset-mode", choices=["clean", "ready", "carryover"], default="ready",
                        help="SkillEnv reset mode")
    parser.add_argument("--skill-profile", default="aggressive",
                        help="Skill target profile")
    parser.add_argument("--gantry-speed-scale", type=float, default=1.0,
                        help="Gantry movement speed multiplier")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for joint-action sampling")
    parser.add_argument("--sampling", choices=SAMPLING_MODES, default="random",
                        help="'random' (default, uniform draw), 'grid' (round-robin "
                             "full 225-combo coverage), or 'mixed' (per-rally mixed "
                             "strategy theta, paper Sec 4.1)")
    parser.add_argument("--p-vertex", type=float, default=DEFAULT_P_VERTEX,
                        help="--sampling mixed only: probability a rally's theta is a "
                             "pure target-preparation pairing (simplex vertex)")
    parser.add_argument("--defend-mechanism", choices=DEFEND_MECHANISMS, default=DEFAULT_DEFEND_MECHANISM,
                        help="'reactive' (receiver's approach bias) or 'preparation' "
                             "(the player who just hit prepares, paper Sec 3.2)")
    parser.add_argument("--reset-targets", action="store_true",
                        help="Restore both racket targets before every rally "
                             "(KukaTennisEnv.reset() otherwise carries them over)")
    parser.add_argument("--encoder", choices=sorted(ENCODER_DIMS), default=STATE_ENCODER_NAME,
                        help="state encoder; attack_defend_joint_v3 adds the ball's spin and the rally status")
    parser.add_argument("--decision-timing", choices=DECISION_TIMINGS, default=DEFAULT_DECISION_TIMING,
                        help="'crossing' (net crossings, legacy) or 'contact' (each racket "
                             "contact starts an exchange, paper Sec 3.2)")
    args = parser.parse_args()

    collect(
        decision_timing=args.decision_timing,
        target_decisions=args.decisions,
        output_path=args.output,
        ppo_path=args.ppo,
        max_steps_per_episode=args.max_steps,
        max_rallies=args.max_rallies,
        progress_every=args.progress_every,
        reset_mode=args.reset_mode,
        skill_profile=args.skill_profile,
        gantry_speed_scale=args.gantry_speed_scale,
        seed=args.seed,
        sampling=args.sampling,
        p_vertex=args.p_vertex,
        defend_mechanism=args.defend_mechanism,
        reset_targets=args.reset_targets,
        encoder=args.encoder,
    )
