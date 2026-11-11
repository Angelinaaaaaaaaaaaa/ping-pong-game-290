#!/usr/bin/env python3
"""Recover actual pre-decision gantries and describe conditional attack choices.

Reads existing detailed evaluator CSVs only. Each input run remains separate
unless --pool-runs is requested. No simulation or model inference is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shlex
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nash_skills.skills import SKILL_NAMES
from v3.constants import DEFENSE_POSITIONS
from v3.state_encoder import STATE_DIM, STATE_ENCODER_NAME, ATTACKER_QPOS, DEFENDER_QPOS
from v3.racket_target_planner import mirror_position_to_world
from v3.eval_fixed_preposition import player_relative_side


def reconstruct_gantries(state, attacker_id, defender_id):
    """Invert the current role-based state transform; return physical world xy."""
    state = np.asarray(state, dtype=float)
    if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
        raise ValueError('Requires a finite corrected 74-dimensional state')
    if attacker_id not in (1, 2) or defender_id != 3 - attacker_id:
        raise ValueError('Inconsistent physical-player/role IDs')
    result = {}
    for player, offset in ((attacker_id, ATTACKER_QPOS.start), (defender_id, DEFENDER_QPOS.start)):
        xyz = np.r_[state[offset:offset + 2], 0.0]
        if attacker_id == 2:
            xyz = mirror_position_to_world(xyz)
        result[player] = np.asarray(xyz[:2], dtype=float)
    return result


def probabilities(value):
    if pd.isna(value):
        return None
    vector = np.asarray(json.loads(value), dtype=float)
    if vector.shape != (len(SKILL_NAMES),) or not np.isfinite(vector).all() or (vector < -1e-8).any() or not np.isclose(vector.sum(), 1, atol=1e-5):
        return None
    return vector


def summarize(rows, relative=False):
    """Include all nine cells and all five skills; empty cells have NaN probabilities."""
    output, opponent_output = [], []
    for (run, strategy, player), data in rows.groupby(['analysis_run','strategy','analysis_player']):
        first, second = ('own_position', 'opponent_position') if relative else ('p1_position', 'p2_position')
        for positions in itertools.product(DEFENSE_POSITIONS, repeat=2):
            subset = data[(data[first] == positions[0]) & (data[second] == positions[1])]
            output.extend(cell_rows(subset, run, strategy, player, {first: positions[0], second: positions[1]}))
        for position in DEFENSE_POSITIONS:
            subset = data[data.opponent_position == position]
            opponent_output.extend(cell_rows(subset, run, strategy, player, {'opponent_position': position}))
    return pd.DataFrame(output), pd.DataFrame(opponent_output)


def cell_rows(subset, run, strategy, player, condition):
    valid = [p for p in subset.policy_vector if p is not None]
    means = np.mean(valid, axis=0) if valid else np.full(len(SKILL_NAMES), np.nan)
    return [dict(analysis_run=run, strategy=strategy, player=player, **condition,
                 attack_skill=skill, n_decisions=len(subset), n_policy_vectors=len(valid),
                 selected_count=int((subset.attack_skill == skill).sum()),
                 empirical_probability=float((subset.attack_skill == skill).mean()) if len(subset) else np.nan,
                 mean_policy_probability=float(means[i])) for i, skill in enumerate(SKILL_NAMES)]


def plot_grid(frame, first, second, path, value, title):
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharey=True)
    colors = plt.get_cmap('tab10')(np.arange(len(SKILL_NAMES)))
    for i, pos1 in enumerate(DEFENSE_POSITIONS):
        for j, pos2 in enumerate(DEFENSE_POSITIONS):
            ax = axes[i, j]
            cell = frame[(frame[first] == pos1) & (frame[second] == pos2)].set_index('attack_skill').reindex(SKILL_NAMES)
            n, nv = int(cell.n_decisions.iloc[0]), int(cell.n_policy_vectors.iloc[0])
            values = cell[value].to_numpy(float)
            if np.isfinite(values).any():
                ax.bar(np.arange(len(SKILL_NAMES)), values, color=colors)
            else:
                ax.text(.5, .5, 'No observations' if n == 0 else 'No valid policy vectors', transform=ax.transAxes, ha='center', fontsize=9)
            ax.set_title(f'{first}={pos1}; {second}={pos2}\nn={n}, vectors={nv}', fontsize=9)
            ax.set_xticks(np.arange(len(SKILL_NAMES)), SKILL_NAMES, rotation=35, ha='right', fontsize=8)
            ax.set_ylim(0, 1); ax.grid(axis='y', alpha=.2)
            if j == 0: ax.set_ylabel('Attack-skill probability')
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(path, dpi=300, bbox_inches='tight'); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', type=Path, default=[Path('skill_eval/v3/q_policy_eval_canonical')], help='decisions.csv files or directories searched recursively')
    parser.add_argument('--strategies', nargs='+', help='Only these attacker_policy values; default all observed strategies')
    parser.add_argument('--center-tolerance', type=float, default=.05, help='Own-side lateral center band in meters (inclusive)')
    parser.add_argument('--snapshot-tolerance', type=float, default=1e-5)
    parser.add_argument('--player-relative', action='store_true', help='Also aggregate P1/P2 by own/opponent bins for the same strategy and analysis run')
    parser.add_argument('--pool-runs', action='store_true', help='Explicitly pool input runs, weighted by observed attacking decisions')
    parser.add_argument('--output-dir', type=Path, default=Path('reports/v3/conditional_skill_selection'))
    args = parser.parse_args(argv)
    if args.center_tolerance < 0 or args.snapshot_tolerance <= 0:
        parser.error('Tolerances must be nonnegative (snapshot tolerance strictly positive)')
    files = sorted({p.resolve() for source in args.input for p in ([source] if source.is_file() else source.rglob('decisions.csv'))})
    if not files: parser.error('No decision CSV files found')
    decoded, audits = [], []
    for file in files:
        frame = pd.read_csv(file)
        run = file.parent.name + '_' + hashlib.sha256(str(file).encode()).hexdigest()[:8]
        reasons = {}; accepted = 0; snapshot_checks = 0; max_error = 0.0
        metadata = {}
        for name in ('summary.json', 'decisions.json'):
            candidate = file.with_name(name)
            if candidate.exists():
                metadata = json.loads(candidate.read_text()).get('metadata', {})
                if metadata: break
        for index, row in frame.iterrows():
            try:
                declared = metadata.get('q_state_encoder')
                if declared and declared != STATE_ENCODER_NAME:
                    raise ValueError('Incompatible declared state encoder: ' + str(declared))
                a, d = int(row.attacker_id), int(row.defender_id)
                world = reconstruct_gantries(json.loads(row.state_json), a, d)
                error = np.nan
                if 'defender_actual_world_at_commit_json' in row and pd.notna(row.defender_actual_world_at_commit_json):
                    observed = np.asarray(json.loads(row.defender_actual_world_at_commit_json), dtype=float)
                    if observed.shape != (2,) or not np.isfinite(observed).all():
                        raise ValueError('Invalid defender coordinate snapshot')
                    error = float(np.max(np.abs(world[d] - observed)))
                    snapshot_checks += 1; max_error = max(max_error, error)
                    if error > args.snapshot_tolerance:
                        raise ValueError('State/defender snapshot mismatch; historical layout or frame cannot be trusted')
                if declared != STATE_ENCODER_NAME and not np.isfinite(error):
                    raise ValueError('State provenance unavailable: no current encoder metadata or coordinate cross-check')
                if row.attack_skill not in SKILL_NAMES:
                    raise ValueError('Unknown attack skill')
                vector = probabilities(row.get('attacker_policy_probabilities_json', np.nan))
                positions = {p: player_relative_side(p, world[p][1], args.center_tolerance) for p in (1,2)}
                record = dict(source_file=str(file), source_row=int(index), source_run=run,
                    episode_index=row.get('episode_index'), decision_index=row.get('decision_index'),
                    strategy=row.attacker_policy, attacker_id=a, defender_id=d, attack_skill=row.attack_skill,
                    p1_world_x=world[1][0], p1_world_y=world[1][1], p2_world_x=world[2][0], p2_world_y=world[2][1],
                    p1_own_lateral_y=world[1][1], p2_own_lateral_y=-world[2][1],
                    p1_position=positions[1], p2_position=positions[2], own_position=positions[a], opponent_position=positions[d],
                    snapshot_error=error, policy_vector=vector,
                    analysis_run='pooled_inputs' if args.pool_runs else run, analysis_player=f'P{a}')
                accepted += 1
                if not args.strategies or row.attacker_policy in args.strategies:
                    decoded.append(record)
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                reason = str(exc); reasons[reason] = reasons.get(reason, 0) + 1
        audits.append(dict(source_file=str(file), input_decisions=len(frame), recoverable_decisions=accepted,
            excluded_decisions=len(frame)-accepted, snapshot_checks=snapshot_checks, max_snapshot_error=max_error,
            declared_encoder=metadata.get('q_state_encoder'), exclusions=reasons))
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    (out/'audit.json').write_text(json.dumps(dict(files=audits, center_tolerance_m=args.center_tolerance,
        snapshot_tolerance_m=args.snapshot_tolerance, pooled_runs=args.pool_runs), indent=2))
    if not decoded:
        raise SystemExit(f'No usable selected attacker decisions. See {out / "audit.json"}')
    rows = pd.DataFrame(decoded)
    export = rows.drop(columns='policy_vector').copy()
    export['attacker_policy_probabilities_json'] = rows.policy_vector.map(lambda p: json.dumps(p.tolist()) if p is not None else '')
    export.to_csv(out/'reconstructed_decisions.csv', index=False)
    joint, opponent = summarize(rows)
    joint.to_csv(out/'joint_position_skill_summary.csv', index=False)
    opponent.to_csv(out/'opponent_position_skill_summary.csv', index=False)
    charts = []
    def charts_for(data, first, second, prefix):
        for (run, strategy, player), subset in data.groupby(['analysis_run','strategy','player']):
            safe_strategy = ''.join(c if c.isalnum() or c in '_-' else '_' for c in strategy)
            for field, label in [('empirical_probability','Empirical attacking choices'), ('mean_policy_probability','Mean logged attack distribution')]:
                name = f'{prefix}_{run}_{safe_strategy}_{player}_{field}.png'
                plot_grid(subset,first,second,out/name,field,f'{label}: {strategy}, {player}\n{run}')
                charts.append(name)
    charts_for(joint, 'p1_position','p2_position','physical_players')
    if args.player_relative:
        relative = rows.copy(); relative['analysis_player'] = 'pooled_P1_P2'
        relative_joint, relative_opponent = summarize(relative, relative=True)
        relative_joint.to_csv(out/'player_relative_joint_summary.csv', index=False)
        relative_opponent.to_csv(out/'player_relative_opponent_summary.csv', index=False)
        charts_for(relative_joint,'own_position','opponent_position','player_relative')
    note = [
        '# Conditional V3 attack-skill selection: coordinate audit and analysis',
        'Both actual gantries are recoverable from the corrected 74-dimensional pre-decision state. Zero-based indices 0/1 contain attacker canonical x/y; 18/19 contain defender canonical x/y. Role ownership comes from attacker_id/defender_id, not a fixed P1/P2 slot.',
        'If attacker_id=1, canonical coordinates equal world coordinates. If attacker_id=2, undo the shared table reflection: world_x=2*TABLE_SHIFT-canonical_x and world_y=-canonical_y. TABLE_SHIFT is read through the existing mirror helper. Assign the two recovered pairs to their physical players using role IDs.',
        'P1 own-side lateral position is +world_y; P2 own-side lateral position is -world_y. Positive own-side y means left, negative means right. Center includes |own-side y| <= '+str(args.center_tolerance)+' meters, using the evaluator helper. These bins classify measured physical positions, not nearest selected target labels. Longitudinal x is exported but is not used in lateral bins.',
        'Timing evidence: eval_q_policy.py computes logged_state from obs/info before env.set_skills and env.set_defense_position. These setters only configure skill targets/commitments; no physics step occurs before position_snapshot. Therefore the independent defender snapshot can validate reconstructed actual coordinates at the same instant. Neither defense_preposition nor defender_target_world_json is an actual-position measurement.',
        'Encoder evidence: state_encoder._role_raw_slices uses P1 raw qpos[0:9] and P2 raw qpos[9:18], with velocities in separate blocks. _canonical_qpos first converts each robot’s local gantry coordinates into world coordinates and then into the attacker frame. mujoco_env_comp.py constructs observations with both qpos blocks before both qvel blocks. Older 76-dimensional or snapshot-inconsistent logs are excluded rather than interpreted with the corrected layout.',
        'Only attacker_policy is used to attribute a selected attack to a strategy; a strategy controlling only defender_policy on a row contributes no attacking observation. P1/P2 remain separate in the main CSVs. Optional player-relative output combines own/opponent cells across physical identities for the same strategy and selected run scope.',
        'Every condition includes all five actions and its decision count. Empirical probabilities are selected_count/n_decisions. Mean policy probabilities are arithmetic means over valid logged five-action probability vectors, with their own n_policy_vectors count. Empty conditions have zero counts and blank probabilities, not zero-valued distributions.',
        'Opponent-only summaries use the actual current defender’s own-side gantry bin. Run pooling is '+str(args.pool_runs)+'. Explicit pooling weights observations by attacking-decision count and can mix opponents, gamma values, episode lengths and configurations; interpret it only as a description of the selected input decisions. Separate runs are the default. Policy averages and empirical frequencies are distinct; small cells may show large sampling differences.',
        'Schema provenance: metadata is checked where available and reconstructed defender coordinates are checked against actual-world snapshots when present. State length alone does not identify a historical schema; unknown-provenance inputs without a coordinate check are excluded. The attacker position has no independent same-row snapshot, so its recovery relies on the verified encoder mapping and metadata/frame convention.',
        '## Input audit',
        '| Input | Rows | Recoverable | Excluded | Maximum defender world error (m) |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    note.extend(f'| {a["source_file"]} | {a["input_decisions"]} | {a["recoverable_decisions"]} | {a["excluded_decisions"]} | {a["max_snapshot_error"]:.3g} |' for a in audits)
    note += ['## Selected attacking observations', '| Strategy | Physical attacker | Decisions |', '| --- | --- | ---: |']
    note.extend(f'| {strategy} | P{player} | {n} |' for (strategy,player), n in rows.groupby(['strategy','attacker_id']).size().items())
    command = ['python3', 'v3/analyze_conditional_skill_selection.py', '--input', *map(str,args.input),
               '--center-tolerance', str(args.center_tolerance), '--snapshot-tolerance', str(args.snapshot_tolerance),
               '--output-dir', str(out)]
    if args.strategies: command += ['--strategies', *args.strategies]
    if args.player_relative: command += ['--player-relative']
    if args.pool_runs: command += ['--pool-runs']
    note += ['## Reproduction', '```bash\n'+shlex.join(command)+'\n```']
    note += ['## Figures'] + [f'![{name}]({name})' for name in charts]
    (out/'report.md').write_text('\n\n'.join(note)+'\n')
    print(f'Recovered {len(rows)} selected attacker decisions from {len(files)} file(s). Saved {len(charts)} figures and CSV summaries to {out}')


if __name__ == '__main__':
    main()
