#!/usr/bin/env python3
"""Compare incoming-aligned human landing proxies with selected robot skills.

Read-only analysis of saved events/decisions. Cluster bootstrap resamples whole
human matches and robot rallies, retaining all positions within each cluster.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v3.analyze_conditional_skill_selection import summarize
from v3.human_comparison.analyze_spatial_v2 import robot_events, human_events
from v3.human_comparison.spatial import position_labels, region_from_dict
from nash_skills.skills import SKILL_NAMES

HUMAN_DATASET = 'human_p1'
POSITIONS = ('left', 'center', 'right')
DATASETS = (HUMAN_DATASET, 'nash_q', 'potential_ibr')
NAMES = {HUMAN_DATASET: 'Human (landing proxy)', 'nash_q': 'Nash-Q (selected)',
         'potential_ibr': 'Potential IBR (selected)'}
LABELS = ('Left', 'Left Short', 'Center Safe', 'Right Short', 'Right')
COLORS = ('#555555', '#2878b5', '#d07920')


def configure_human_cohort(name):
    global HUMAN_DATASET, DATASETS, NAMES
    if name not in ('human_p1', 'human_p2', 'human_both'):
        raise ValueError('Unknown human cohort')
    HUMAN_DATASET = name
    DATASETS = (name, 'nash_q', 'potential_ibr')
    label = {'human_p1': 'Human P1', 'human_p2': 'Human P2', 'human_both': 'Human P1+P2'}[name]
    NAMES = {name: label + ' (landing proxy)', 'nash_q': 'Nash-Q P1 (selected)',
             'potential_ibr': 'Potential IBR P1 (selected)'}
    return DATASETS, NAMES


def similarity(p, q):
    """Jensen–Shannon divergence (base 2, not square root), TV, Spearman."""
    p, q = np.broadcast_arrays(np.asarray(p, float), np.asarray(q, float))
    middle = .5 * (p + q)
    def kl(x):
        ratio = np.divide(x, middle, out=np.ones_like(x), where=(x > 0) & (middle > 0))
        return np.sum(np.where(x > 0, x * np.log2(ratio), 0), axis=-1)
    js = .5 * (kl(p) + kl(q))
    tv = .5 * np.abs(p - q).sum(axis=-1)
    # Average ranks handle ties; correlation is undefined for constant vectors.
    rp = pd.DataFrame(p.reshape(-1, p.shape[-1])).rank(axis=1).to_numpy()
    rq = pd.DataFrame(q.reshape(-1, q.shape[-1])).rank(axis=1).to_numpy()
    rp -= rp.mean(axis=1, keepdims=True); rq -= rq.mean(axis=1, keepdims=True)
    scale = np.sqrt((rp * rp).sum(axis=1) * (rq * rq).sum(axis=1))
    rho = np.divide((rp * rq).sum(axis=1), scale, out=np.full(len(scale), np.nan), where=scale > 0)
    rho = rho.reshape(p.shape[:-1])
    invalid = ~np.isfinite(p).all(axis=-1) | ~np.isfinite(q).all(axis=-1)
    return {k: np.where(invalid, np.nan, v) for k, v in
            [('js_divergence_bits', js), ('total_variation', tv), ('spearman_rho', rho)]}


def cluster_counts(frame):
    clusters = pd.Index(sorted(frame.cluster.unique()))
    counts = np.zeros((len(clusters), 3, 5), dtype=np.int64)
    np.add.at(counts, (clusters.get_indexer(frame.cluster),
        pd.Index(POSITIONS).get_indexer(frame.opponent_position),
        pd.Index(SKILL_NAMES).get_indexer(frame.attack_skill)), 1)
    return clusters, counts


def normalize(counts):
    total = counts.sum(axis=-1, keepdims=True)
    return np.divide(counts, total, out=np.full(counts.shape, np.nan, dtype=float), where=total > 0)


def bootstrap(counts, replicates, rng):
    """Resample whole clusters; preserve arbitrary condition/category axes."""
    n = len(counts)
    if n == 0:
        raise ValueError('No clusters available')
    shape = counts.shape[1:]
    samples = np.empty((replicates, *shape), dtype=float)
    for start in range(0, replicates, 64):
        end = min(start + 64, replicates)
        weights = rng.multinomial(n, np.full(n, 1 / n), size=end - start)
        samples[start:end] = normalize((weights @ counts.reshape(n, -1)).reshape(-1, *shape))
    return samples


def interval(values):
    values = np.asarray(values); values = values[np.isfinite(values)]
    return (*np.quantile(values, [.025, .975]), len(values)) if len(values) else (np.nan, np.nan, 0)


def load_observations(args):
    root = args.human_analysis_dir
    meta = json.loads((root / 'summary.json').read_text())
    region = region_from_dict(meta['primary_region'])
    if region.name != 'four_quadrants_center_override':
        raise ValueError('Human source must use four_quadrants_center_override')
    region.validate()
    band = meta['primary_band_m']
    players = {'p1': (1,), 'p2': (2,), 'both': (1, 2)}[args.human_players]
    if args.human_events_dir:
        h = human_events(args.human_events_dir, players=players)
        raw_sources = [args.human_events_dir / f for f in ('shot_events.csv', 'all_shot_events.csv', 'summary.json')]
    else:
        h = pd.read_csv(root / 'human_standardized_events.csv', low_memory=False)
        raw_sources = [root / 'human_standardized_events.csv']
    source_player_counts = h.groupby('attacker_id').valid_landing.agg(['size', 'sum']).reset_index()
    h = h[h.valid_landing.fillna(False) & h.attacker_id.isin(players) &
          (h.position_time == 'incoming_crossing')].copy()
    h['dataset'] = HUMAN_DATASET
    h['attack_skill'] = region.classify(h.depth, h.lateral)
    if not h.attack_skill.isin(SKILL_NAMES).all():
        raise ValueError('Accepted human landings outside complete partition')
    if not h.cluster.eq(h.rally_id.str.rsplit('_', n=1).str[0]).all():
        raise ValueError('Human match cluster identifiers do not match rally identifiers')
    frames = [h]; metadata = {'human_dataset': HUMAN_DATASET, 'human_players': list(players),
        'human_region': asdict(region), 'center_band_m': band,
        'human_source_player_counts': source_player_counts.to_dict('records'),
        'max_abs_position_normalized': meta['max_abs_position_normalized'], 'robot_runs': {}}
    paths = [*raw_sources, root / 'summary.json']
    for policy, file in [('nash_q', args.nash_decisions), ('potential_ibr', args.ibr_decisions)]:
        frame, run_meta = robot_events(file, policy)
        header = pd.read_csv(file, nrows=0).columns
        orientation = dict(status='logged_command_coordinates_unavailable')
        if 'attack_target_local_y' in header:
            commands = pd.read_csv(file, usecols=['attacker_id', 'attacker_policy', 'attack_skill', 'attack_target_local_y'])
            commands = commands[(commands.attacker_id == 1) & (commands.attacker_policy == policy)]
            expected = commands.attack_skill.map({'left': 1, 'left_short': 1, 'center_safe': 0, 'right_short': -1, 'right': -1})
            if not np.array_equal(np.sign(commands.attack_target_local_y), expected):
                raise ValueError(f'{policy}: logged skill left/right semantics differ from hitter-relative convention')
            orientation = dict(status='all_logged_command_directions_match', n=len(commands),
                meaning='left commands have positive local y; right commands negative; coordinates audit names only, never substitute for landings')
        frame['attack_skill'] = frame.selected_skill
        frames.append(frame); metadata['robot_runs'][policy] = run_meta
        metadata.setdefault('command_orientation_checks', {})[policy] = orientation
        paths.append(file)
        if file.with_name('summary.json').exists(): paths.append(file.with_name('summary.json'))
    exclusions = []
    for frame in frames:
        frame['p1_position'] = position_labels(frame.p1_own_lateral, band)
        frame['p2_position'] = position_labels(frame.p2_own_lateral, band)
        valid = (frame[['p1_own_lateral', 'p2_own_lateral']].abs() <= meta['max_abs_position_normalized']).all(axis=1)
        valid &= frame.attack_skill.isin(SKILL_NAMES)
        exclusions.append(dict(dataset=frame.dataset.iloc[0], before_context_screen=len(frame),
            excluded_unknown_or_extreme_context=int((~valid).sum()), included=int(valid.sum())))
        frame.drop(frame.index[~valid], inplace=True)
        frame['opponent_position'] = np.where(frame.attacker_id == 1, frame.p2_position, frame.p1_position)
        frame['analysis_run'] = frame.dataset
        frame['strategy'] = frame.dataset
        frame['analysis_player'] = frame.attacker_id.map({1: 'P1', 2: 'P2'})
        frame['policy_vector'] = None
    observations = pd.concat(frames, ignore_index=True)
    metadata['human_incoming_counts_by_player'] = observations[observations.dataset == HUMAN_DATASET].groupby('attacker_id').size().to_dict()
    if not args.human_events_dir and int((observations.dataset == HUMAN_DATASET).sum()) != meta['incoming_context_valid_landings']:
        raise ValueError('Human cohort differs from source incoming nine-context cohort')
    return observations, metadata, exclusions, paths


def aggregate(observations):
    # Reuse the established robotic summarizer; it pools raw observations.
    joint, opponent = summarize(observations)
    def pool(table, conditions):
        table = table.rename(columns={'strategy': 'dataset'})
        result = table.groupby(['dataset', *conditions, 'attack_skill'], sort=False)[['selected_count', 'n_decisions', 'n_policy_vectors']].sum().reset_index()
        result['empirical_probability'] = result.selected_count / result.n_decisions.replace(0, np.nan)
        result['mean_policy_probability'] = np.nan
        return result
    joint = pool(joint, ['p1_position', 'p2_position'])
    opponent = pool(opponent, ['opponent_position'])
    for dataset in DATASETS:
        j = joint[joint.dataset == dataset]
        o = opponent[opponent.dataset == dataset]
        for position in POSITIONS:
            source = observations[(observations.dataset == dataset) & (observations.opponent_position == position)]
            raw = source.attack_skill.value_counts().reindex(SKILL_NAMES, fill_value=0)
            pooled = o[o.opponent_position == position].set_index('attack_skill').selected_count.reindex(SKILL_NAMES)
            np.testing.assert_array_equal(raw, pooled)
        assert o.selected_count.sum() == (observations.dataset == dataset).sum()
    return joint, opponent


def save_plot(fig, path):
    fig.savefig(path, dpi=300, bbox_inches='tight'); plt.close(fig)


def plot_outputs(out, probabilities, draws, totals, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout='constrained')
    for j, (position, ax) in enumerate(zip(POSITIONS, axes)):
        for i, dataset in enumerate(DATASETS):
            x = np.arange(5) + (i - 1) * .25
            ax.bar(x, probabilities[dataset][j], .24, color=COLORS[i], label=NAMES[dataset])
            ci = np.nanquantile(draws[dataset][:, j, :], [.025, .975], axis=0)
            ax.vlines(x, ci[0], ci[1], color='black', lw=.8)
        ax.set_xticks(range(5), LABELS, rotation=35, ha='right')
        ax.set_ylim(0, 1); ax.set_ylabel('Conditional probability')
        ax.set_title('Opponent ' + position.title() + '\n' + ', '.join(
            f'{name}: n={int(totals[d][j])}' for d, name in zip(DATASETS, ['Human', 'Nash-Q', 'Potential IBR'])), fontsize=9)
    axes[0].legend(fontsize=8)
    fig.suptitle('Attacking categories conditioned only on opponent position\n95% match/rally cluster-bootstrap intervals; human landing proxies versus selected robot skills')
    save_plot(fig, out / 'opponent_position_grouped_bars.png')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
    for ax, dataset in zip(axes, DATASETS):
        im = ax.imshow(probabilities[dataset], vmin=0, vmax=1, cmap='Blues', aspect='auto')
        ax.set_xticks(range(5), LABELS, rotation=35, ha='right')
        ax.set_yticks(range(3), [f'P2 {p.title()} (n={int(n)})' for p, n in zip(POSITIONS, totals[dataset])])
        ax.set_title(NAMES[dataset])
        for j in range(3):
            for k in range(5):
                value = probabilities[dataset][j, k]
                ax.text(k, j, f'{value:.3f}' if np.isfinite(value) else 'NA', ha='center', va='center',
                    color='white' if value > .5 else 'black', fontsize=9)
    fig.colorbar(im, ax=list(axes), label='Conditional probability', shrink=.75)
    save_plot(fig, out / 'opponent_position_probability_heatmaps.png')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    for ax, metric, title, limits in zip(axes,
            ['js_divergence_bits', 'total_variation', 'spearman_rho'],
            ['Jensen–Shannon divergence (bits)', 'Total variation distance', 'Spearman correlation (5 categories)'],
            [(0, 1), (0, 1), (-1, 1)]):
        for i, policy in enumerate(DATASETS[1:]):
            rows = metrics[metrics.policy == policy].set_index('opponent_position').reindex(POSITIONS)
            x = np.arange(3) + (i - .5) * .16
            ax.vlines(x, rows[metric + '_ci_low'], rows[metric + '_ci_high'], color=COLORS[i + 1])
            ax.scatter(x, rows[metric], color=COLORS[i + 1], label=NAMES[policy], zorder=3)
        ax.set_xticks(range(3), [p.title() for p in POSITIONS]); ax.set_ylim(*limits)
        ax.set(xlabel='Opponent position', title=title)
    axes[0].legend(fontsize=8)
    fig.suptitle('Exploratory human-proxy versus robot-action similarity (95% cluster-bootstrap intervals)')
    save_plot(fig, out / 'opponent_position_similarity.png')

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout='constrained')
    for j, ax in enumerate(axes):
        for i, dataset in enumerate(DATASETS):
            p = probabilities[dataset][j]
            direction = np.array([p[0] + p[1], p[2], p[3] + p[4]])
            ax.bar(np.arange(3) + (i - 1) * .25, direction, .24, color=COLORS[i], label=NAMES[dataset])
        ax.set_xticks(range(3), ['Left + Left Short', 'Center Safe', 'Right + Right Short'], rotation=20, ha='right')
        ax.set(title='Opponent ' + POSITIONS[j].title(), ylabel='Conditional probability', ylim=(0, 1))
    axes[0].legend(fontsize=8); fig.suptitle('Attacking direction after combining short/deep categories')
    save_plot(fig, out / 'opponent_position_attacking_direction.png')


def run(args):
    configure_human_cohort('human_' + args.human_players)
    out = args.output_dir
    if out.resolve() == args.human_analysis_dir.resolve(): raise ValueError('Use a separate output directory')
    out.mkdir(parents=True, exist_ok=True)
    observations, meta, exclusions, paths = load_observations(args)
    joint, opponent = aggregate(observations)
    joint.to_csv(out / 'nine_context_raw_counts.csv', index=False)
    probabilities = {}; draws = {}; totals = {}; clusters_report = []; rows = []
    rng = np.random.default_rng(args.seed)
    for dataset in DATASETS:
        frame = observations[observations.dataset == dataset]
        cluster_ids, counts = cluster_counts(frame)
        sums = counts.sum(axis=0); totals[dataset] = sums.sum(axis=1)
        probabilities[dataset] = normalize(sums)
        print(f'{dataset}: {len(frame)} observations, {len(cluster_ids)} clusters; bootstrap {args.bootstrap}', flush=True)
        draws[dataset] = bootstrap(counts, args.bootstrap, rng)
        for j, position in enumerate(POSITIONS):
            nc = int((counts[:, j, :].sum(axis=1) > 0).sum())
            clusters_report.append(dict(dataset=dataset, opponent_position=position, n=int(totals[dataset][j]),
                visitation_fraction=float(totals[dataset][j] / len(frame)), n_clusters=nc,
                total_clusters=len(cluster_ids), cluster_unit='match' if dataset == HUMAN_DATASET else 'rally'))
            for k, skill in enumerate(SKILL_NAMES):
                low, high, reps = interval(draws[dataset][:, j, k])
                rows.append(dict(dataset=dataset, opponent_position=position, attack_skill=skill,
                    measurement='realized_landing_proxy' if dataset == HUMAN_DATASET else 'selected_high_level_skill',
                    n_observations=int(totals[dataset][j]), count=int(sums[j, k]), probability=probabilities[dataset][j, k],
                    probability_ci_low=low, probability_ci_high=high, valid_bootstrap_replicates=reps,
                    n_clusters=nc, visitation_fraction=float(totals[dataset][j] / len(frame))))
        baseline = opponent[opponent.dataset == dataset].set_index(['opponent_position', 'attack_skill'])
        for j, position in enumerate(POSITIONS):
            np.testing.assert_array_equal(sums[j], baseline.loc[position].reindex(SKILL_NAMES).selected_count)
    distributions = pd.DataFrame(rows)
    distributions.to_csv(out / 'opponent_position_conditional_distributions.csv', index=False)
    direction_rows = []
    for dataset in DATASETS:
        for j, position in enumerate(POSITIONS):
            for direction, indices in [('left', [0, 1]), ('center_safe', [2]), ('right', [3, 4])]:
                value = probabilities[dataset][j, indices].sum()
                low, high, reps = interval(draws[dataset][:, j, indices].sum(axis=-1))
                direction_rows.append(dict(dataset=dataset, opponent_position=position, direction=direction,
                    n_observations=int(totals[dataset][j]), probability=value,
                    probability_ci_low=low, probability_ci_high=high, valid_bootstrap_replicates=reps))
    pd.DataFrame(direction_rows).to_csv(out / 'opponent_position_direction_distributions.csv', index=False)
    pd.DataFrame(clusters_report).to_csv(out / 'opponent_position_visitation.csv', index=False)
    observations[['dataset', 'cluster', 'rally_cluster', 'attacker_id', 'defender_id', 'p1_position', 'p2_position', 'opponent_position', 'attack_skill']].to_csv(out / 'analysis_observations.csv', index=False)
    human = observations[observations.dataset == HUMAN_DATASET]
    human.to_csv(out / 'human_incoming_events.csv', index=False)
    export_human_player_counts(human, out)
    metric_rows = []; bootstrap_metrics = {}
    for policy in DATASETS[1:]:
        point = similarity(probabilities[HUMAN_DATASET], probabilities[policy])
        boot = similarity(draws[HUMAN_DATASET], draws[policy]); bootstrap_metrics[policy] = boot
        for j, position in enumerate(POSITIONS):
            row = dict(policy=policy, opponent_position=position, human_n=int(totals[HUMAN_DATASET][j]), robot_n=int(totals[policy][j]))
            for name, values in point.items():
                lo, hi, reps = interval(boot[name][:, j])
                row.update({name: float(values[j]), name + '_ci_low': lo, name + '_ci_high': hi,
                            name + '_valid_replicates': reps})
            metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows); metrics.to_csv(out / 'similarity_metrics.csv', index=False)
    winners = []
    for j, position in enumerate(POSITIONS):
        r = metrics[metrics.opponent_position == position].set_index('policy')
        delta = bootstrap_metrics['potential_ibr']['js_divergence_bits'][:, j] - bootstrap_metrics['nash_q']['js_divergence_bits'][:, j]
        low, high, reps = interval(delta)
        winners.append(dict(opponent_position=position, closer_by_js=r.js_divergence_bits.idxmin(),
            closer_by_tv=r.total_variation.idxmin(), js_ibr_minus_nash=float(r.loc['potential_ibr', 'js_divergence_bits'] - r.loc['nash_q', 'js_divergence_bits']),
            difference_ci_low=low, difference_ci_high=high, valid_replicates=reps,
            interval_excludes_zero=bool(low > 0 or high < 0)))
    pd.DataFrame(winners).to_csv(out / 'similarity_comparison.csv', index=False)
    plot_outputs(out, probabilities, draws, totals, metrics)
    validation = validate_references(args, joint, meta)
    paths += [Path(check['reference']) for check in validation if Path(check['reference']).exists()]
    paths += [Path(__file__), Path(__file__).with_name('spatial.py'),
              Path(__file__).with_name('analyze_spatial_v2.py'), Path('v3/analyze_conditional_skill_selection.py')]
    provenance = []
    for path in paths:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
        provenance.append(dict(path=str(path), sha256=digest.hexdigest(), bytes=path.stat().st_size))
    audit = dict(metadata=meta, exclusions=exclusions, reference_checks=validation, sources=provenance,
        seed=args.seed, bootstrap_replicates=args.bootstrap, totals={d: int(totals[d].sum()) for d in DATASETS},
        versions=dict(numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__))
    (out / 'audit.json').write_text(json.dumps(audit, indent=2))
    write_report(args, audit, clusters_report, metrics, winners)
    print(json.dumps(dict(totals=audit['totals'], comparisons=winners), indent=2))


def validate_references(args, joint, meta):
    checks = []
    sources = [] if args.human_events_dir else [(HUMAN_DATASET, args.human_analysis_dir / 'context_landing_statistics.csv', 'region_count')]
    for policy, directory, decisions in [
            ('nash_q', 'conditional_skill_selection_nash', args.nash_decisions),
            ('potential_ibr', 'conditional_skill_selection_potential_ibr', args.ibr_decisions)]:
        default = Path('skill_eval/v3') / ('nash_vs_random_10000' if policy == 'nash_q' else 'potential_ibr_vs_random_10000') / 'decisions.csv'
        if decisions.resolve() == default.resolve() and np.isclose(meta['center_band_m'], .05):
            sources.append((policy, Path('reports/v3') / directory / 'joint_position_skill_summary.csv', 'selected_count'))
    for dataset, path, column in sources:
        if not path.exists():
            checks.append(dict(dataset=dataset, reference=str(path), status='reference_missing')); continue
        f = pd.read_csv(path)
        if dataset == HUMAN_DATASET:
            f = f[f.dataset == dataset].rename(columns={'region': 'attack_skill'})
        else:
            f = f[(f.strategy == dataset) & (f.player == 'P1')]
        key = ['p1_position', 'p2_position', 'attack_skill']
        expected = f.set_index(key)[column].sort_index()
        actual = joint[joint.dataset == dataset].set_index(key).selected_count.sort_index()
        pd.testing.assert_series_equal(actual, expected, check_names=False, check_dtype=False)
        checks.append(dict(dataset=dataset, reference=str(path), status='all_45_counts_match', total=int(actual.sum())))
    return checks


def export_human_player_counts(human, out):
    for kind in ('joint', 'opponent'):
        rows = []
        for player, frame in human.groupby('attacker_id'):
            conditions = [(a, b) for a in POSITIONS for b in POSITIONS] if kind == 'joint' else [(p,) for p in POSITIONS]
            for condition in conditions:
                if kind == 'joint':
                    subset = frame[(frame.p1_position == condition[0]) & (frame.p2_position == condition[1])]
                    fields = dict(p1_position=condition[0], p2_position=condition[1])
                else:
                    subset = frame[frame.opponent_position == condition[0]]
                    fields = dict(opponent_position=condition[0])
                for skill in SKILL_NAMES:
                    count = int((subset.attack_skill == skill).sum())
                    rows.append(dict(attacker_id=int(player), **fields, attack_skill=skill,
                        n_observations=len(subset), count=count, probability=count / len(subset) if len(subset) else np.nan))
        pd.DataFrame(rows).to_csv(out / f'human_by_player_{kind}_distributions.csv', index=False)


def write_report(args, audit, visitation, metrics, winners):
    extra_cli = ('--human-events-dir '+str(args.human_events_dir)+' --human-players '+args.human_players+' ') if args.human_events_dir else ''
    rows = ['# Opponent-position-only attacking-category comparison',
        'This is an exploratory comparison of human **realized landing proxies** with robot **selected high-level skills**. It does not establish equivalent intentions or equivalent physical landing distributions.',
        '## Data and aggregation',
        'The selected human players use saved incoming-crossing poses, quality-screened valid landings, and the recorded four_quadrants_center_override configuration. Nash-Q and Potential IBR use only their own P1 attacker decisions against random, decoded actual pre-decision P2 gantry positions and logged selected skills. No extraction, evaluation, model training or simulation was run.',
        'The established robotic summarize() function is reused. For each defender-position condition, raw counts are pooled across attacker positions and selected human players, then normalized. No unweighted average of conditional probability vectors is used. All 45 original nine-context counts per dataset are checked against saved references when available.',
        'Player-relative lateral coordinates are +world-y for P1 and −world-y for P2. Center includes |normalized own lateral| <= '+str(audit['metadata']['center_band_m'])+'/0.7625. Human normalization uses its table half-width; robots use 0.7625 m. The previous shared pose-quality bound and incoming cohort are retained. P1 positions are used only for cohort consistency and the nine-to-three count validation, not to stratify the final distributions.',
        'Robot command direction names are audited against saved attack_target_local_y when available: left/left_short must have positive local y, right/right_short negative. This verifies the V3 acting-player-relative naming convention and prevents accidentally using legacy reversed names. Target coordinates are used only for this metadata check; the robot category is always the logged selected skill, and no realized landing is inferred.',
        'Human region configuration: `'+json.dumps(audit['metadata']['human_region'], sort_keys=True)+'`.',
        'Human cohort: `'+audit['metadata']['human_dataset']+'`. Source counts by player: `'+json.dumps(audit['metadata']['human_source_player_counts'])+'`. Eligible incoming counts by player: `'+json.dumps(audit['metadata']['human_incoming_counts_by_player'])+'`.',
        'For human P1 attacks the opponent is physical P2; for human P2 attacks the opponent is physical P1. The opponent is binned in their own-side coordinates. Physical joint pairs stay (P1,P2). Thus a pooled opponent condition is not simply a fixed P2 column of a pooled physical joint table. Both players from a human match share one bootstrap cluster. Per-player joint and opponent distributions are saved separately.',
        '## Counts and visitation',
        '| Dataset | Opponent position | n | Visit fraction | Contributing clusters |\n|---|---|---:|---:|---:|\n' + '\n'.join(
            f"| {NAMES[r['dataset']]} | {r['opponent_position']} | {r['n']} | {r['visitation_fraction']:.3%} | {r['n_clusters']} |" for r in visitation)]
    rows += ['Each nonempty probability row sums to one. Empty conditions have undefined probabilities. Unequal visitation is reported above; the three conditions are compared separately. These are observed, visitation-weighted mixtures over each dataset’s own attacker states, not matched-state causal responses to opponent position.',
        '## Similarity results',
        '| Opponent position | Policy | JS divergence (bits), 95% CI | TV, 95% CI | Spearman |\n|---|---|---:|---:|---:|\n' + '\n'.join(
            f'| {r.opponent_position} | {r.policy} | {r.js_divergence_bits:.4f} [{r.js_divergence_bits_ci_low:.4f}, {r.js_divergence_bits_ci_high:.4f}] | {r.total_variation:.4f} [{r.total_variation_ci_low:.4f}, {r.total_variation_ci_high:.4f}] | {r.spearman_rho:.3f} |' for r in metrics.itertuples())]
    rows += [f"At opponent **{w['opponent_position']}**, the smaller empirical JS divergence is **{w['closer_by_js']}** and the smaller TV distance is **{w['closer_by_tv']}**. Potential-IBR-minus-Nash JS difference = {w['js_ibr_minus_nash']:.4f}, bootstrap 95% interval [{w['difference_ci_low']:.4f}, {w['difference_ci_high']:.4f}]." for w in winners]
    rows += ['JS uses base-2 logarithms and is the divergence, not its square root (range 0–1 bits). TV is half the L1 distance (range 0–1). Spearman ranks the five category probabilities with average ranks for ties; constant vectors are undefined. With only five categories, Spearman is a limited descriptive statistic; no significance p-values or claims of equivalence are made.',
        '## Cluster uncertainty and limitations',
        f"The {args.bootstrap} bootstrap replicates use seed {args.seed}. Whole contributing human matches and whole contributing robot rallies are independently resampled with replacement, including all their defender-position conditions. Counts are pooled and renormalized in each replicate. Human draws are shared across the two policy comparisons, preserving their dependence for the difference interval. Probability/metric intervals are percentile 95% intervals; missing-context replicates are excluded and their valid replicate counts are saved. These intervals condition on the observed eligible sample and do not correct reconstruction error, category definitions, actor identities, cohort selection, one robot evaluation seed, or unobserved match/rally dependence. Robot rally bootstrap is not uncertainty across evaluation seeds or trained checkpoints.",
        'Human unsuccessful or unreconstructed shots are excluded, while robot selections include decisions whose shots may fail or never contact the ball. Foot-midpoint poses and gantry poses are different measurements. Human incoming alignment covers a selected subset of reconstructed returns. These measurement/selection differences limit direct comparison. Human match labels identify source matches; P1 does not imply one athlete across matches. Boundaries were not fitted to robot similarity.',
        '## Reproduction',
        '```bash\npython3 v3/human_comparison/analyze_opponent_position.py '+extra_cli+' \\\n  --human-analysis-dir '+str(args.human_analysis_dir)+' \\\n  --nash-decisions '+str(args.nash_decisions)+' \\\n  --ibr-decisions '+str(args.ibr_decisions)+' \\\n  --bootstrap '+str(args.bootstrap)+' --seed '+str(args.seed)+' \\\n  --output-dir '+str(args.output_dir)+'\n```',
        'audit.json records source hashes, model/evaluation metadata, exclusions, cluster settings and reference-count validation. analysis_observations.csv permits direct count reconstruction. All previous reports are preserved.',
        '## Figures']
    rows += [f'![{p.stem}]({p.name})' for p in sorted(args.output_dir.glob('*.png'))]
    (args.output_dir / 'report.md').write_text('\n\n'.join(rows) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--human-analysis-dir', type=Path, default=Path('v3/human_comparison/reports/spatial_comparison_v3_four_quadrants_center_override'))
    p.add_argument('--nash-decisions', type=Path, default=Path('skill_eval/v3/nash_vs_random_10000/decisions.csv'))
    p.add_argument('--ibr-decisions', type=Path, default=Path('skill_eval/v3/potential_ibr_vs_random_10000/decisions.csv'))
    p.add_argument('--human-events-dir', type=Path, help='Saved raw event folder; never reads original trajectories')
    p.add_argument('--human-players', choices=['p1', 'p2', 'both'], default='p1')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.human_players != 'p1' and not args.human_events_dir: p.error('P2/both require --human-events-dir')
    if args.output_dir is None:
        suffix = '_' + args.human_events_dir.name + ('_' + args.human_players if args.human_players != 'both' else '') if args.human_events_dir else ''
        args.output_dir = Path('v3/human_comparison/reports/opponent_position_comparison' + suffix)
    if args.human_events_dir and args.output_dir.resolve() == Path('v3/human_comparison/reports/opponent_position_comparison').resolve():
        p.error('Use a new output directory for the new human cohort')
    if args.bootstrap < 100: p.error('--bootstrap must be at least 100')
    run(args)


if __name__ == '__main__': main()
