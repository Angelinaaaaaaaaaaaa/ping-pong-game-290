#!/usr/bin/env python3
"""Nine-context human landing-proxy versus robot selected-skill comparison.

Uses the exact saved eligible observations of analyze_opponent_position.py.
No trajectory extraction, policy evaluation or training is performed.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v3.human_comparison.analyze_opponent_position import (
    DATASETS, NAMES, HUMAN_DATASET, configure_human_cohort, LABELS, COLORS, POSITIONS, SKILL_NAMES,
    normalize, bootstrap, interval, similarity, save_plot,
)

PAIRS = tuple(itertools.product(POSITIONS, repeat=2))
PAIR_NAMES = tuple(a[0].upper() + b[0].upper() for a, b in PAIRS)


def joint_cluster_counts(frame):
    clusters = pd.Index(sorted(frame.cluster.unique()))
    pairs = pd.MultiIndex.from_tuples(PAIRS)
    indices = pairs.get_indexer(pd.MultiIndex.from_frame(frame[['p1_position', 'p2_position']]))
    skills = pd.Index(SKILL_NAMES).get_indexer(frame.attack_skill)
    if (indices < 0).any() or (skills < 0).any():
        raise ValueError('Unknown position or attack category')
    counts = np.zeros((len(clusters), 9, 5), dtype=np.int64)
    np.add.at(counts, (clusters.get_indexer(frame.cluster), indices, skills), 1)
    return clusters, counts


def draw_figures(out, probabilities, draws, totals, metrics):
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), layout='constrained')
    for j, (ax, pair) in enumerate(zip(axes.flat, PAIRS)):
        for i, dataset in enumerate(DATASETS):
            x = np.arange(5) + (i - 1) * .25
            ax.bar(x, probabilities[dataset][j], .24, color=COLORS[i], label=NAMES[dataset])
            # Undefined bootstrap rows occur if a sparse context is not sampled.
            for k, xp in enumerate(x):
                lo, hi, _ = interval(draws[dataset][:, j, k])
                ax.vlines(xp, lo, hi, color='black', lw=.8)
        ax.set_xticks(range(5), LABELS, rotation=30, ha='right')
        ax.set_ylim(0, 1); ax.set_ylabel('Conditional probability')
        ns = ', '.join(f'{name}={int(totals[d][j])}' for d, name in
                       zip(DATASETS, ['Human', 'Nash', 'IBR']))
        ax.set_title(f'P1 {pair[0]} / P2 {pair[1]}\n{ns}', fontsize=10)
        if totals[HUMAN_DATASET][j] < 30:
            ax.text(.02, .85, 'Sparse human context', transform=ax.transAxes, fontsize=8, color='#a02020')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Nine joint-position contexts: human landing proxies versus selected robot skills\n95% match/rally cluster-bootstrap intervals; pairs ordered (P1, P2)')
    save_plot(fig, out / 'joint_position_grouped_bars.png')

    fig, axes = plt.subplots(1, 3, figsize=(16, 8), layout='constrained')
    for ax, dataset in zip(axes, DATASETS):
        im = ax.imshow(probabilities[dataset], cmap='Blues', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(5), LABELS, rotation=35, ha='right')
        ax.set_yticks(range(9), [f'{p} (n={int(n)})' for p, n in zip(PAIR_NAMES, totals[dataset])])
        ax.set_title(NAMES[dataset]); ax.set_ylabel('(P1, P2) position')
        for j, k in itertools.product(range(9), range(5)):
            v = probabilities[dataset][j, k]
            ax.text(k, j, f'{v:.3f}' if np.isfinite(v) else 'NA', ha='center', va='center',
                    color='white' if v > .5 else 'black', fontsize=8)
    fig.colorbar(im, ax=list(axes), label='Conditional probability', shrink=.8)
    save_plot(fig, out / 'joint_position_probability_heatmaps.png')

    fig, axes = plt.subplots(3, 2, figsize=(11, 13), layout='constrained')
    for i, (metric, title, low, high, cmap) in enumerate([
            ('js_divergence_bits', 'JS divergence (bits; lower is closer)', 0, 1, 'viridis'),
            ('spearman_rho', 'Spearman correlation (higher rank agreement)', -1, 1, 'coolwarm'),
            ('total_variation', 'Total variation distance', 0, 1, 'viridis')]):
        for j, policy in enumerate(DATASETS[1:]):
            ax = axes[i, j]
            m = metrics[metrics.policy == policy].set_index(['p1_position', 'p2_position']).reindex(PAIRS)
            values = m[metric].to_numpy().reshape(3, 3)
            im = ax.imshow(values, vmin=low, vmax=high, cmap=cmap)
            ax.set_xticks(range(3), ['P2 ' + p.title() for p in POSITIONS])
            ax.set_yticks(range(3), ['P1 ' + p.title() for p in POSITIONS])
            ax.set_title(NAMES[policy] + '\n' + title, fontsize=10)
            for a, b in itertools.product(range(3), repeat=2):
                value = values[a, b]; n = int(m.human_n.iloc[a * 3 + b])
                ax.text(b, a, (f'{value:.3f}' if np.isfinite(value) else 'NA') + f'\nhuman n={n}',
                        ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=.65, edgecolor='none'))
        fig.colorbar(im, ax=list(axes[i]), shrink=.8)
    save_plot(fig, out / 'joint_position_similarity_heatmaps.png')


def run(args):
    global HUMAN_DATASET, DATASETS, NAMES
    source = args.input_dir
    out = args.output_dir
    if out.resolve() == source.resolve(): raise ValueError('Use a separate output directory')
    frame = pd.read_csv(source / 'analysis_observations.csv')
    audit = json.loads((source / 'audit.json').read_text())
    HUMAN_DATASET = audit.get('metadata', {}).get('human_dataset', 'human_p1')
    DATASETS, NAMES = configure_human_cohort(HUMAN_DATASET)
    reference = pd.read_csv(source / 'nine_context_raw_counts.csv')
    if set(frame.dataset) != set(DATASETS): raise ValueError('Require exactly the three comparison datasets')
    if frame[['cluster', 'rally_cluster', 'p1_position', 'p2_position', 'attack_skill']].isna().any().any():
        raise ValueError('Incomplete saved observations')
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    probabilities = {}; draws = {}; totals = {}; rows = []; visits = []
    for dataset in DATASETS:
        subset = frame[frame.dataset == dataset]
        clusters, counts = joint_cluster_counts(subset)
        sums = counts.sum(axis=0); totals[dataset] = sums.sum(axis=1)
        if int(sums.sum()) != audit['totals'][dataset]: raise ValueError('Source audit total mismatch')
        expected = reference[reference.dataset == dataset].set_index(['p1_position', 'p2_position', 'attack_skill']).sort_index()
        for j, pair in enumerate(PAIRS):
            np.testing.assert_array_equal(sums[j], expected.loc[pair].reindex(SKILL_NAMES).selected_count)
        probabilities[dataset] = normalize(sums)
        print(f'{dataset}: n={len(subset)}, clusters={len(clusters)}, nine-context bootstrap={args.bootstrap}', flush=True)
        draws[dataset] = bootstrap(counts, args.bootstrap, rng)
        for j, pair in enumerate(PAIRS):
            nc = int((counts[:, j, :].sum(axis=1) > 0).sum())
            common = dict(dataset=dataset, p1_position=pair[0], p2_position=pair[1],
                n_observations=int(totals[dataset][j]), n_clusters=nc,
                visitation_fraction=float(totals[dataset][j] / len(subset)))
            visits.append(dict(**common, total_clusters=len(clusters), cluster_unit='match' if dataset == HUMAN_DATASET else 'rally'))
            for k, skill in enumerate(SKILL_NAMES):
                lo, hi, reps = interval(draws[dataset][:, j, k])
                rows.append(dict(**common, attack_skill=skill, count=int(sums[j, k]),
                    probability=probabilities[dataset][j, k], probability_ci_low=lo, probability_ci_high=hi,
                    valid_bootstrap_replicates=reps,
                    measurement='realized_landing_proxy' if dataset == HUMAN_DATASET else 'selected_high_level_skill'))
    pd.DataFrame(rows).to_csv(out / 'joint_position_conditional_distributions.csv', index=False)
    pd.DataFrame(visits).to_csv(out / 'joint_position_visitation.csv', index=False)
    metric_rows = []; boot_metrics = {}
    for policy in DATASETS[1:]:
        point = similarity(probabilities[HUMAN_DATASET], probabilities[policy])
        boot_metrics[policy] = similarity(draws[HUMAN_DATASET], draws[policy])
        for j, pair in enumerate(PAIRS):
            row = dict(policy=policy, p1_position=pair[0], p2_position=pair[1],
                human_n=int(totals[HUMAN_DATASET][j]), robot_n=int(totals[policy][j]),
                sparse_human_context=bool(totals[HUMAN_DATASET][j] < 30))
            for metric, values in point.items():
                lo, hi, reps = interval(boot_metrics[policy][metric][:, j])
                row.update({metric: float(values[j]), metric + '_ci_low': lo,
                            metric + '_ci_high': hi, metric + '_valid_replicates': reps})
            metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / 'similarity_metrics.csv', index=False)
    comparisons = []
    for j, pair in enumerate(PAIRS):
        cell = metrics[(metrics.p1_position == pair[0]) & (metrics.p2_position == pair[1])].set_index('policy')
        delta = boot_metrics['potential_ibr']['js_divergence_bits'][:, j] - boot_metrics['nash_q']['js_divergence_bits'][:, j]
        lo, hi, reps = interval(delta)
        finite = cell.js_divergence_bits.dropna()
        comparisons.append(dict(p1_position=pair[0], p2_position=pair[1], human_n=int(totals[HUMAN_DATASET][j]),
            closer_by_js=finite.idxmin() if len(finite) == 2 and finite.nunique() > 1 else 'tie_or_undefined',
            js_ibr_minus_nash=float(cell.loc['potential_ibr', 'js_divergence_bits'] - cell.loc['nash_q', 'js_divergence_bits']),
            difference_ci_low=lo, difference_ci_high=hi, valid_replicates=reps,
            interval_excludes_zero=bool(lo > 0 or hi < 0)))
    pd.DataFrame(comparisons).to_csv(out / 'similarity_comparison.csv', index=False)
    draw_figures(out, probabilities, draws, totals, metrics)
    provenance = []
    for path in [source / 'analysis_observations.csv', source / 'audit.json', source / 'nine_context_raw_counts.csv',
                 Path(__file__), Path(__file__).with_name('analyze_opponent_position.py')]:
        provenance.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result_audit = dict(source=str(source), source_audit=audit, sources=provenance, seed=args.seed,
        bootstrap_replicates=args.bootstrap, reference_check='all 135 nine-context counts match',
        totals={d: int(totals[d].sum()) for d in DATASETS},
        versions=dict(numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__))
    (out / 'audit.json').write_text(json.dumps(result_audit, indent=2))
    write_report(args, metrics, comparisons, visits)
    print(metrics[['p1_position', 'p2_position', 'policy', 'human_n', 'robot_n', 'js_divergence_bits', 'spearman_rho']].to_string(index=False))


def write_report(args, metrics, comparisons, visits):
    totals = pd.DataFrame(visits).groupby('dataset').n_observations.sum()
    table = ['| P1 | P2 | Human n | Nash n | IBR n | JS: Nash | JS: IBR | Spearman: Nash | Spearman: IBR |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for pair in PAIRS:
        cell = metrics[(metrics.p1_position == pair[0]) & (metrics.p2_position == pair[1])].set_index('policy')
        n = cell.loc['nash_q']; p = cell.loc['potential_ibr']
        table.append(f'| {pair[0]} | {pair[1]} | {n.human_n} | {n.robot_n} | {p.robot_n} | {n.js_divergence_bits:.4f} | {p.js_divergence_bits:.4f} | {n.spearman_rho:.3f} | {p.spearman_rho:.3f} |')
    rows = ['# Human–robot attacking-category similarity in nine position contexts',
        'Pairs are ordered **(physical P1 position, physical P2 position)**, with each position defined relative to that player’s own side. L/C/R means Left/Center/Right. Each five-category distribution is conditioned on both positions.',
        'The exact eligible observations from the opponent-only analysis are reused: incoming-aligned quality-screened selected-human-player realized-landing proxies (four_quadrants_center_override) and logged selected high-level skills for Nash-Q P1 and Potential IBR P1 against random. Totals are '+f"{totals[HUMAN_DATASET]:,} human landings, {totals['nash_q']:,} Nash-Q decisions and {totals['potential_ibr']:,} Potential IBR decisions."+' No reconstruction or evaluations were rerun. Source metadata, source hashes and all original count validations are retained in audit.json.',
        'Human cohort: `'+HUMAN_DATASET+'`. When P1 and P2 human shots are pooled, pair labels remain physical (P1,P2), not (hitter,defender). Human P2 categories are already hitter-relative and are not mirrored a second time. Consequently a physical joint context pools different attacker roles; it should not be interpreted as an exactly matched hitter/defender context against P1-only robots. The opponent-only analysis uses each shot’s actual defender instead. Separate per-player human distributions are available in the source opponent-analysis directory.',
        'All 135 raw category counts are checked against the original nine-context counts. Each nonempty context is normalized by its own observation count; empty distributions are undefined. Context visitation is reported separately. Category order is Left, Left Short, Center Safe, Right Short, Right.',
        '## Jensen–Shannon divergence and correlation', '\n'.join(table),
        'JS is base-2 Jensen–Shannon **divergence**, not the square-root distance; lower is closer (0–1 bits). Correlation is **Spearman**, matching the opponent-only analysis: average ranks for ties across the five probabilities, undefined for constant vectors. Higher means more similar category rankings; rank correlation is not probability calibration. Total variation is also included in similarity_metrics.csv.',
        '## Uncertainty and interpretation',
        f'{args.bootstrap} bootstrap replicates use seed {args.seed}. Whole human matches and whole robot rallies are resampled globally, preserving every joint context in each sampled cluster, then renormalized per context. The same human bootstrap draws enter both robot comparisons. Marginal probability/metric percentile 95% intervals, valid replicate counts and contributing-cluster counts are saved in CSV. Empty bootstrap contexts and constant-rank correlations remain undefined and are excluded from that metric’s interval. Robot policies are resampled independently; these intervals do not model paired common-seed dependence between policy runs or training/evaluation-seed variation.',
        'Human n < 30 is flagged as sparse (a display/review threshold, not a guarantee that larger cells are reliable). Inspect context sample counts before interpreting a correlation or divergence. Bootstrap uncertainty does not fix reconstruction error, region-definition uncertainty, cohort selection, or the distinction between successful human landing proxies and all robot skill selections. Results are exploratory, not evidence of equivalent intentions or realized robot placement.',
        '## Relative JS comparison']
    for r in comparisons:
        rows.append(f"{r['p1_position']}/{r['p2_position']} (human n={r['human_n']}): smaller empirical JS = **{r['closer_by_js']}**; IBR-minus-Nash JS 95% interval [{r['difference_ci_low']:.4f}, {r['difference_ci_high']:.4f}]. An interval containing zero does not resolve the ordering.")
    rows += ['## Reproduce',
        '```bash\npython3 v3/human_comparison/analyze_joint_position.py \\\n  --input-dir '+str(args.input_dir)+' \\\n  --bootstrap '+str(args.bootstrap)+' --seed '+str(args.seed)+' \\\n  --output-dir '+str(args.output_dir)+'\n```',
        'Previous analyses are preserved. The source opponent-position report contains the detailed coordinate/classifier audit.']
    rows += [f'![{p.stem}]({p.name})' for p in sorted(args.output_dir.glob('*.png'))]
    (args.output_dir / 'report.md').write_text('\n\n'.join(rows) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=Path('v3/human_comparison/reports/opponent_position_comparison'))
    p.add_argument('--output-dir', type=Path, default=Path('v3/human_comparison/reports/joint_position_comparison'))
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.bootstrap < 100: p.error('--bootstrap must be at least 100')
    run(args)


if __name__ == '__main__': main()
