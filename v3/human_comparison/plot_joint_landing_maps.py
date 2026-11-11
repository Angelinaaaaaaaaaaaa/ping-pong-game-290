#!/usr/bin/env python3
"""Plot saved human landings using the complete four-quadrant/center partition.

Audit the existing both-player comparisons against freshly classified saved
coordinates, retain their unchanged statistics, and add nine-context dot maps.
"""
import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v3.human_comparison.analyze_spatial_v2 import background, partition_outlines, COLORS
from v3.human_comparison.events import HALF_WIDTH
from v3.human_comparison.spatial import region_from_dict, position_labels
from nash_skills.skills import SKILL_NAMES

POSITIONS = ('left', 'center', 'right')
PAIRS = tuple(itertools.product(POSITIONS, repeat=2))
LABELS = ('Left', 'Left Short', 'Center Safe', 'Right Short', 'Right')
EDGES = [np.linspace(0, 1, 21), np.linspace(-1, 1, 21)]


def assign_categories(frame, region, band, pose_bound):
    f = frame.copy()
    f['human_landing_category'] = region.classify(f.depth, f.lateral)
    if not f.human_landing_category.isin(SKILL_NAMES).all():
        raise ValueError('Invalid coordinates in accepted landing cohort')
    f['classification_method'] = region.name
    f['baseline_nearest_target_proxy'] = f.human_skill_proxy
    f['context_pose_quality_ok'] = (f[['p1_own_lateral', 'p2_own_lateral']].abs() <= pose_bound).all(axis=1)
    for player in (1, 2):
        f[f'p{player}_position'] = position_labels(f[f'p{player}_own_lateral'], band)
        f.loc[~f.context_pose_quality_ok, f'p{player}_position'] = 'unknown'
    f['opponent_position'] = np.where(f.attacker_id == 1, f.p2_position, f.p1_position)
    return f


def subset(frame, pair):
    return frame[(frame.p1_position == pair[0]) & (frame.p2_position == pair[1])]


def draw_panel(ax, frame, pair, region, density, vmax):
    f = subset(frame, pair)
    background(ax, region)
    if density and len(f) >= 20:
        mass, _, _ = np.histogram2d(f.depth, f.lateral, bins=EDGES)
        ax.pcolormesh(*EDGES, (mass / len(f)).T, cmap='magma', vmin=0, vmax=vmax, shading='auto', rasterized=True)
        partition_outlines(ax, region, color='white')
    elif len(f):
        ax.scatter(f.depth, f.lateral, s=4, color='black', alpha=.24, linewidths=0, rasterized=True)
        if density: ax.text(.02, .02, 'Sparse: dots, no smoothing', transform=ax.transAxes, fontsize=7)
    else:
        ax.text(.5, .5, 'No observed landings', transform=ax.transAxes, ha='center', fontsize=9)
    n1 = int((f.attacker_id == 1).sum()); n2 = int((f.attacker_id == 2).sum())
    ax.set_title(f'P1 {pair[0].title()} / P2 {pair[1].title()}\nn={len(f)} (P1 shots={n1}, P2 shots={n2})', fontsize=10)
    ax.set(xlabel='Depth: 0 net → 1 endline', ylabel='Hitter lateral: −1 right → +1 left')


def save_grid(frame, region, out, name, description, density=False, vmax=.05):
    fig, axes = plt.subplots(3, 3, figsize=(12, 11), layout='constrained')
    for ax, pair in zip(axes.flat, PAIRS):
        draw_panel(ax, frame, pair, region, density, vmax)
    fig.suptitle(description + '\nRows = physical P1 position; columns = physical P2 position', fontsize=13)
    if density:
        fig.colorbar(ScalarMappable(norm=Normalize(0, vmax), cmap='magma'), ax=list(axes.flat),
            label='Probability mass per 20×20 bin within context', shrink=.75, pad=.02)
    else:
        fig.legend([Patch(facecolor=COLORS[i], alpha=.35) for i in range(5)], LABELS,
            loc='lower center', bbox_to_anchor=(.5, -.025), ncol=5, fontsize=9)
    fig.savefig(out / f'{name}.png', dpi=300, bbox_inches='tight'); plt.close(fig)


def verify_comparison_counts(incoming, opponent_table, joint_table):
    for condition, reference in [(['opponent_position'], opponent_table),
                                 (['p1_position', 'p2_position'], joint_table)]:
        expected = reference[reference.dataset == 'human_both'].set_index([*condition, 'attack_skill'])['count'].sort_index()
        actual = incoming.groupby([*condition, 'human_landing_category']).size()
        actual = actual.reindex(expected.index, fill_value=0).sort_index()
        pd.testing.assert_series_equal(actual, expected, check_names=False, check_dtype=False)


def run(args):
    out = args.output_dir
    if out.resolve() in {p.resolve() for p in [args.human_dir, args.analysis_dir, args.joint_analysis_dir]}:
        raise ValueError('Choose a separate output directory')
    audit = json.loads((args.analysis_dir / 'audit.json').read_text())
    meta = audit['metadata']; region = region_from_dict(meta['human_region']); region.validate()
    if region.name != 'four_quadrants_center_override':
        raise ValueError('Expected the complete four-quadrant/center-override method')
    band = meta['center_band_m']; pose_bound = meta['max_abs_position_normalized']
    incoming = pd.read_csv(args.analysis_dir / 'human_incoming_events.csv', low_memory=False)
    incoming = assign_categories(incoming, region, band, pose_bound)
    if not incoming.context_pose_quality_ok.all(): raise ValueError('Unexpected invalid incoming context')
    if not incoming.human_landing_category.eq(incoming.attack_skill).all():
        raise ValueError('Existing comparison labels differ from explicit quadrant classification')
    raw = pd.read_csv(args.human_dir / 'shot_events.csv', low_memory=False)
    outgoing = raw[raw.classified & raw.attacker_id.isin([1, 2])].copy()
    outgoing['depth'] = outgoing.landing_depth_normalized
    outgoing['lateral'] = outgoing.landing_lateral_normalized
    outgoing['p1_own_lateral'] = outgoing.p1_crossing_y_m / HALF_WIDTH
    outgoing['p2_own_lateral'] = -outgoing.p2_crossing_y_m / HALF_WIDTH
    outgoing['position_time'] = 'outgoing_crossing'
    outgoing = assign_categories(outgoing, region, band, pose_bound)
    opponent_table = pd.read_csv(args.analysis_dir / 'opponent_position_conditional_distributions.csv')
    joint_table = pd.read_csv(args.joint_analysis_dir / 'joint_position_conditional_distributions.csv')
    verify_comparison_counts(incoming, opponent_table, joint_table)
    out.mkdir(parents=True, exist_ok=True)
    keep = ['rally_id', 'shot_index', 'attacker_id', 'position_time', 'depth', 'lateral',
        'p1_own_lateral', 'p2_own_lateral', 'p1_position', 'p2_position', 'opponent_position',
        'human_landing_category', 'classification_method', 'baseline_nearest_target_proxy', 'context_pose_quality_ok']
    pd.concat([incoming[keep], outgoing[keep]], ignore_index=True).to_csv(out / 'human_landings_explicit_rectangular_labels.csv', index=False)
    (out / 'region_definition.json').write_text(json.dumps(asdict(region), indent=2))
    cohorts = []
    for timing, frame in [('incoming', incoming), ('outgoing', outgoing)]:
        for player, label in [(None, 'both_players'), (1, 'p1'), (2, 'p2')]:
            f = frame if player is None else frame[frame.attacker_id == player]
            cohorts.append((timing, label, f))
    vmax = 0.
    for _, _, f in cohorts:
        for pair in PAIRS:
            s = subset(f, pair)
            if len(s) >= 20:
                hist, _, _ = np.histogram2d(s.depth, s.lateral, bins=EDGES)
                vmax = max(vmax, float(hist.max() / len(s)))
    counts = []; density_rows = []
    for timing, label, frame in cohorts:
        description = ('PRIMARY: incoming decision context' if timing == 'incoming' else 'BASELINE: outgoing crossing context') + ' | ' + label.replace('_', ' ')
        for density in (False, True):
            save_grid(frame, region, out, f'human_{timing}_{label}_nine_context_'+('density' if density else 'landing_marks'), description, density, vmax)
        for pair in PAIRS:
            f = subset(frame, pair)
            for category in SKILL_NAMES:
                count = int((f.human_landing_category == category).sum())
                counts.append(dict(position_time=timing, player_cohort=label, p1_position=pair[0],
                    p2_position=pair[1], n=len(f), category=category, count=count,
                    probability=count / len(f) if len(f) else np.nan))
            if label == 'both_players':
                hist, _, _ = np.histogram2d(f.depth, f.lateral, bins=EDGES)
                for i, j in itertools.product(range(20), repeat=2):
                    density_rows.append(dict(position_time=timing, p1_position=pair[0], p2_position=pair[1],
                        depth_bin=i, lateral_bin=j, n=len(f), count=int(hist[i, j]), probability=hist[i, j]/len(f) if len(f) else np.nan))
    pd.DataFrame(counts).to_csv(out / 'human_nine_context_region_distributions.csv', index=False)
    pd.DataFrame(density_rows).to_csv(out / 'density_bin_counts_and_probabilities.csv', index=False)
    for pair in PAIRS:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout='constrained')
        for ax, player, title in zip(axes, [None, 1, 2], ['Both human players', 'Human P1 shots', 'Human P2 shots']):
            f = incoming if player is None else incoming[incoming.attacker_id == player]
            draw_panel(ax, f, pair, region, False, vmax); ax.set_title(title + '\n' + ax.get_title(), fontsize=10)
        fig.suptitle('Incoming-aligned realized landings | four rectangles + Center Safe override')
        fig.savefig(out / f'context_{pair[0]}_{pair[1]}_landing_marks_by_player.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for ax in axes:
        background(ax, region); ax.set(xlabel='Depth: 0 net → 1 endline', ylabel='Hitter lateral: −1 right → +1 left')
    axes[0].set_title('Explicit five-region partition — no nearest target')
    for x, y, text in [(.10, .7, 'Left Short'), (.82, .7, 'Left'), (.45, 0, 'Center Safe'),
                       (.10, -.7, 'Right Short'), (.82, -.7, 'Right')]:
        axes[0].text(x, y, text, ha='center', fontsize=10)
    axes[1].scatter(outgoing.depth, outgoing.lateral, color='black', s=2, alpha=.08, linewidths=0, rasterized=True)
    axes[1].set_title(f'All {len(outgoing):,} valid reconstructed landings (P1+P2)')
    fig.savefig(out / 'four_rectangles_center_override_and_landings.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    copies = []
    for scope, source in [('opponent', args.analysis_dir), ('joint', args.joint_analysis_dir)]:
        files = ['similarity_metrics.csv', 'similarity_comparison.csv']
        files += ['opponent_position_conditional_distributions.csv', 'opponent_position_grouped_bars.png', 'opponent_position_probability_heatmaps.png'] if scope == 'opponent' else ['joint_position_conditional_distributions.csv', 'joint_position_grouped_bars.png', 'joint_position_similarity_heatmaps.png']
        for name in files:
            path = source / name; target = out / (f'{scope}_' + name if name.startswith('similarity') else name)
            shutil.copyfile(path, target); copies.append(path)
    summary = dict(classification=asdict(region), classifier_verified_against_saved_analysis=True,
        all_valid_landings=len(outgoing), incoming_valid_landings=len(incoming),
        incoming_by_player={str(k):int(v) for k,v in incoming.groupby('attacker_id').size().items()},
        outgoing_unplottable_contexts=int((~outgoing.context_pose_quality_ok).sum()),
        changed_from_nearest_all_valid=int((outgoing.human_landing_category != outgoing.baseline_nearest_target_proxy).sum()),
        changed_from_nearest_incoming=int((incoming.human_landing_category != incoming.baseline_nearest_target_proxy).sum()),
        shared_density_vmax=vmax, center_band_m=band, pose_quality_bound=pose_bound,
        input_sha256=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in
            [args.human_dir/'shot_events.csv', args.analysis_dir/'human_incoming_events.csv', args.analysis_dir/'audit.json',
             args.joint_analysis_dir/'audit.json', Path(__file__), *copies]])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    write_report(args, summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='input_sha256'}, indent=2))


def write_report(args, summary):
    out = args.output_dir
    rows = ['# Four rectangles with Center Safe override: both-player landing maps',
        '**The latest both-player comparisons already used this rectangular method. They did not use nearest-target labels.** The original 10k_both/shot_events.csv retains its nearest-target baseline in human_skill_proxy. This report recomputes the rectangular labels from stored coordinates, verifies all 45 joint and 15 opponent human category counts, and copies the unchanged comparison statistics. No extraction, training or simulations were run.',
        '## Exact rule',
        'Outside Center Safe: y ≥ 0 and d < 0.51 is Left Short; y ≥ 0 and d ≥ 0.51 is Left; y < 0 and d < 0.51 is Right Short; y < 0 and d ≥ 0.51 is Right. The inclusive rectangle 0.20 ≤ d ≤ 0.70 and −0.35 ≤ y ≤ 0.35 always overrides those quadrants. Every valid landing gets exactly one of five labels. These are realized landing proxies, not annotated intentions.',
        'Exact saved configuration: `'+json.dumps(summary['classification'], sort_keys=True)+'`.',
        '## Cohorts and coordinates',
        f"Primary maps use {summary['incoming_valid_landings']:,} incoming-aligned, pose-quality-screened landings: {summary['incoming_by_player']}. The outgoing baseline starts from all {summary['all_valid_landings']:,} accepted reconstructed landings; {summary['outgoing_unplottable_contexts']} lack a plausible outgoing position context under the shared pose bound and are not plotted in a nine-cell context. All accepted landing points still appear in the unconditioned region figure and explicit-label CSV.",
        'For both humans, depth is normalized opponent-half distance from net (0) to endline (1), and lateral +1 is the hitter’s own left, −1 their right. P2 landings were already mirrored into this hitter-relative convention; they are not flipped again. Pair labels always mean physical (P1 position,P2 position), each defined on that player’s own side. Thus pooling human P1+P2 in a physical pair mixes attacker roles; separate P1/P2 maps are included. Opponent-only conditioning uses physical P2 when P1 hits and physical P1 when P2 hits.',
        f"Player-position bins use the same normalized center half-band {summary['center_band_m']}/0.7625, with |normalized lateral|>{summary['pose_quality_bound']} excluded from contexts. Black marks are observed reconstructed landings, not nominal targets. Density plots use 20×20 bins and one shared scale (probability mass per bin within a context); cells below 20 observations show dots, with no smoothing.",
        '## Classification audit',
        f"Relative to the retained nearest-target baseline, {summary['changed_from_nearest_all_valid']:,}/{summary['all_valid_landings']:,} full-cohort labels and {summary['changed_from_nearest_incoming']:,}/{summary['incoming_valid_landings']:,} incoming-cohort labels differ. The explicit CSV uses human_landing_category for the rectangular label and baseline_nearest_target_proxy for the old label, so the two cannot be confused.",
        '## Same comparison analysis',
        'Copied CSVs include all probabilities, JS divergence in bits, Spearman correlations, TV distances, counts, and the existing 2,000-replicate match/rally bootstrap intervals. The results are unchanged because recomputed rectangular counts match the previously analyzed counts exactly. Robot categories remain logged selected skills; no robot physical landings are inferred.',
        '### Opponent-only JS and correlation', metric_table(pd.read_csv(out/'opponent_similarity_metrics.csv'), ['opponent_position']),
        '### Nine physical-pair JS and correlation', metric_table(pd.read_csv(out/'joint_similarity_metrics.csv'), ['p1_position', 'p2_position']),
        '## Reproduce',
        '```bash\npython3 v3/human_comparison/plot_joint_landing_maps.py\n```',
        'CLI options --human-dir, --analysis-dir, --joint-analysis-dir and --output-dir select saved inputs and a separate output directory. Previous outputs are untouched.',
        '## Primary pooled landing maps',
        '![Incoming landing marks](human_incoming_both_players_nine_context_landing_marks.png)',
        '![Incoming density](human_incoming_both_players_nine_context_density.png)',
        '![Regions](four_rectangles_center_override_and_landings.png)',
        'Separate P1/P2, outgoing baseline and nine per-context figures are saved alongside these maps.']
    (out/'report.md').write_text('\n\n'.join(rows)+'\n')


def metric_table(metrics, conditions):
    rows = ['| Context | Policy | Human n | JS divergence (bits) | Spearman ρ |', '|---|---|---:|---:|---:|']
    for _, r in metrics.iterrows():
        name='/'.join(str(r[c]) for c in conditions)
        rows.append(f'| {name} | {r.policy} | {int(r.human_n)} | {r.js_divergence_bits:.4f} | {r.spearman_rho:.3f} |')
    return '\n'.join(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--human-dir',type=Path,default=Path('v3/human_comparison/reports/10k_both'))
    p.add_argument('--analysis-dir',type=Path,default=Path('v3/human_comparison/reports/opponent_position_comparison_10k_both'))
    p.add_argument('--joint-analysis-dir',type=Path,default=Path('v3/human_comparison/reports/joint_position_comparison_10k_both'))
    p.add_argument('--output-dir',type=Path,default=Path('v3/human_comparison/reports/four_quadrants_10k_both_landing_maps'))
    run(p.parse_args())


if __name__=='__main__': main()
