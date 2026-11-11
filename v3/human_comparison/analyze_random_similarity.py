#!/usr/bin/env python3
"""Human-proxy similarity versus learned and random selected-skill baselines.

Uses saved both-player human observations and the two existing 10k robot runs.
Random P2 attacks from each opponent run remain separate empirical baselines.
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
from v3.analyze_conditional_skill_selection import reconstruct_gantries, probabilities
from v3.human_comparison.spatial import position_labels, region_from_dict
from v3.human_comparison.analyze_opponent_position import normalize, similarity, interval
from nash_skills.skills import SKILL_NAMES

POSITIONS = ('left', 'center', 'right')
PAIRS = tuple(itertools.product(POSITIONS, repeat=2))
LEARNED = ('nash_q', 'potential_ibr')
RANDOM = ('random_p2_vs_nash_q', 'random_p2_vs_potential_ibr')
UNIFORM = 'random_uniform_exact'
MODELS = (*LEARNED, *RANDOM, UNIFORM)
NAMES = dict(nash_q='Nash-Q P1', potential_ibr='Potential IBR P1',
    random_p2_vs_nash_q='Random P2\nvs Nash-Q', random_p2_vs_potential_ibr='Random P2\nvs Potential IBR',
    random_uniform_exact='Exact uniform\nrandom')
SCOPES = ('joint_physical', 'opponent_only', 'joint_role_relative')
METRICS = ('js_divergence_bits', 'total_variation', 'spearman_rho', 'pearson_r')


def all_metrics(p, q):
    result = similarity(p, q)
    p, q = np.broadcast_arrays(np.asarray(p, float), np.asarray(q, float))
    a = p - p.mean(axis=-1, keepdims=True); b = q - q.mean(axis=-1, keepdims=True)
    scale = np.sqrt((a*a).sum(axis=-1) * (b*b).sum(axis=-1))
    result['pearson_r'] = np.divide((a*b).sum(axis=-1), scale,
        out=np.full(scale.shape, np.nan), where=scale > 1e-15)
    return result


def context_indices(frame):
    """Physical pairs, actual defender only, then (hitter,defender) pairs."""
    p1 = pd.Index(POSITIONS).get_indexer(frame.p1_position)
    p2 = pd.Index(POSITIONS).get_indexer(frame.p2_position)
    if (p1 < 0).any() or (p2 < 0).any(): raise ValueError('Unknown position bin')
    own = np.where(frame.attacker_id == 1, p1, p2)
    opponent = np.where(frame.attacker_id == 1, p2, p1)
    return np.column_stack([3*p1+p2, 9+opponent, 12+3*own+opponent])


def context_info(index):
    if index < 9:
        a, b = PAIRS[index]
        return dict(scope='joint_physical', context=a+'/'+b, p1_position=a, p2_position=b,
                    own_position=None, opponent_position=None)
    if index < 12:
        p = POSITIONS[index-9]
        return dict(scope='opponent_only', context=p, p1_position=None, p2_position=None,
                    own_position=None, opponent_position=p)
    a, b = PAIRS[index-12]
    return dict(scope='joint_role_relative', context=a+'/'+b, p1_position=None, p2_position=None,
                own_position=a, opponent_position=b)


def count_clusters(frame, roster=None):
    roster = pd.Index(sorted(frame.cluster.unique()) if roster is None else roster)
    indices = roster.get_indexer(frame.cluster)
    actions = pd.Index(SKILL_NAMES).get_indexer(frame.attack_skill)
    if (indices < 0).any() or (actions < 0).any(): raise ValueError('Unknown cluster/action')
    counts = np.zeros((len(roster), 21, 5), dtype=np.int64)
    contexts = context_indices(frame)
    for view in range(3): np.add.at(counts, (indices, contexts[:, view], actions), 1)
    return roster, counts


def paired_bootstrap(matrices, replicates, rng):
    """Share resampled episode weights for both players in one saved run."""
    n = len(next(iter(matrices.values())))
    if not all(x.shape == (n, 21, 5) for x in matrices.values()):
        raise ValueError('Inconsistent paired cluster matrices')
    names = list(matrices)
    flat = np.concatenate([matrices[name].reshape(n, -1) for name in names], axis=1).astype(float)
    result = {name: np.empty((replicates, 21, 5)) for name in names}
    for start in range(0, replicates, 64):
        end = min(start + 64, replicates)
        weights = rng.multinomial(n, np.full(n, 1/n), size=end-start)
        sampled = (weights @ flat).reshape(end-start, len(names), 21, 5)
        for i, name in enumerate(names): result[name][start:end] = normalize(sampled[:, i])
    return result


def load_run(file, learned, random_name, meta):
    summary = json.loads(file.with_name('summary.json').read_text())
    metadata = summary.get('metadata', {})
    raw = pd.read_csv(file)
    f = raw[((raw.attacker_id == 1) & (raw.attacker_policy == learned)) |
            ((raw.attacker_id == 2) & (raw.attacker_policy == 'random'))].copy()
    worlds = [reconstruct_gantries(json.loads(r.state_json), int(r.attacker_id), int(r.defender_id)) for r in f.itertuples()]
    errors = []
    for row, world in zip(f.itertuples(), worlds):
        snap = getattr(row, 'defender_actual_world_at_commit_json', None)
        if isinstance(snap, str):
            observed = np.asarray(json.loads(snap), float)
            if observed.shape != (2,) or not np.isfinite(observed).all(): raise ValueError('Invalid coordinate snapshot')
            errors.append(float(np.max(np.abs(observed-world[int(row.defender_id)]))))
    if not errors or max(errors) > 1e-5: raise ValueError('Cannot verify mirrored state/defender coordinates')
    random_rows = f[f.attacker_policy == 'random']
    vectors = [probabilities(v) for v in random_rows.attacker_policy_probabilities_json]
    if any(v is None or not np.allclose(v, np.full(5, .2), rtol=0, atol=1e-12) for v in vectors):
        raise ValueError('Logged random policy is not uniform over five actions')
    f['p1_own_lateral'] = [world[1][1]/.7625 for world in worlds]
    f['p2_own_lateral'] = [-world[2][1]/.7625 for world in worlds]
    known = (f[['p1_own_lateral', 'p2_own_lateral']].abs() <= meta['max_abs_position_normalized']).all(axis=1)
    f['p1_position'] = position_labels(f.p1_own_lateral, meta['center_band_m'])
    f['p2_position'] = position_labels(f.p2_own_lateral, meta['center_band_m'])
    f['dataset'] = np.where(f.attacker_id == 1, learned, random_name)
    f['cluster'] = f.episode_index.astype(int).astype(str)
    f['source_run'] = file.parent.name
    f['source_file'] = str(file)
    f['seed'] = metadata.get('seed')
    report = dict(path=str(file), metadata=metadata, episode_count=summary.get('episode_count'),
        raw_policy_counts=f.groupby('dataset').size().to_dict(), excluded_contexts=int((~known).sum()),
        snapshot_checks=len(errors), maximum_snapshot_error=max(errors), uniform_vectors_verified=len(vectors))
    columns = ['dataset', 'cluster', 'attacker_id', 'defender_id', 'p1_position', 'p2_position',
               'p1_own_lateral', 'p2_own_lateral', 'attack_skill', 'source_run', 'source_file', 'seed', 'episode_index', 'decision_index']
    return f.loc[known, columns], report


def verify_learned_source(actual, previous, model):
    keys = ['cluster', 'p1_position', 'p2_position', 'attack_skill']
    a = actual[actual.dataset == model].groupby(keys).size().sort_index()
    b = previous[previous.dataset == model].groupby(keys).size().sort_index()
    pd.testing.assert_series_equal(a, b)


def prepare_inputs(args):
    audit = json.loads((args.input_dir/'audit.json').read_text()); meta = audit['metadata']
    if meta['human_region']['name'] != 'four_quadrants_center_override': raise ValueError('Wrong human classifier')
    previous = pd.read_csv(args.input_dir/'analysis_observations.csv', dtype={'cluster': str})
    human_name = meta['human_dataset']
    human = pd.read_csv(args.input_dir/'human_incoming_events.csv', dtype={'cluster': str}, low_memory=False)
    region = region_from_dict(meta['human_region']); region.validate()
    labels = region.classify(human.depth, human.lateral)
    if not np.array_equal(labels, human.attack_skill): raise ValueError('Human landing labels do not match fixed rectangular classifier')
    human['dataset'] = human_name
    if not (human[['p1_own_lateral','p2_own_lateral']].abs() <= meta['max_abs_position_normalized']).all().all():
        raise ValueError('Unexpected missing or implausible human position')
    keys = ['cluster','rally_cluster','attacker_id','p1_position','p2_position','attack_skill']
    pd.testing.assert_series_equal(human.groupby(keys).size().sort_index(),
        previous[previous.dataset == human_name].groupby(keys).size().sort_index())
    if human.empty or human.attacker_id.isna().any(): raise ValueError('Missing eligible human cohort or player IDs')
    human['source_run'] = 'LATTE-MV'; human['source_file'] = str(args.input_dir/'human_incoming_events.csv')
    observations = [human]; runs = []
    for file, learned, random_name in zip([args.nash_decisions, args.ibr_decisions], LEARNED, RANDOM):
        frame, run_audit = load_run(file, learned, random_name, meta)
        verify_learned_source(frame, previous, learned)
        observations.append(frame); runs.append(run_audit)
    all_obs = pd.concat(observations, ignore_index=True)
    return all_obs, meta, runs


def rebin_positions(frame, band):
    """Same dimensionless own-side bins for every player and dataset."""
    result = frame.copy()
    for player in (1,2):
        result[f'p{player}_position'] = position_labels(result[f'p{player}_own_lateral'], band)
    result['opponent_position'] = np.where(result.attacker_id == 1, result.p2_position, result.p1_position)
    result['defense_center_half_band_m'] = band
    return result


def run(args, prepared=None):
    out = args.output_dir
    if out.resolve() == args.input_dir.resolve(): raise ValueError('Use a separate output directory')
    all_obs, meta, runs = prepare_inputs(args) if prepared is None else prepared
    source_band = meta['center_band_m']
    band = source_band if args.center_band is None else args.center_band
    if band != source_band and out.resolve() == Path('v3/human_comparison/reports/similarity_with_random_10k_both').resolve():
        raise ValueError('Use a new output directory for a different position band')
    all_obs = rebin_positions(all_obs, band)
    human_name = meta['human_dataset']; human = all_obs[all_obs.dataset == human_name]
    out.mkdir(parents=True, exist_ok=True)
    all_obs.to_csv(out/'analysis_observations.csv', index=False)
    rng = np.random.default_rng(args.seed)
    points = {}; draws = {}; counts = {}; total_clusters = {}; cluster_support = {}
    _, hc = count_clusters(human)
    points[human_name] = normalize(hc.sum(axis=0)); counts[human_name] = hc.sum(axis=0)
    total_clusters[human_name] = len(hc); cluster_support[human_name] = (hc.sum(axis=-1)>0).sum(axis=0)
    draws.update(paired_bootstrap({human_name: hc}, args.bootstrap, rng))
    for learned, random_name in zip(LEARNED, RANDOM):
        f = all_obs[all_obs.dataset.isin([learned, random_name])]
        roster = sorted(f.cluster.unique())
        matrices = {}
        for name in [learned, random_name]:
            _, matrix = count_clusters(f[f.dataset == name], roster)
            matrices[name] = matrix; counts[name] = matrix.sum(axis=0)
            points[name] = normalize(counts[name]); total_clusters[name] = len(roster)
            cluster_support[name] = (matrix.sum(axis=-1)>0).sum(axis=0)
        print(f'Bootstrapping paired {learned} / {random_name}: {len(roster)} rallies', flush=True)
        draws.update(paired_bootstrap(matrices, args.bootstrap, rng))
    points[UNIFORM] = np.full((21, 5), .2)
    draws[UNIFORM] = np.broadcast_to(points[UNIFORM], (args.bootstrap, 21, 5))
    distribution_rows = []; visitation = []
    for name in [human_name, *MODELS]:
        for j in range(21):
            observed = name != UNIFORM
            n = int(counts[name][j].sum()) if observed else np.nan
            scope = context_info(j)['scope']
            denominator = len(all_obs[all_obs.dataset == name]) if observed else 0
            visitation.append(dict(model=name, **context_info(j), n=n,
                n_contributing_clusters=int(cluster_support[name][j]) if observed else np.nan,
                total_clusters=total_clusters.get(name, np.nan),
                visitation_fraction=n/denominator if denominator else np.nan))
            for k, skill in enumerate(SKILL_NAMES):
                lo, hi, reps = interval(draws[name][:, j, k])
                distribution_rows.append(dict(model=name, **context_info(j), attack_skill=skill,
                    count=int(counts[name][j,k]) if observed else np.nan, n=n,
                    probability=points[name][j,k], probability_ci_low=lo, probability_ci_high=hi,
                    valid_bootstrap_replicates=reps, source_type='empirical' if observed else 'exact_policy_distribution'))
    pd.DataFrame(distribution_rows).to_csv(out/'conditional_distributions.csv', index=False)
    pd.DataFrame(visitation).to_csv(out/'counts_and_visitation.csv', index=False)
    metrics = []; samples = {}
    for model in MODELS:
        point = all_metrics(points[human_name], points[model])
        samples[model] = all_metrics(draws[human_name], draws[model])
        for j in range(21):
            row = dict(comparison='human_vs_'+model, model=model, **context_info(j),
                human_n=int(counts[human_name][j].sum()), model_n=int(counts[model][j].sum()) if model != UNIFORM else np.nan,
                physical_attacker='P2' if model in RANDOM else 'P1' if model in LEARNED else 'not_applicable',
                reference='human_realized_landing_proxy', model_measurement='selected_skill' if model != UNIFORM else 'exact_uniform_action_distribution')
            for metric in METRICS:
                lo, hi, reps = interval(samples[model][metric][:, j])
                row.update({metric: float(point[metric][j]), metric+'_ci_low': lo,
                            metric+'_ci_high': hi, metric+'_valid_replicates': reps})
            metrics.append(row)
    metric_table = pd.DataFrame(metrics)
    metric_table.to_csv(out/'all_similarity_metrics.csv', index=False)
    for scope in SCOPES: metric_table[metric_table.scope==scope].to_csv(out/(scope+'_similarity_metrics.csv'), index=False)
    wide = metric_table.pivot(index=['scope','context'], columns='model', values=[*METRICS,'human_n','model_n'])
    wide.columns = [model+'__'+metric for metric,model in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(out/'comparison_metrics_wide.csv', index=False)
    for scope in SCOPES: wide[wide.scope==scope].to_csv(out/(scope+'_metrics_wide.csv'), index=False)
    improvements = []
    for learned, baseline in itertools.product(LEARNED, [*RANDOM, UNIFORM]):
        for j in range(21):
            m = metric_table[(metric_table.scope==context_info(j)['scope']) & (metric_table.context==context_info(j)['context'])].set_index('model')
            row = dict(policy=learned, baseline=baseline, **context_info(j),
                human_n=int(counts[human_name][j].sum()), paired_robot_rally_bootstrap=bool((learned,baseline) in zip(LEARNED,RANDOM)))
            for metric in METRICS:
                sign = 1 if metric in ('js_divergence_bits','total_variation') else -1
                value = sign*(m.loc[baseline,metric]-m.loc[learned,metric])
                boot = sign*(samples[baseline][metric][:,j]-samples[learned][metric][:,j])
                lo, hi, reps = interval(boot)
                row.update({metric+'_improvement': value, metric+'_improvement_ci_low':lo,
                    metric+'_improvement_ci_high':hi, metric+'_valid_replicates':reps})
            defined = np.isfinite(row['js_divergence_bits_improvement'])
            row['closer_than_random_by_js'] = bool(row['js_divergence_bits_improvement']>0) if defined else pd.NA
            row['js_interval_result'] = 'undefined' if not defined else 'closer' if row['js_divergence_bits_improvement_ci_low']>0 else 'farther' if row['js_divergence_bits_improvement_ci_high']<0 else 'unresolved'
            improvements.append(row)
    improvements = pd.DataFrame(improvements); improvements.to_csv(out/'learned_vs_random_improvements.csv', index=False)
    plot_heatmaps(out, metric_table, band)
    sources = [args.input_dir/'audit.json', args.input_dir/'analysis_observations.csv', args.input_dir/'human_incoming_events.csv',
               args.nash_decisions, args.ibr_decisions, args.nash_decisions.with_name('summary.json'),
               args.ibr_decisions.with_name('summary.json'), Path(__file__), Path('v3/human_comparison/analyze_opponent_position.py')]
    provenance = []
    for path in sources:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
        provenance.append(dict(path=str(path),sha256=digest.hexdigest()))
    summary = dict(human_dataset=human_name, human_region=meta['human_region'], totals=all_obs.groupby('dataset').size().to_dict(),
        center_band_m=band, normalized_center_half_band=band/.7625, source_center_band_m=source_band,
        robot_run_audits=runs, bootstrap_replicates=args.bootstrap, seed=args.seed, sources=provenance,
        learned_reference_check='all episode/physical-context/skill counts match preserved comparison',
        software=dict(numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__))
    (out/'audit.json').write_text(json.dumps(summary,indent=2))
    write_report(args,summary,metric_table,improvements)
    print(metric_table[metric_table.scope=='opponent_only'][['model','context','human_n','model_n','js_divergence_bits','spearman_rho']].to_string(index=False))


def plot_heatmaps(out, metrics, band=None):
    for metric, limits, cmap, label in [('js_divergence_bits',(0,1),'viridis','JS divergence (bits; lower is closer)'),
                                        ('spearman_rho',(-1,1),'coolwarm','Spearman correlation')]:
        fig, axes = plt.subplots(1,3,figsize=(19,8),layout='constrained')
        for ax,scope in zip(axes,SCOPES):
            contexts = [context_info(i)['context'] for i in range(21) if context_info(i)['scope']==scope]
            data = metrics[metrics.scope==scope].pivot(index='context',columns='model',values=metric).reindex(index=contexts,columns=MODELS)
            im=ax.imshow(np.ma.masked_invalid(data.to_numpy()),vmin=limits[0],vmax=limits[1],cmap=cmap,aspect='auto')
            ax.set_xticks(range(5),[NAMES[m] for m in MODELS],rotation=30,ha='right',fontsize=8)
            ax.set_yticks(range(len(contexts)),contexts);ax.set_title(scope.replace('_',' '))
            for i,j in itertools.product(range(len(contexts)),range(5)):
                value=data.iloc[i,j];ax.text(j,i,f'{value:.3f}' if pd.notna(value) else 'NA',ha='center',va='center',fontsize=8,
                    bbox=dict(facecolor='white',alpha=.65,edgecolor='none'))
        fig.colorbar(im,ax=list(axes),label=label,shrink=.75)
        fig.suptitle('Human landing-proxy similarity: learned policies and random baselines'+
                     (f' | position center ±{band:g} m / robot half-width' if band is not None else ''))
        fig.savefig(out/(metric+'_comparisons.png'),dpi=300,bbox_inches='tight');plt.close(fig)


def write_report(args,audit,metrics,improvements):
    rows=['# Human similarity with empirical and exact random baselines',
        'All values compare human **realized landing proxies** (four_quadrants_center_override) with robot **selected high-level skills**. Lower JS/TV means a closer category distribution, not better playing strength, win rate, or equivalent intentions. No simulations, reconstruction or model training were run.',
        '## Inputs and baseline definitions',
        'The saved incoming-aligned both-player human cohort is unchanged. Each learned policy remains P1. Random attackers are P2 in the existing runs, not a new P1 random evaluation. The two empirical random baselines are kept separate because they faced different learned opponents and visited different states. The exact uniform baseline is the existing random policy’s 20%-per-skill action distribution, verified against every logged random probability vector; it is not an observed landing distribution. Its sample count is blank rather than invented.',
        'Observation totals: `'+json.dumps(audit['totals'],sort_keys=True)+'`. Source snapshots validate inversion of the P2 canonical frame, and learned per-episode/physical-context/action counts match the preserved comparison exactly.',
        '## Coordinate and conditioning conventions',
        f"Position center half-band = {audit['center_band_m']:g} m at robot scale, normalized by 0.7625 for both humans and robots. It applies to both players’ own-side positions. Source verification was performed at the original {audit['source_center_band_m']:g} m band before rebinning. Landing categories and the eligible shot cohort remain fixed; wider center bins redistribute observations rather than create shots.",
        'Each player’s Left means their own left: P1 +world-y, P2 −world-y. joint_physical preserves the previous (physical P1,physical P2) ordering. Since learned attacks are P1 and logged random attacks P2, these physical-pair comparisons are not role-matched controls. The additional joint_role_relative view uses (hitter position,defender position), swapping the two bins for human or robot P2 attacks. opponent_only always uses the actual defender’s own-side position. All views pool raw counts and then normalize. Unequal visitation and sample counts remain explicit in the CSVs.',
        '## Metrics and uncertainty',
        'all_similarity_metrics.csv includes JS divergence in bits (not square-root distance), TV, Spearman and Pearson correlations, counts, and 95% cluster-bootstrap intervals. Correlation with an exactly uniform vector is mathematically undefined, so those cells are blank/NA. No zero correlation is substituted. Sparse/empty contexts and constant sampled vectors can also reduce the valid-replicate count.',
        'comparison_metrics_wide.csv puts every model’s point metrics side by side in one row per context; separate *_metrics_wide.csv files contain each conditioning view. Detailed uncertainty remains in all_similarity_metrics.csv. Empirical Random correlations rank small sampling fluctuations around 20% and can therefore be unstable or accidentally high; these correlations are not evidence that Random learned a structured human-like response.',
        f"The {audit['bootstrap_replicates']} replicates use seed {audit['seed']}. Human matches are resampled once per replicate across all human shots and contexts. For each robot run, one set of resampled rally weights is shared by the learned P1 and random P2 counts, including zero-decision clusters. Different saved runs are resampled independently. The exact uniform probabilities are fixed; their similarity intervals reflect human uncertainty only. Intervals are pointwise and do not include training-seed, reconstruction, or region-boundary uncertainty.",
        'learned_vs_random_improvements.csv reports baseline JS/TV minus learned JS/TV; positive means the learned policy is closer. Correlation improvements are learned minus baseline. Positive differences are not claims of stronger gameplay. The shared human draws and within-run paired robot draws are retained in the difference intervals.',
        '## Opponent-only results', table_text(metrics[metrics.scope=='opponent_only']),
        '## Nine physical-pair results', table_text(metrics[metrics.scope=='joint_physical']),
        '## Role-matched joint sensitivity', table_text(metrics[metrics.scope=='joint_role_relative']),
        '## Comparison against exact uniform random',
        '| View | Policy | Closer JS point estimates | Farther JS point estimates | Total contexts |\n|---|---|---:|---:|---:|']
    summary=[]
    for scope,policy in itertools.product(SCOPES,LEARNED):
        f=improvements[(improvements.scope==scope)&(improvements.policy==policy)&(improvements.baseline==UNIFORM)]
        summary.append(f'| {scope} | {policy} | {int(f.closer_than_random_by_js.sum())} | {int((f.js_divergence_bits_improvement<0).sum())} | {len(f)} |')
    rows[-1]+='\n'+'\n'.join(summary)
    rows += ['Inspect the improvement intervals before interpreting a point ordering. State visitation, physical role, and the human-successful-landing versus robot-all-selections distinction remain possible confounds. Exact-uniform and empirical-random results are labeled separately.',
        '## Reproduce',
        '```bash\nOPENBLAS_NUM_THREADS=1 python3 v3/human_comparison/analyze_random_similarity.py --center-band '+str(audit['center_band_m'])+' --bootstrap '+str(args.bootstrap)+' --seed '+str(args.seed)+' --output-dir '+str(args.output_dir)+'\n```',
        'Previous reports are unchanged. Inputs, metadata, validation counts and hashes are recorded in audit.json.',
        '![JS comparisons](js_divergence_bits_comparisons.png)',
        '![Correlation comparisons](spearman_rho_comparisons.png)']
    (args.output_dir/'report.md').write_text('\n\n'.join(rows)+'\n')


def table_text(frame):
    rows=['| Context | Model | Human n | Model n | JS (bits) | TV | Spearman | Pearson |',
          '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in frame.itertuples():
        n=str(int(r.model_n)) if pd.notna(r.model_n) else 'exact'
        def fmt(v):return f'{v:.4f}' if np.isfinite(v) else 'NA'
        rows.append(f'| {r.context} | {r.model} | {r.human_n} | {n} | {fmt(r.js_divergence_bits)} | {fmt(r.total_variation)} | {fmt(r.spearman_rho)} | {fmt(r.pearson_r)} |')
    return '\n'.join(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,default=Path('v3/human_comparison/reports/opponent_position_comparison_10k_both'))
    p.add_argument('--nash-decisions',type=Path,default=Path('skill_eval/v3/nash_vs_random_10000/decisions.csv'))
    p.add_argument('--ibr-decisions',type=Path,default=Path('skill_eval/v3/potential_ibr_vs_random_10000/decisions.csv'))
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--center-band',type=float,help='Position center half-band in robot meters; normalized identically for human/robot. Does not alter landing regions.')
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.bootstrap<100:p.error('--bootstrap must be at least 100')
    if args.center_band is not None and not 0<=args.center_band<=.7625:p.error('--center-band must be between 0 and 0.7625')
    if args.output_dir is None:
        suffix='' if args.center_band is None else '_center_'+f'{args.center_band:g}'.replace('.','p')
        args.output_dir=Path('v3/human_comparison/reports/similarity_with_random_10k_both'+suffix)
    run(args)


if __name__=='__main__':main()
