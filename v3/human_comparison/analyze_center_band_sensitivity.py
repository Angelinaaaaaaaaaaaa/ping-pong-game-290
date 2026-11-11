#!/usr/bin/env python3
"""Widen player-position center bins while freezing human landing regions.

Only saved incoming-aligned observations and robot decisions are read. The
main half-band is declared before metrics; alternative widths are sensitivity
checks, not a search for the most human-like learned-policy result.
"""
import argparse
from argparse import Namespace
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
from v3.human_comparison import analyze_random_similarity as comparison
from v3.human_comparison.spatial import region_from_dict
from v3.human_comparison.plot_joint_landing_maps import save_grid, subset, PAIRS, EDGES


def band_directory(band):
    return 'center_' + f'{band:g}'.replace('.', 'p')


def draw_sensitivity(out, coverage, metrics, primary):
    human = coverage[coverage.model.str.startswith('human_')]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, scope, context, title in zip(axes, ['opponent_only', 'joint_physical'], ['center','center/center'],
            ['Human defender in Center', 'Human physical (Center, Center)']):
        f = human[(human.scope==scope)&(human.context==context)].sort_values('center_band_m')
        ax.plot(f.center_band_m, f.n, 'o-', color='#2878b5')
        for r in f.itertuples(): ax.annotate(str(int(r.n)), (r.center_band_m,r.n), xytext=(0,7),textcoords='offset points',ha='center')
        ax.axvline(primary,color='gray',ls=':',lw=.8)
        ax.set(xlabel='Position center half-band (robot m; shared normalized bins)',ylabel='Human observations',title=title)
        ax.set_ylim(0,max(f.n)*1.18)
    fig.savefig(out/'human_center_coverage_vs_band.png',dpi=300,bbox_inches='tight');plt.close(fig)
    fig, axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    colors=['#2878b5','#d07920','#26875d','#9467bd','#555555']
    for ax,position in zip(axes,comparison.POSITIONS):
        for model,color in zip(comparison.MODELS,colors):
            f=metrics[(metrics.scope=='opponent_only')&(metrics.context==position)&(metrics.model==model)].sort_values('center_band_m')
            ax.plot(f.center_band_m,f.js_divergence_bits,'o-',color=color,label=comparison.NAMES[model].replace('\n',' '))
            ax.fill_between(f.center_band_m,f.js_divergence_bits_ci_low,f.js_divergence_bits_ci_high,color=color,alpha=.10)
        ax.axvline(primary,color='gray',ls=':',lw=.8)
        ax.set(xlabel='Position center half-band (robot m)',ylabel='JS divergence from human (bits)',title='Defender '+position)
    axes[0].legend(fontsize=7);fig.suptitle('Fixed four-block + Center Safe landing classifier; 95% cluster-bootstrap intervals')
    fig.savefig(out/'opponent_similarity_vs_center_band.png',dpi=300,bbox_inches='tight');plt.close(fig)


def run(args):
    out=args.output_dir
    if out.resolve()==args.input_dir.resolve():raise ValueError('Use a separate output directory')
    out.mkdir(parents=True,exist_ok=True)
    setup=Namespace(input_dir=args.input_dir,nash_decisions=args.nash_decisions,ibr_decisions=args.ibr_decisions)
    prepared=comparison.prepare_inputs(setup)
    observations,meta,runs=prepared
    region=region_from_dict(meta['human_region']);region.validate()
    human_name=meta['human_dataset']
    # This is written before calculating similarity, so the main band cannot
    # be silently selected afterward using a favorable policy comparison.
    config=dict(primary_band_m=args.primary_band,bands_m=args.bands,source_center_band_m=meta['center_band_m'],
        fixed_landing_regions=meta['human_region'],selection_rule='primary specified before metrics; widths chosen for increased center coverage',
        bootstrap_replicates=args.bootstrap,seed=args.seed,cohort_totals=observations.groupby('dataset').size().to_dict())
    (out/'configuration.json').write_text(json.dumps(config,indent=2))
    original_category_counts=observations.groupby(['dataset','attack_skill']).size().sort_index()
    metrics=[];improvements=[];coverage=[];wide=[];landings=[];probabilities=[]
    for band in args.bands:
        directory=out/band_directory(band)
        print(f'Analyzing center half-band ±{band:g} m (primary={args.primary_band:g})',flush=True)
        current=Namespace(**vars(setup),output_dir=directory,center_band=band,bootstrap=args.bootstrap,seed=args.seed)
        comparison.run(current,prepared=prepared)
        f=comparison.rebin_positions(observations,band)
        pd.testing.assert_series_equal(f.groupby(['dataset','attack_skill']).size().sort_index(),original_category_counts)
        h=f[f.dataset==human_name].copy()
        h['human_landing_category']=region.classify(h.depth,h.lateral)
        if not h.human_landing_category.eq(h.attack_skill).all():raise ValueError('Landing classifier changed during position sensitivity')
        h['classification_method']=region.name
        h[['rally_id','shot_index','attacker_id','depth','lateral','p1_position','p2_position',
           'opponent_position','human_landing_category','classification_method','defense_center_half_band_m']].to_csv(
               directory/'human_landing_points.csv',index=False)
        landings.append((band,h))
        for filename,destination in [('all_similarity_metrics.csv',metrics),('learned_vs_random_improvements.csv',improvements),
                                     ('counts_and_visitation.csv',coverage),('comparison_metrics_wide.csv',wide),
                                     ('conditional_distributions.csv',probabilities)]:
            table=pd.read_csv(directory/filename);table['center_band_m']=band;destination.append(table)
    metrics=pd.concat(metrics,ignore_index=True);improvements=pd.concat(improvements,ignore_index=True)
    coverage=pd.concat(coverage,ignore_index=True);wide=pd.concat(wide,ignore_index=True)
    probs=pd.concat(probabilities,ignore_index=True)
    metrics.to_csv(out/'all_bands_similarity_metrics.csv',index=False)
    improvements.to_csv(out/'all_bands_learned_vs_random_improvements.csv',index=False)
    coverage.to_csv(out/'all_bands_counts_and_visitation.csv',index=False)
    wide.to_csv(out/'all_bands_metrics_wide.csv',index=False)
    probs.to_csv(out/'all_bands_conditional_distributions.csv',index=False)
    primary_wide=wide[np.isclose(wide.center_band_m,args.primary_band)]
    primary_wide.to_csv(out/'primary_metrics_wide.csv',index=False)
    for scope in comparison.SCOPES:
        primary_wide[primary_wide.scope==scope].to_csv(out/('primary_'+scope+'_metrics_wide.csv'),index=False)
    summaries=[]
    for key,g in improvements.groupby(['center_band_m','scope','policy','baseline'],sort=False):
        summaries.append(dict(center_band_m=key[0],scope=key[1],policy=key[2],baseline=key[3],
            contexts=len(g),closer_point_estimates=int((g.js_divergence_bits_improvement>0).sum()),
            farther_point_estimates=int((g.js_divergence_bits_improvement<0).sum()),
            closer_interval_excludes_zero=int((g.js_interval_result=='closer').sum()),
            farther_interval_excludes_zero=int((g.js_interval_result=='farther').sum()),
            unresolved_intervals=int((g.js_interval_result=='unresolved').sum())))
    summaries=pd.DataFrame(summaries);summaries.to_csv(out/'robustness_summary.csv',index=False)
    overall=observations.groupby(['dataset','attack_skill']).size().rename('count').reset_index()
    overall['probability']=overall['count']/overall.groupby('dataset')['count'].transform('sum')
    overall.to_csv(out/'fixed_overall_category_frequencies.csv',index=False)
    mix=[]
    for model,g in overall.groupby('dataset'):
        mix.append(dict(model=model,n=int(g['count'].sum()),
            short_fraction=float(g[g.attack_skill.isin(['left_short','right_short'])].probability.sum()),
            center_safe_fraction=float(g[g.attack_skill=='center_safe'].probability.sum()),
            deep_side_fraction=float(g[g.attack_skill.isin(['left','right'])].probability.sum())))
    pd.DataFrame(mix).to_csv(out/'fixed_overall_short_center_deep_mix.csv',index=False)
    vmax=0.
    for _,h in landings:
        for pair in PAIRS:
            s=subset(h,pair)
            if len(s)>=20:vmax=max(vmax,float(np.histogram2d(s.depth,s.lateral,bins=EDGES)[0].max()/len(s)))
    for band,h in landings:
        directory=out/band_directory(band)
        for density in (False,True):
            save_grid(h,region,directory,'human_nine_context_'+('density' if density else 'landing_marks'),
                f'Incoming-aligned human P1+P2 | position Center ±{band:g} m | fixed four-block landing classifier',density,vmax)
    draw_sensitivity(out,coverage,metrics,args.primary_band)
    result=dict(**config,landing_categories_unchanged=True,eligible_cohort_unchanged=True,
        common_bootstrap_random_seed_across_bands=True,shared_landing_density_vmax=vmax,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'audit.json').write_text(json.dumps(result,indent=2))
    write_report(args,result,coverage,metrics,summaries,pd.DataFrame(mix))
    print('Main widened-band opponent-only results:')
    print(metrics[(metrics.scope=='opponent_only')&np.isclose(metrics.center_band_m,args.primary_band)][
        ['context','model','human_n','js_divergence_bits','spearman_rho']].to_string(index=False))


def write_report(args,audit,coverage,metrics,summary,mix):
    human=coverage[coverage.model.str.startswith('human_')]
    counts=['| Center half-band (robot m) | Human opponent-center n | Human (C,C) n |', '|---:|---:|---:|']
    for band in args.bands:
        g=human[np.isclose(human.center_band_m,band)]
        oc=int(g[(g.scope=='opponent_only')&(g.context=='center')].n.iloc[0])
        cc=int(g[(g.scope=='joint_physical')&(g.context=='center/center')].n.iloc[0])
        counts.append(f'| ±{band:g} | {oc} | {cc} |')
    primary=metrics[np.isclose(metrics.center_band_m,args.primary_band)]
    tested=summary[summary.baseline==comparison.UNIFORM]
    total_wins=int(tested.closer_point_estimates.sum())
    total_losses=int(tested.farther_point_estimates.sum())
    empirical=summary[summary.baseline!=comparison.UNIFORM]
    conclusion=(f'Across all tested bands and conditioning views, learned policies have lower JS than exact uniform Random in {total_wins} context comparisons, and higher JS in {total_losses}. '
        f'Against the two separate logged Random baselines, learned policies have lower JS in {int(empirical.closer_point_estimates.sum())} context comparisons and higher JS in {int(empirical.farther_point_estimates.sum())}. '
        'These context comparisons overlap and are not independent replications. The wider bins add center coverage but do not support greater human-category similarity of the learned policies under the fixed classifier. This is not a ranking of playing strength.')
    robustness=['| Band | View | Policy | Exact-uniform Random closer JS contexts | Total contexts |','|---:|---|---|---:|---:|']
    for r in summary[summary.baseline==comparison.UNIFORM].itertuples():
        robustness.append(f'| ±{r.center_band_m:g} | {r.scope} | {r.policy} | {r.farther_point_estimates} | {r.contexts} |')
    mixtures=['| Dataset | n | Short categories | Center Safe | Deep-side categories |','|---|---:|---:|---:|---:|']
    for r in mix.itertuples():mixtures.append(f'| {r.model} | {r.n} | {r.short_fraction:.2%} | {r.center_safe_fraction:.2%} | {r.deep_side_fraction:.2%} |')
    texts=['# Wider defensive-position center bins: human, learned-policy and random similarity',
        f"The primary player-position center half-band was set to **±{args.primary_band:g} m** before similarity was calculated. Bands {args.bands} are reported as sensitivity checks. The primary choice addresses observation coverage; no threshold is optimized for human–robot agreement. All bands use the same eligible incoming-aligned cohort and identical landing-category labels.",
        '## Position bins and fixed landing classifier',
        'Position Center means |own-side lateral / table half-width| ≤ band/0.7625. The same normalized definition is applied to human feet and robot gantries, for both physical players. P1’s own left is +world-y; P2’s own left is −world-y. The actual human half-width is 0.762 m, so the effective human meter boundary differs only by this table-width ratio. Widening redistributes existing observations; it does not recover previously unobserved incoming crossings or create new data.',
        'Landing Center Safe is a separate concept and stays fixed. The complete four base blocks split normalized depth at **0.51** and hitter-relative lateral at **0**. The inclusive rectangle **0.20 ≤ depth ≤ 0.70, −0.35 ≤ lateral ≤ 0.35** overrides the base blocks. Every valid landing receives exactly one of five categories. No nearest-target classifier or unmapped region is used. Exact definition: `'+json.dumps(audit['fixed_landing_regions'],sort_keys=True)+'`.',
        '## Center coverage', '\n'.join(counts),
        'The total human cohort remains '+str(audit['cohort_totals'].get('human_both'))+'; underlying skill/category counts are verified unchanged at every band. Raw position/context counts for all datasets and all nine scenarios are in all_bands_counts_and_visitation.csv.',
        '## Main finding',conclusion,
        '## Primary widened-band opponent-only comparison',comparison.table_text(primary[primary.scope=='opponent_only']),
        '## Primary nine physical pairs',comparison.table_text(primary[primary.scope=='joint_physical']),
        '## Primary role-matched joint sensitivity',comparison.table_text(primary[primary.scope=='joint_role_relative']),
        '## Does the learned-versus-random ordering survive wider bins?', '\n'.join(robustness),
        'The table counts JS point orderings against exact uniform Random, without treating all point differences as resolved. robustness_summary.csv and all_bands_learned_vs_random_improvements.csv retain separate empirical Random baselines, TV/correlation differences, and uncertainty. Positive JS/TV improvement means a learned policy is closer; negative means Random is closer. Check the pointwise bootstrap intervals for each context, especially sparse ones. No family-wise significance claim is made.',
        '## Category-mix investigation', '\n'.join(mixtures),
        'The aggregate short/center/deep mix does not change with a position boundary. A persistent mismatch in that mix can contribute to JS differences even as center contexts gain observations. Human labels describe successful reconstructed placements; robot labels describe selected commands, including failed shots. A short command is not proof of a short realized landing. Thus these results measure category-proxy similarity and cannot establish identical intentions, physical placement, or playing strength.',
        '## Method and limitations',
        f"JS divergence uses base-2 logarithms; TV, Spearman and Pearson are also saved. Each band uses {args.bootstrap} cluster-bootstrap replicates with seed {args.seed}; the same cluster resampling sequence across bands improves comparability. Human P1/P2 shots from a match share a cluster. Learned and Random players within each robot run share rally weights. Different runs are independent in this bootstrap. Exact uniform Random is 20% per skill and has undefined correlation; its intervals reflect human uncertainty. Empirical Random correlations can be unstable around near-uniform frequency vectors.",
        'Empirical Random is physical P2; learned policies are P1. The opponent-only view always conditions on the actual defender’s own-side position. joint_physical preserves physical (P1,P2) order; joint_role_relative uses (hitter,defender) and swaps P2-attacker bins. The latter is the clearer joint view for cross-role comparisons. Human and robot pose measurements, state visitation, success selection and reconstruction quality remain different. Broader bins also mix more distinct spatial states. No simulation, reconstruction, Q/PPO/potential training or policy behavior was changed.',
        '## Outputs and reproduce',
        'primary_*_metrics_wide.csv provides side-by-side metrics for the widened primary band. Each center_* directory contains complete distributions, counts, intervals, learned-versus-random improvements, and landing scatter/density maps. The combined all_bands_* files retain the sensitivity dimension explicitly. Baseline ±0.05 m outputs are regenerated in this new directory; original analyses stay untouched.',
        '```bash\nOPENBLAS_NUM_THREADS=1 python3 v3/human_comparison/analyze_center_band_sensitivity.py --bands '+' '.join(map(str,args.bands))+' --primary-band '+str(args.primary_band)+' --bootstrap '+str(args.bootstrap)+' --seed '+str(args.seed)+' --output-dir '+str(args.output_dir)+'\n```',
        '![Human center coverage](human_center_coverage_vs_band.png)',
        '![Opponent JS sensitivity](opponent_similarity_vs_center_band.png)',
        f'![Main widened landing marks]({band_directory(args.primary_band)}/human_nine_context_landing_marks.png)',
        f'![Main widened landing density]({band_directory(args.primary_band)}/human_nine_context_density.png)']
    (args.output_dir/'report.md').write_text('\n\n'.join(texts)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,default=Path('v3/human_comparison/reports/opponent_position_comparison_10k_both'))
    p.add_argument('--nash-decisions',type=Path,default=Path('skill_eval/v3/nash_vs_random_10000/decisions.csv'))
    p.add_argument('--ibr-decisions',type=Path,default=Path('skill_eval/v3/potential_ibr_vs_random_10000/decisions.csv'))
    p.add_argument('--output-dir',type=Path,default=Path('v3/human_comparison/reports/center_band_sensitivity_10k_both'))
    p.add_argument('--bands',type=float,nargs='+',default=[.05,.15,.25])
    p.add_argument('--primary-band',type=float,default=.15)
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args();args.bands=sorted(set(args.bands))
    if args.bootstrap<100:p.error('--bootstrap must be at least 100')
    if any(not 0<=b<=.7625 for b in args.bands):p.error('Position half-bands must be between 0 and 0.7625')
    if args.primary_band not in args.bands:p.error('--primary-band must be included in --bands')
    run(args)


if __name__=='__main__':main()
