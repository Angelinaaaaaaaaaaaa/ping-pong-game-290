#!/usr/bin/env python3
"""Side-by-side rectangular human landings, Nash-Q and Potential Max actions."""
import argparse
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
from v3.human_comparison.spatial import region_from_dict
from v3.human_comparison.plot_human_rectangular_skill_bars import (
    summarize_humans, draw_bars, PAIRS, LABELS, SKILL_NAMES,
)

MODELS = ('human_both', 'nash_q', 'potential_max')
NAMES = {'human_both':'Human P1+P2 (four-box landing categories)',
         'nash_q':'Nash-Q P1 (selected skills)', 'potential_max':'Potential Max P1 (selected skills)'}
MODEL_COLORS = ('#555555', '#2878b5', '#d07920')


def assemble_comparison(human_frame, region, reference):
    _, human = summarize_humans(human_frame, region)
    human = human[human.player_cohort=='both_players'].rename(columns={'landing_category':'attack_skill'}).copy()
    human['model']='human_both'
    ref = reference[reference.scope=='joint_physical']
    keys = ['p1_position','p2_position','attack_skill']
    expected = ref[ref.model=='human_both'].set_index(keys).sort_index()
    actual = human.set_index(keys).sort_index()
    pd.testing.assert_series_equal(actual['count'],expected['count'],check_names=False,check_dtype=False)
    np.testing.assert_allclose(actual.probability,expected.probability,equal_nan=True)
    human=human.merge(ref[ref.model=='human_both'][[*keys,'probability_ci_low','probability_ci_high']],on=keys,validate='one_to_one')
    robot=ref[ref.model.isin(MODELS[1:])].copy()
    if set(robot.model)!=set(MODELS[1:]):raise ValueError('Require both Nash-Q and Potential Max distributions')
    columns=['model',*keys,'count','n','probability','probability_ci_low','probability_ci_high']
    data=pd.concat([human[columns],robot[columns]],ignore_index=True)
    data['measurement']=np.where(data.model=='human_both','realized_landing_category','selected_high_level_skill')
    for _,g in data.groupby(['model','p1_position','p2_position']):
        if len(g)!=5 or g['count'].sum()!=g.n.iloc[0]:raise ValueError('Incomplete context counts')
        if g.n.iloc[0] and not np.isclose(g.probability.sum(),1):raise ValueError('Non-normalized probabilities')
    return data


def cell(data, model, pair):
    return data[(data.model==model)&(data.p1_position==pair[0])&(data.p2_position==pair[1])]


def run(args):
    out=args.output_dir
    if out.resolve() in {args.human_dir.resolve(),args.robot_dir.resolve()}:raise ValueError('Choose a separate output directory')
    hmeta=json.loads((args.human_dir/'summary.json').read_text())
    rmeta=json.loads((args.robot_dir/'audit.json').read_text())
    if hmeta['classification']!=rmeta['human_region']:raise ValueError('Human/robot comparison uses different landing definitions')
    band=rmeta['center_band_m']
    if not np.isclose(hmeta['position_center_half_band_m'],band):raise ValueError('Position bin mismatch')
    region=region_from_dict(hmeta['classification']);region.validate()
    if region.name!='four_quadrants_center_override':raise ValueError('Require explicit four-box human classifier')
    human=pd.read_csv(args.human_dir/'human_reclassified_landing_points.csv')
    reference=pd.read_csv(args.robot_dir/'conditional_distributions.csv')
    data=assemble_comparison(human,region,reference)
    out.mkdir(parents=True,exist_ok=True)
    data['position_center_half_band_m']=band
    data['human_classification_method']=region.name
    data.to_csv(out/'human_nash_q_potential_max_distributions.csv',index=False)
    wide=[]
    for pair in PAIRS:
        row=dict(p1_position=pair[0],p2_position=pair[1])
        for model in MODELS:
            g=cell(data,model,pair).set_index('attack_skill').reindex(SKILL_NAMES)
            row[model+'_n']=int(g.n.iloc[0])
            for skill in SKILL_NAMES:
                row[model+'_'+skill+'_count']=int(g.loc[skill,'count'])
                row[model+'_'+skill+'_probability']=g.loc[skill,'probability']
        wide.append(row)
    pd.DataFrame(wide).to_csv(out/'nine_pair_distributions_side_by_side.csv',index=False)
    # One grouped comparison panel for every physical-player context.
    fig,axes=plt.subplots(3,3,figsize=(16,12),layout='constrained')
    for ax,pair in zip(axes.flat,PAIRS):
        sizes=[]
        for i,model in enumerate(MODELS):
            g=cell(data,model,pair).set_index('attack_skill').reindex(SKILL_NAMES)
            sizes.append(int(g.n.iloc[0]));x=np.arange(5)+(i-1)*.25
            if sizes[-1]:
                bars=ax.bar(x,g.probability,.24,color=MODEL_COLORS[i],label=NAMES[model])
                ax.vlines(x,g.probability_ci_low,g.probability_ci_high,color='#222222',lw=.7)
                for xp,value,high in zip(x,g.probability,g.probability_ci_high):
                    y=max(value,high) if pd.notna(high) else value
                    ax.text(xp,y+.012,f'{value:.0%}',ha='center',fontsize=7)
        ax.set_xticks(range(5),LABELS,fontsize=9);ax.set_ylim(0,1)
        ax.set_ylabel('Conditional probability');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        ax.set_title(f'P1 {pair[0].title()} / P2 {pair[1].title()}\nHuman n={sizes[0]} | Nash-Q n={sizes[1]} | Potential Max n={sizes[2]}',fontsize=9)
    axes[0,0].legend(fontsize=7)
    fig.suptitle(f'Human vs Nash-Q vs Potential Max — nine physical position pairs | Center ±{band:g} m\n'+
        'Human: four boxes + Center Safe override; robot: selected skills | lines: 95% cluster-bootstrap intervals',fontsize=13)
    fig.savefig(out/'human_nash_q_potential_max_nine_pairs_grouped_bars.png',dpi=300,bbox_inches='tight');plt.close(fig)
    # Literal three-panel side-by-side figures, one independently usable per pair.
    for pair in PAIRS:
        fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
        for ax,model in zip(axes,MODELS):
            g=cell(data,model,pair).rename(columns={'attack_skill':'landing_category'})
            draw_bars(ax,g,pair,NAMES[model])
            ax.set_ylabel('Probability of observed landing category' if model=='human_both' else 'Probability of selected skill')
        fig.suptitle(f'Incoming context | position Center ±{band:g} m | human regions: four boxes + Center Safe override',fontsize=12)
        fig.savefig(out/f'context_{pair[0]}_{pair[1]}_human_nash_q_potential_max.png',dpi=300,bbox_inches='tight');plt.close(fig)
    # Matching category colors/layouts also permit comparing the full grids.
    for model in MODELS:
        fig,axes=plt.subplots(3,3,figsize=(14,12),layout='constrained')
        for ax,pair in zip(axes.flat,PAIRS):
            g=cell(data,model,pair).rename(columns={'attack_skill':'landing_category'})
            draw_bars(ax,g,pair)
            ax.set_ylabel('Probability of landing category' if model=='human_both' else 'Probability of selected skill')
        fig.suptitle(NAMES[model]+f' | nine physical position pairs | Center ±{band:g} m',fontsize=14)
        fig.savefig(out/f'{model}_nine_pairs_bars.png',dpi=300,bbox_inches='tight');plt.close(fig)
    metrics=pd.read_csv(args.robot_dir/'joint_physical_similarity_metrics.csv')
    metrics=metrics[metrics.model.isin(MODELS[1:])]
    for r in metrics.itertuples():
        pair=(r.p1_position,r.p2_position)
        assert r.human_n==cell(data,'human_both',pair).n.iloc[0]
        assert r.model_n==cell(data,r.model,pair).n.iloc[0]
    metrics.to_csv(out/'human_vs_nash_q_and_potential_max_metrics.csv',index=False)
    inputs=[args.human_dir/'summary.json',args.human_dir/'human_reclassified_landing_points.csv',
        args.robot_dir/'audit.json',args.robot_dir/'conditional_distributions.csv',args.robot_dir/'joint_physical_similarity_metrics.csv']
    audit=dict(classification=hmeta['classification'],position_center_half_band_m=band,position_time='incoming_crossing',
        totals=data.groupby('model')['count'].sum().to_dict(),human_categories_recomputed_from_coordinates=True,
        sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs])
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    rows=['# Human rectangular landing categories versus Nash-Q and Potential Max',
        'Human P1+P2 observations are reclassified directly from depth/lateral coordinates using four_quadrants_center_override. The exact same 45 human context/category counts from the human-only charts are used. Robot bars are empirical logged selected skills, not measured landings. No nearest-target human labels enter these figures.',
        'Fixed human landing regions: depth split 0.51; lateral split 0; Center Safe depth [0.20,0.70] and lateral [-0.35,0.35], inclusive and overriding all base boxes. Exact configuration: `'+json.dumps(hmeta['classification'],sort_keys=True)+'`.',
        f'All contexts use incoming-crossing positions and the same normalized position-center half-band ±{band:g} m. Physical pair order is (P1,P2), each position relative to that player’s own side. Totals: `'+json.dumps(audit['totals'])+'`.',
        'The main 3×3 figure groups Human, Nash-Q, and Potential Max bars at each category; its intervals are copied from the verified existing 2,000-replicate match/rally cluster bootstrap. Each of the nine contexts additionally has a three-panel figure with category colors, probabilities, counts and n. Separate complete grids for each dataset use matching layouts.',
        'Human bars pool both human players; robot attacks are P1. Thus physical pairs mix attacker roles for human P2 shots. These figures describe physical-context category distributions rather than exactly role-matched behavior. Human categories condition on valid reconstructed landings, while robot selections include failures. JS/correlations remain exploratory proxy comparisons, not equality of intentions or physical placement.',
        '![Grouped nine-pair comparison](human_nash_q_potential_max_nine_pairs_grouped_bars.png)',
        '## Individual context comparisons']
    rows += [f'![{a.title()}/{b.title()}](context_{a}_{b}_human_nash_q_potential_max.png)' for a,b in PAIRS]
    rows += ['## CSV outputs',
        'nine_pair_distributions_side_by_side.csv puts category probabilities and counts next to each other for each pair. human_nash_q_potential_max_distributions.csv includes the cached intervals. human_vs_nash_q_and_potential_max_metrics.csv retains JS, TV, Spearman, Pearson and interval columns from the verified analysis.',
        '## Reproduce','```bash\npython3 v3/human_comparison/plot_human_robot_skill_bars.py\n```']
    (out/'report.md').write_text('\n\n'.join(rows)+'\n')
    print(json.dumps(audit['totals'],indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--human-dir',type=Path,default=Path('v3/human_comparison/reports/human_rectangular_skill_bars_center_0p25'))
    p.add_argument('--robot-dir',type=Path,default=Path('v3/human_comparison/reports/potential_max_vs_human_center_0p25'))
    p.add_argument('--output-dir',type=Path,default=Path('v3/human_comparison/reports/human_nash_q_potential_max_side_by_side_center_0p25'))
    run(p.parse_args())


if __name__=='__main__':main()
