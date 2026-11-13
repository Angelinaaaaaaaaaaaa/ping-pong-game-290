#!/usr/bin/env python3
"""Human-only nine-context bar charts using explicit rectangular landings.

Always reclassifies stored depth/lateral coordinates. Historical nearest-target
labels and robot action distributions are not inputs to these probabilities.
"""
import argparse
from dataclasses import asdict
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
from v3.human_comparison.spatial import region_from_dict
from v3.human_comparison.analyze_spatial_v2 import COLORS, background
from nash_skills.skills import SKILL_NAMES

POSITIONS = ('left', 'center', 'right')
PAIRS = tuple(itertools.product(POSITIONS, repeat=2))
LABELS = ('Left', 'Left\nShort', 'Center\nSafe', 'Right\nShort', 'Right')
COHORTS = (('both_players', None), ('p1', 1), ('p2', 2))


def summarize_humans(frame, region):
    f = frame.copy()
    f['human_landing_category'] = region.classify(f.depth, f.lateral)
    if not f.human_landing_category.isin(SKILL_NAMES).all():
        raise ValueError('Invalid/outside landing in accepted observations')
    if not f.attacker_id.isin([1,2]).all(): raise ValueError('Unexpected human attacker identity')
    if not f.p1_position.isin(POSITIONS).all() or not f.p2_position.isin(POSITIONS).all():
        raise ValueError('Unknown physical position context')
    rows = []
    for name, player in COHORTS:
        cohort = f if player is None else f[f.attacker_id == player]
        for p1, p2 in PAIRS:
            cell = cohort[(cohort.p1_position==p1)&(cohort.p2_position==p2)]
            for skill in SKILL_NAMES:
                count = int((cell.human_landing_category==skill).sum())
                rows.append(dict(player_cohort=name,p1_position=p1,p2_position=p2,n=len(cell),
                    landing_category=skill,count=count,probability=count/len(cell) if len(cell) else np.nan,
                    classification_method=region.name))
    return f, pd.DataFrame(rows)


def draw_bars(ax, cell, pair, title=None):
    cell = cell.set_index('landing_category').reindex(SKILL_NAMES)
    n = int(cell.n.iloc[0]); values = cell.probability.to_numpy(float)
    ax.set_ylim(0,1)
    if n:
        bars = ax.bar(range(5),values,color=COLORS[:5],edgecolor='white',linewidth=.6)
        for bar, value, count in zip(bars, values, cell['count']):
            ax.text(bar.get_x()+bar.get_width()/2,value+.018,f'{value:.1%}\n({int(count)})',
                    ha='center',va='bottom',fontsize=8)
    else:
        ax.text(.5,.5,'No observations\nprobabilities undefined',ha='center',transform=ax.transAxes)
    ax.set_xticks(range(5),LABELS,fontsize=9)
    ax.set_ylabel('Probability of realized landing category')
    ax.set_title((title+'\n' if title else '')+f'P1 {pair[0].title()} / P2 {pair[1].title()} | n={n}',fontsize=11)
    ax.grid(axis='y',alpha=.18); ax.set_axisbelow(True)


def run(args):
    source=args.input_dir; out=args.output_dir
    if out.resolve()==source.resolve(): raise ValueError('Use a separate output directory')
    audit=json.loads((source/'audit.json').read_text())
    region=region_from_dict(audit['human_region']);region.validate()
    if region.name!='four_quadrants_center_override':raise ValueError('Require four-box + Center Safe classifier')
    raw=pd.read_csv(source/'human_landing_points.csv')
    if raw.duplicated(['rally_id','shot_index','attacker_id']).any():raise ValueError('Duplicate human shot IDs')
    frame, stats=summarize_humans(raw,region)
    band=audit['center_band_m']
    if not np.allclose(raw.defense_center_half_band_m,band):raise ValueError('Inconsistent position-center band')
    # Compare against the rectangular counts in the current comparison run.
    reference=pd.read_csv(source/'conditional_distributions.csv')
    reference=reference[(reference.model==audit['human_dataset'])&(reference.scope=='joint_physical')]
    expected=reference.set_index(['p1_position','p2_position','attack_skill'])['count'].sort_index()
    actual=stats[stats.player_cohort=='both_players'].set_index(['p1_position','p2_position','landing_category'])['count'].sort_index()
    pd.testing.assert_series_equal(expected,actual,check_names=False,check_dtype=False)
    out.mkdir(parents=True,exist_ok=True)
    stats['position_center_half_band_m']=band
    stats.to_csv(out/'human_rectangular_conditional_distributions.csv',index=False)
    frame['classification_method']=region.name
    frame.to_csv(out/'human_reclassified_landing_points.csv',index=False)
    wide=[]
    for (cohort,p1,p2),g in stats.groupby(['player_cohort','p1_position','p2_position'],sort=False):
        g=g.set_index('landing_category').reindex(SKILL_NAMES)
        row=dict(player_cohort=cohort,p1_position=p1,p2_position=p2,n=int(g.n.iloc[0]))
        for skill in SKILL_NAMES:
            row[skill+'_count']=int(g.loc[skill,'count']);row[skill+'_probability']=g.loc[skill,'probability']
        row['probability_vector_json']=json.dumps([float(v) if pd.notna(v) else None for v in g.probability])
        wide.append(row)
    pd.DataFrame(wide).to_csv(out/'human_rectangular_conditional_distributions_wide.csv',index=False)
    for name,player in COHORTS:
        title='Both human players pooled' if player is None else f'Human P{player} shots only'
        data=stats[stats.player_cohort==name]
        fig,axes=plt.subplots(3,3,figsize=(14,12),layout='constrained')
        for ax,pair in zip(axes.flat,PAIRS):
            draw_bars(ax,data[(data.p1_position==pair[0])&(data.p2_position==pair[1])],pair)
        fig.suptitle(title+' — FOUR BOXES + CENTER SAFE OVERRIDE\n'+
            f'Incoming-aligned human landings | position Center ±{band:g} m | rows=P1, columns=P2',fontsize=14)
        fig.savefig(out/f'human_{name}_nine_pairs_rectangular_bars.png',dpi=300,bbox_inches='tight');plt.close(fig)
        # Individual files make each of the nine contexts independently usable.
        for pair in PAIRS:
            fig,ax=plt.subplots(figsize=(7,5),layout='constrained')
            draw_bars(ax,data[(data.p1_position==pair[0])&(data.p2_position==pair[1])],pair,title)
            fig.suptitle(f'Four boxes + Center Safe override | incoming | position Center ±{band:g} m',fontsize=11)
            fig.savefig(out/f'human_{name}_{pair[0]}_{pair[1]}_rectangular_bars.png',dpi=300,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5),layout='constrained');background(ax,region)
    for x,y,text in [(.10,.7,'Left Short'),(.82,.7,'Left'),(.45,0,'Center Safe'),(.10,-.7,'Right Short'),(.82,-.7,'Right')]:
        ax.text(x,y,text,ha='center',fontsize=11)
    ax.set(xlabel='Normalized opponent-half depth: 0 net → 1 endline',ylabel='Hitter-relative lateral: +left / −right',
        title='Landing-category definition used for every bar')
    fig.savefig(out/'four_boxes_center_safe_definition.png',dpi=300,bbox_inches='tight');plt.close(fig)
    summary=dict(source=str(source),classification=asdict(region),position_time='incoming_crossing',
        position_center_half_band_m=band,shots=len(frame),by_player=frame.groupby('attacker_id').size().to_dict(),
        verified_against_current_rectangular_comparison=True,
        sources=[dict(path=str(source/name),sha256=hashlib.sha256((source/name).read_bytes()).hexdigest())
                 for name in ['audit.json','human_landing_points.csv','conditional_distributions.csv']])
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'report.md').write_text('# Human-only rectangular landing-category bar charts\n\n'
        f'{len(frame):,} incoming-aligned human landings; position-center half-band ±{band:g} m. '
        'Physical pair order is (P1 position,P2 position), each position relative to that player’s own side. '
        'P1/P2 landing coordinates are already hitter-relative.\n\n'
        'Every bar is computed by reclassifying stored landing coordinates with four_quadrants_center_override. '
        'No nearest-target label or robot selected action is used. Raw counts are pooled before normalization. '
        'These are human realized-placement proxies, not intentional skill annotations.\n\n'
        'The four boxes split at depth 0.51 and lateral 0. The inclusive Center Safe rectangle, '
        'depth [0.20,0.70] and lateral [-0.35,0.35], overrides the base boxes. '
        'Every valid landing receives one of five labels. Exact configuration: `'+json.dumps(asdict(region))+'`.\n\n'
        'The 10k_both/human_P1_empirical_probability.png and human_P2_empirical_probability.png files '
        'belong to the original nearest-target extraction baseline. They are preserved. '
        'The new rectangular figures are in this separate directory and explicitly labeled. '
        'All 45 pooled counts match the current rectangular comparison.\n\n'
        '![Definition](four_boxes_center_safe_definition.png)\n\n'
        '![Both players](human_both_players_nine_pairs_rectangular_bars.png)\n\n'
        '![P1](human_p1_nine_pairs_rectangular_bars.png)\n\n'
        '![P2](human_p2_nine_pairs_rectangular_bars.png)\n\n'
        'Each overview also has nine separate context image files. Percentages and counts are printed on bars; '
        'n is shown for every condition. Empty conditions have undefined probabilities. '
        'No simulations or LATTE-MV extraction were rerun.\n\n'
        '```bash\npython3 v3/human_comparison/plot_human_rectangular_skill_bars.py\n```\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='sources'},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,default=Path('v3/human_comparison/reports/center_band_sensitivity_10k_both/center_0p25'))
    p.add_argument('--output-dir',type=Path,default=Path('v3/human_comparison/reports/human_rectangular_skill_bars_center_0p25'))
    run(p.parse_args())


if __name__=='__main__':main()
