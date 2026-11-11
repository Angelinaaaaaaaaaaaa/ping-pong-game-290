#!/usr/bin/env python3
"""Read-only v2 spatial comparison from saved human and robot events."""
import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import sys
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from v3.human_comparison.spatial import Regions,definitions,region_from_dict,position_labels,conditional_statistics,CATEGORIES
from v3.human_comparison.events import HALF_WIDTH,normalized_landing,target_prototypes
from v3.analyze_conditional_skill_selection import reconstruct_gantries
from nash_skills.skills import SKILL_NAMES

PAIRS=list(itertools.product(('left','center','right'),repeat=2))
COLORS=[*plt.get_cmap('tab10')(np.arange(5)),(.85,.85,.85,1)]


def human_events(path, players=(1,)):
    f=pd.read_csv(path/'shot_events.csv');f=f[f.attacker_id.isin(players)].copy()
    all_events=pd.read_csv(path/'all_shot_events.csv')
    lookup={(r.rally_id,int(r.shot_index)):r for r in all_events.itertuples()}
    for p in (1,2):f[f'p{p}_incoming_y_m']=np.nan
    f['incoming_frame']=np.nan;f['incoming_alignment']='unavailable_use_separate_pre_contact_sensitivity'
    for index,r in f.iterrows():
        previous=lookup.get((r.rally_id,int(r.shot_index)-1))
        if previous is None:continue
        if (previous.attacker_id==3-r.attacker_id and previous.interval_end_frame==r.contact_frame and
            previous.valid_net_crossing and previous.valid_opponent_landing and
            previous.contact_frame<previous.crossing_frame<r.contact_frame):
            for p in (1,2):f.loc[index,f'p{p}_incoming_y_m']=getattr(previous,f'p{p}_crossing_y_m')
            f.loc[index,'incoming_frame']=previous.crossing_frame
            f.loc[index,'incoming_alignment']='preceding_validated_opponent_shot'
    f['dataset']='human_p1';f['cluster']=f.rally_id.str.rsplit('_',n=1).str[0]
    f['rally_cluster']=f.rally_id;f['selected_skill']=None
    f['depth']=f.landing_depth_normalized;f['lateral']=f.landing_lateral_normalized
    f['valid_landing']=f.classified.astype(bool) # Preserve the exact accepted baseline cohort.
    f['landing_observed']=f.valid_landing
    f['attempt_basis']='detected_human_racket_contact'
    f['landing_status']=np.where(f.valid_landing,'valid_reconstructed_landing','excluded_reconstruction_not_known_failure')
    for time,old in [('outgoing','crossing'),('pre_contact','pre_contact'),('incoming','incoming')]:
        for p in (1,2):f[f'p{p}_{time}_own_lateral']=(1 if p==1 else -1)*f[f'p{p}_{old}_y_m']/HALF_WIDTH
    f['p1_own_lateral']=f.p1_incoming_own_lateral;f['p2_own_lateral']=f.p2_incoming_own_lateral
    f['position_time']='incoming_crossing';return f


def robot_events(file,policy):
    f=pd.read_csv(file);f=f[(f.attacker_id==1)&(f.attacker_policy==policy)].copy()
    points=[reconstruct_gantries(json.loads(r.state_json),int(r.attacker_id),int(r.defender_id)) for r in f.itertuples()]
    f['p1_own_lateral']=[p[1][1]/.7625 for p in points];f['p2_own_lateral']=[-p[2][1]/.7625 for p in points]
    for player in (1,2):
        for axis,k in [('x',0),('y',1)]:f[f'p{player}_decision_{axis}_m']=[p[player][k] for p in points]
    f['dataset']=policy;f['cluster']=f.episode_index.astype(str);f['rally_cluster']=f['cluster']
    f['selected_skill']=f.attack_skill;f['position_time']='incoming_crossing'
    f['attempt_basis']='selected_robot_high_level_decision'
    outcomes=[json.loads(v) if isinstance(v,str) and v.strip() else {} for v in f.get('realized_shot_outcome_json',pd.Series(['']*len(f),index=f.index))]
    f['landing_observed']=[bool(o.get('observation_complete',False)) for o in outcomes]
    f['valid_landing']=pd.array([o.get('valid_opponent_landing',False) if o.get('observation_complete',False) else pd.NA for o in outcomes],dtype='boolean')
    f['depth']=[o.get('landing_depth_normalized',np.nan) for o in outcomes]
    f['lateral']=[o.get('landing_lateral_normalized',np.nan) for o in outcomes]
    f['landing_status']=[o.get('exclusion_reason','measured') if o else 'not_logged' for o in outcomes]
    f['actual_racket_contact_count']=[len(o.get('racket_contacts',[])) if o else np.nan for o in outcomes]
    f['source_file']=str(file)
    meta=file.with_name('summary.json');m=json.loads(meta.read_text()).get('metadata',{}) if meta.exists() else {}
    f['seed']=m.get('seed')
    columns=['dataset','cluster','rally_cluster','seed','episode_index','decision_index','attacker_id','defender_id',
        'selected_skill','position_time','attempt_basis','p1_own_lateral','p2_own_lateral',
        'p1_decision_x_m','p1_decision_y_m','p2_decision_x_m','p2_decision_y_m',
        'valid_landing','landing_observed','depth','lateral','landing_status','actual_racket_contact_count','source_file',
        'contact_observed','successful_return_observed','attacker_miss','attacker_out_of_bounds','termination_reason','realized_shot_outcome_json']
    return f[[c for c in columns if c in f]],m


def background(ax,region):
    if region.name=='four_quadrants_center_override':
        from matplotlib.patches import Rectangle
        for y0,y1,short,deep in [(region.lateral_split,1,'left_short','left'),(-1,region.lateral_split,'right_short','right')]:
            for x0,x1,category in [(0,region.depth_split,short),(region.depth_split,1,deep)]:
                ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor=COLORS[CATEGORIES.index(category)],alpha=.2,linewidth=0))
        ax.add_patch(Rectangle((region.center_depth_min,region.center_lateral_min),
            region.center_depth_max-region.center_depth_min,region.center_lateral_max-region.center_lateral_min,
            facecolor=COLORS[2],alpha=1,linewidth=0))
        # Opaque pale center replaces rather than blends the underlying colors.
        ax.patches[-1].set_facecolor(tuple(.2*np.asarray(COLORS[2][:3])+.8))
        partition_outlines(ax,region)
    else:
        d,l=np.meshgrid(np.linspace(0,1,151),np.linspace(-1,1,161));labels=region.classify(d,l)
        c=np.array([[CATEGORIES.index(v) if v in CATEGORIES else 5 for v in row] for row in labels])
        ax.pcolormesh(d,l,c,cmap=ListedColormap(COLORS),vmin=0,vmax=5,alpha=.20,shading='auto',rasterized=True)
    ax.set(xlim=(0,1),ylim=(-1,1));ax.axhline(0,color='gray',lw=.5)


def partition_outlines(ax,region,color='black'):
    if region.name!='four_quadrants_center_override':return
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((region.center_depth_min,region.center_lateral_min),
        region.center_depth_max-region.center_depth_min,region.center_lateral_max-region.center_lateral_min,
        fill=False,edgecolor=color,linewidth=.6))
    ys=[(-1,1)]
    if region.center_depth_min<=region.depth_split<=region.center_depth_max:
        ys=[(-1,region.center_lateral_min),(region.center_lateral_max,1)]
    for lo,hi in ys:ax.plot([region.depth_split]*2,[lo,hi],color=color,lw=.6)
    xs=[(0,1)]
    if region.center_lateral_min<=region.lateral_split<=region.center_lateral_max:
        xs=[(0,region.center_depth_min),(region.center_depth_max,1)]
    for lo,hi in xs:ax.plot([lo,hi],[region.lateral_split]*2,color=color,lw=.6)


def save(fig,path):
    fig.tight_layout();fig.savefig(path,dpi=220,bbox_inches='tight');plt.close(fig)


def context(frame,pair,band):
    return frame[(position_labels(frame.p1_own_lateral,band)==pair[0])&(position_labels(frame.p2_own_lateral,band)==pair[1])]


def panel(ax,frame,region,pair,band,density=False,vmax=1):
    sub=context(frame,pair,band);known=bool(frame.landing_observed.any())
    valid=sub[sub.valid_landing & sub.depth.notna() & sub.lateral.notna()]
    unmapped=int((region.classify(valid.depth,valid.lateral)=='other/unmapped').sum())
    background(ax,region)
    if not known:
        ax.text(.5,.5,'REALIZED LANDINGS\nUNAVAILABLE',transform=ax.transAxes,ha='center',fontsize=9)
    elif density and len(valid)>=20:
        mass,xe,ye=np.histogram2d(valid.depth,valid.lateral,bins=[np.linspace(0,1,21),np.linspace(-1,1,21)])
        mass=mass/len(valid);ax.pcolormesh(xe,ye,mass.T,vmin=0,vmax=vmax,cmap='magma',shading='auto',rasterized=True)
        partition_outlines(ax,region,color='white')
    elif len(valid):
        ax.scatter(valid.depth,valid.lateral,s=5,alpha=.20,color='black',rasterized=True)
        if density:ax.text(.02,.02,'Sparse: scatter shown',transform=ax.transAxes,fontsize=7)
    else:ax.text(.5,.5,'No observed valid landings',transform=ax.transAxes,ha='center',fontsize=8)
    ax.set_title(f'{pair[0]}/{pair[1]} | attempts={len(sub)}\n'+(f'n={len(valid)}, unmapped={unmapped}' if known else 'n=unknown'),fontsize=9)


def grid(frame,region,band,out,name,density=False,vmax=1):
    fig,axes=plt.subplots(3,3,figsize=(11,10))
    for ax,pair in zip(axes.flat,PAIRS):
        panel(ax,frame,region,pair,band,density,vmax)
        ax.set(xlabel='Opponent-half depth (0 net → 1 endline)',ylabel='Hitter-left lateral (−1 right → +1 left)')
    fig.suptitle(name+f' | {region.name} regions | band ±{band:g} m / robot half-width\n'+('Mass per 20×20 bin within context; shared scale; no smoothing' if density else 'Black = observed valid landings; missing outcomes are never filled with targets'))
    if density:
        from matplotlib.cm import ScalarMappable
        fig.subplots_adjust(right=.87,top=.88,hspace=.5,wspace=.35)
        cax=fig.add_axes([.94,.25,.018,.5])
        fig.colorbar(ScalarMappable(norm=plt.Normalize(0,vmax),cmap='magma'),cax=cax,label='Conditional probability per bin')
        fig.savefig(out/f'{name}.png',dpi=220,bbox_inches='tight');plt.close(fig)
    else:save(fig,out/f'{name}.png')


def run(args):
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    if out.resolve()==args.human_dir.resolve():raise ValueError('Use a new output directory; preserve the baseline human run')
    regions=[region_from_dict(r) for r in json.loads(args.regions_json.read_text())] if args.regions_json else definitions()
    for r in regions:r.validate()
    primary=next((r for r in regions if r.name==args.primary_method),None) if args.primary_method else regions[0]
    if primary is None:raise ValueError(f'Unknown primary method: {args.primary_method}')
    if primary.name=='four_quadrants_center_override' and out.resolve()==Path('v3/human_comparison/reports/spatial_comparison_v2').resolve():
        raise ValueError('Choose a new --output-dir to preserve spatial_comparison_v2')
    primary_categories=getattr(primary,'categories',CATEGORIES)
    (out/'ROBOT_LOGGING_VALIDATION.md').write_text((Path(__file__).parent/'ROBOT_LOGGING_VALIDATION.md').read_text())
    (out/'region_definitions.json').write_text(json.dumps([asdict(r) for r in regions],indent=2))
    human=human_events(args.human_dir);robots=[];audit={}
    for policy,file in [('nash_q',args.nash_decisions),('potential_ibr',args.ibr_decisions)]:
        if file.exists():
            f,meta=robot_events(file,policy);robots.append(f)
            audit[policy]=dict(source=str(file),attempts=len(f),measured_outcomes=int(f.landing_observed.sum()),metadata=meta,
                missing_checkpoints=[p for p in [meta.get('model'),meta.get('potential_model')] if p and not Path(p).exists()])
    for frame in [human,*robots]:
        for definition in regions:
            labels=definition.classify(frame.depth,frame.lateral)
            frame['landing_region_'+definition.name]=np.where(frame.valid_landing.fillna(False).astype(bool),labels,None)
        frame['geometric_region']=frame['landing_region_'+primary.name]
        bad=(frame[['p1_own_lateral','p2_own_lateral']].abs()>args.max_abs_position).any(axis=1)
        frame['position_quality_flag']=np.where(bad,'extreme_lateral_position_excluded_from_primary_context','')
        for column in ('p1_own_lateral','p2_own_lateral'):
            frame.loc[frame[column].abs()>args.max_abs_position,column]=np.nan
    data=pd.concat([human,*robots],ignore_index=True)
    print(f'Loaded {len(human)} human attempts and {sum(len(f) for f in robots)} policy-owned P1 robot decisions; no trajectory replay.',flush=True)
    data['valid_landing']=data.valid_landing.astype('boolean')
    human.to_csv(out/'human_standardized_events.csv',index=False)
    data.to_csv(out/'standardized_events.csv',index=False)
    (out/'robot_landing_audit.json').write_text(json.dumps(audit,indent=2))
    valid=human[human.valid_landing];transitions=[];classification=[]
    fig,axes=plt.subplots(1,len(regions)+1,figsize=(5*(len(regions)+1),5),squeeze=False)
    from v3.human_comparison.events import landing_category
    saved_meta=json.loads((args.human_dir/'summary.json').read_text())
    saved_prototypes=saved_meta.get('target_prototypes_normalized',{})
    prototypes=np.array([saved_prototypes[s] for s in SKILL_NAMES]) if all(s in saved_prototypes for s in SKILL_NAMES) else target_prototypes('aggressive')
    nearest_labels=np.array(SKILL_NAMES)[np.argmin(((valid[['depth','lateral']].to_numpy()[:,None,:]-prototypes)**2).sum(axis=2),axis=1)]
    baseline_mismatch=int((nearest_labels!=valid.human_skill_proxy).sum())
    d,l=np.meshgrid(np.linspace(.001,1,151),np.linspace(-.999,.999,161))
    nearest=np.argmin(np.sum((np.stack([d,l],axis=-1)[...,None,:]-prototypes)**2,axis=-1),axis=-1)
    axes[0,0].pcolormesh(d,l,nearest,cmap=ListedColormap(COLORS[:5]),vmin=0,vmax=4,alpha=.2)
    axes[0,0].scatter(valid.depth,valid.lateral,s=1,alpha=.05,c='black');axes[0,0].set_title('Preserved nearest-target baseline')
    for ax,region in zip(axes.flat[1:],regions):
        labels=region.classify(valid.depth,valid.lateral);changed=int((labels!=valid.human_skill_proxy).sum())
        for category in getattr(region,'categories',CATEGORIES):
            count=int((labels==category).sum());classification.append(dict(method=region.name,category=category,count=count,percentage=100*count/len(valid),n=len(valid),changed_from_nearest=changed,changed_percentage=100*changed/len(valid)))
        table=pd.crosstab(valid.human_skill_proxy,pd.Series(labels,index=valid.index),dropna=False).reindex(index=SKILL_NAMES,columns=getattr(region,'categories',CATEGORIES),fill_value=0)
        table.to_csv(out/f'transition_nearest_to_{region.name}.csv')
        background(ax,region);ax.scatter(valid.depth,valid.lateral,s=1,alpha=.05,c='black');ax.set_title(f'{region.name}: {changed}/{len(valid)} switched')
    for category in SKILL_NAMES:
        n=int((valid.human_skill_proxy==category).sum());classification.append(dict(method='preserved_nearest',category=category,count=n,percentage=100*n/len(valid),n=len(valid),changed_from_nearest=0,changed_percentage=0))
    for ax in axes.flat:ax.set(xlabel='Opponent-half depth',ylabel='Hitter-left lateral',xlim=(0,1),ylim=(-1,1))
    save(fig,out/'classification_method_regions_and_landings.png')
    pd.DataFrame(classification).to_csv(out/'classification_sensitivity.csv',index=False)
    compare_human_conditionals(human,regions,primary,prototypes,args.primary_band,out)
    # Separate pose time/quality cohorts and use identical normalized bands in both domains.
    sensitivities=[];imbalance=[];fig,axes=plt.subplots(2,2,figsize=(11,7))
    for p in (1,2):
        limit=max(3.,float(human.loc[human.valid_landing,[f'p{p}_{t}_own_lateral' for t in ['outgoing','pre_contact','incoming']]].abs().max().max())+.01)
        for time in ['outgoing','pre_contact','incoming']:
            values=human.loc[human.valid_landing,f'p{p}_{time}_own_lateral'].dropna()
            axes[0,p-1].hist(values,bins=np.linspace(-limit,limit,121),density=True,histtype='step',label=time)
        axes[0,p-1].set(title=f'Human P{p}: signed own-side y / half-width',xlabel='Positive = own left',ylabel='Density');axes[0,p-1].legend()
        for f in robots:
            axes[1,p-1].hist(f[f'p{p}_own_lateral'].dropna(),bins=np.linspace(-3,3,81),density=True,histtype='step',label=f.dataset.iloc[0])
        axes[1,p-1].set(title=f'Robot P{p}: actual gantry y / half-width',xlabel='Positive = own left',ylabel='Density');axes[1,p-1].legend()
    save(fig,out/'raw_lateral_distributions.png')
    global_rows=[];fig,axes=plt.subplots(1,2,figsize=(11,4))
    for p,ax in zip((1,2),axes):
        for time in ['outgoing','pre_contact','incoming']:
            values=human.loc[human.valid_landing,f'p{p}_{time}_own_lateral'].dropna()
            tails=int((values.abs()>3).sum())
            ax.hist(values,bins=np.linspace(-3,3,81),density=True,histtype='step',label=f'{time}; {tails} outside view')
            global_rows.append(dict(player=p,position_time=time,n=len(values),minimum=values.min(),maximum=values.max(),median=values.median(),own_left_sign_fraction=(values>0).mean(),outside_normalized_3=tails))
        ax.set(title=f'P{p} central view (tails reported)',xlabel='Own-side lateral / half-width',ylabel='Density within shown range');ax.legend(fontsize=7)
    save(fig,out/'central_lateral_distributions_with_tail_counts.png')
    pd.DataFrame(global_rows).to_csv(out/'raw_lateral_global_summary.csv',index=False)
    for time in ['outgoing','pre_contact','incoming']:
        copy=human.copy()
        for p in (1,2):copy[f'p{p}_own_lateral']=copy[f'p{p}_{time}_own_lateral']
        datasets=[copy,*robots] if time=='incoming' else [copy]
        for f in datasets:
            for cohort in ['all_attempts','accepted_landings']:
                subset=f if cohort=='all_attempts' else f[f.valid_landing]
                for band in args.center_bands:
                    stats=conditional_statistics(subset,primary,band).drop_duplicates(['dataset','p1_position','p2_position'])
                    stats['position_time']=time;stats['cohort']=cohort;sensitivities.append(stats)
                for match,group in subset.groupby('cluster'):
                    for p in (1,2):
                        v=group[f'p{p}_own_lateral'].dropna()
                        imbalance.append(dict(dataset=f.dataset.iloc[0],position_time=time,cohort=cohort,cluster=match,player=p,n=len(v),mean=v.mean(),median=v.median(),positive_fraction=(v>0).mean() if len(v) else np.nan,left_at_baseline=(v>.05/.7625).mean() if len(v) else np.nan))
    sensitivity=pd.concat(sensitivities);sensitivity.to_csv(out/'position_threshold_sensitivity.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(17,4))
    for ax,time in zip(axes,['outgoing','pre_contact','incoming']):
        for offset,band in enumerate(args.center_bands):
            sub=sensitivity[(sensitivity.dataset=='human_p1')&(sensitivity.cohort=='accepted_landings')&(sensitivity.position_time==time)&np.isclose(sensitivity.band_m,band)].set_index(['p1_position','p2_position'])
            values=[sub.loc[p,'context_attempt_frequency'] if p in sub.index else np.nan for p in PAIRS]
            ax.bar(np.arange(9)+(offset-1)*.24,values,width=.24,label=f'±{band:g} m')
        ax.set_xticks(range(9),[a[0].upper()+b[0].upper() for a,b in PAIRS]);ax.set(title=time,xlabel='P1/P2 own-side contexts',ylabel='Fraction of known contexts');ax.legend(fontsize=8)
    save(fig,out/'human_position_context_threshold_sensitivity.png')
    pd.DataFrame(imbalance).to_csv(out/'player_match_lateral_imbalance.csv',index=False)
    stats=conditional_statistics(data,primary,args.primary_band);stats.to_csv(out/'context_landing_statistics.csv',index=False)
    print('Wrote classification, timing and position-band sensitivity tables; generating common-geometry figures.',flush=True)
    density_rows=[];mass_max=0
    for dataset,f in data.groupby('dataset'):
        for pair in PAIRS:
            sub=context(f,pair,args.primary_band);v=sub[sub.valid_landing];counts,xe,ye=np.histogram2d(v.depth,v.lateral,bins=[np.linspace(0,1,21),np.linspace(-1,1,21)])
            if len(v)>=20:mass_max=max(mass_max,float(counts.max()/len(v)))
            for i,j in itertools.product(range(20),repeat=2):
                available=bool(f.landing_observed.any())
                density_rows.append(dict(dataset=dataset,p1_position=pair[0],p2_position=pair[1],depth_bin=i,lateral_bin=j,count=int(counts[i,j]) if available else np.nan,n_valid=len(v) if available else np.nan,probability=counts[i,j]/len(v) if len(v) else np.nan,outcomes_available=available))
    pd.DataFrame(density_rows).to_csv(out/'density_bins_counts_and_mass.csv',index=False)
    for dataset,f in data.groupby('dataset'):
        suffix='' if f.landing_observed.any() else '_UNAVAILABLE'
        grid(f,primary,args.primary_band,out,dataset+'_nine_context_scatter'+suffix)
        grid(f,primary,args.primary_band,out,dataset+'_nine_context_density'+suffix,True,max(mass_max,.001))
    for time in ['outgoing','pre_contact']:
        copy=human.copy()
        for p in (1,2):copy[f'p{p}_own_lateral']=copy[f'p{p}_{time}_own_lateral']
        grid(copy,primary,args.primary_band,out,'human_'+time+'_sensitivity_scatter')
    for pair in PAIRS:
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        for ax,name in zip(axes,['human_p1','nash_q','potential_ibr']):
            f=data[data.dataset==name]
            if len(f):panel(ax,f,primary,pair,args.primary_band)
            else:background(ax,primary);ax.text(.5,.5,'Dataset unavailable',transform=ax.transAxes,ha='center')
            ax.set_xlabel('Opponent-half depth');ax.set_ylabel('Hitter-left lateral');ax.text(.5,1.23,name,transform=ax.transAxes,ha='center')
        save(fig,out/f'context_{pair[0]}_{pair[1]}_human_nash_ibr.png')
    skill_stats=[];confusion=[]
    for f in robots:
        policy=f.dataset.iloc[0];fig,axes=plt.subplots(1,5,figsize=(18,4))
        for ax,skill in zip(axes,SKILL_NAMES):
            group=f[f.selected_skill==skill];v=group[group.valid_landing];known=bool(group.landing_observed.any());labels=primary.classify(v.depth,v.lateral)
            counts={c:int((labels==c).sum()) for c in primary_categories}
            for c in primary_categories:confusion.append(dict(policy=policy,selected_skill=skill,realized_region=c,count=counts[c] if known else np.nan,probability_given_valid_landing=counts[c]/len(v) if len(v) else np.nan))
            measured=int(group.landing_observed.sum())
            contacts=int((group.actual_racket_contact_count>0).sum()) if group.actual_racket_contact_count.notna().any() else np.nan
            skill_stats.append(dict(policy=policy,selected_skill=skill,n_attempts=len(group),n_racket_contact_decisions=contacts,n_observed_outcomes=measured,n_unknown_outcomes=len(group)-measured,n_valid_landings=len(v) if known else np.nan,landing_success_fraction=len(v)/measured if measured else np.nan,landing_success_lower_bound_all_attempts=len(v)/len(group) if known and len(group) else np.nan,most_frequent_region=max(counts,key=counts.get) if len(v) else None,nominal_region_fraction=counts[skill]/len(v) if len(v) else np.nan))
            background(ax,primary)
            if len(v):ax.scatter(v.depth,v.lateral,s=3,alpha=.2,c='black')
            if not known:ax.text(.5,.5,'Realized landings unavailable',transform=ax.transAxes,ha='center',fontsize=8)
            ax.set_title(f'{skill}\nattempts={len(group)}; valid='+ (str(len(v)) if known else 'unknown'),fontsize=9)
        save(fig,out/f'{policy}_selected_skill_landing_distributions.png')
    pd.DataFrame(skill_stats).to_csv(out/'robot_selected_skill_outcome_statistics.csv',index=False)
    pd.DataFrame(confusion).to_csv(out/'robot_selected_skill_vs_realized_region.csv',index=False)
    for policy,cf in pd.DataFrame(confusion).groupby('policy'):
        matrix=cf.pivot(index='selected_skill',columns='realized_region',values='probability_given_valid_landing').reindex(index=SKILL_NAMES,columns=primary_categories)
        matrix.to_csv(out/f'{policy}_selected_skill_region_probability_matrix.csv')
        fig,ax=plt.subplots(figsize=(9,5));im=ax.imshow(np.ma.masked_invalid(matrix.to_numpy(float)),vmin=0,vmax=1,cmap='Blues')
        ax.set_xticks(range(len(primary_categories)),primary_categories,rotation=30,ha='right');ax.set_yticks(range(5),SKILL_NAMES)
        ax.set(xlabel='Measured landing region',ylabel='Selected high-level skill',title=policy+': P(region | selected skill, valid measured landing)')
        if matrix.notna().any().any():
            for i,j in itertools.product(range(5),range(len(primary_categories))):
                v=matrix.iloc[i,j]
                if pd.notna(v):ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=8)
        else:ax.text(.5,.5,'UNAVAILABLE: no measured robot landing outcomes',transform=ax.transAxes,ha='center',bbox=dict(facecolor='white',alpha=.9))
        fig.colorbar(im,ax=ax,label='Probability among valid measured landings');save(fig,out/f'{policy}_skill_vs_landing_confusion.png')
    missing=any(not f.landing_observed.any() for f in robots) or len(robots)<2
    (out/'distribution_comparison_status.json').write_text(json.dumps(dict(js_divergence=None,cluster_bootstrap_ci=None,
        status='deferred: no measured robot landing data' if missing else 'deferred: region definitions still exploratory; independent validation required',
        cluster_units='human match, robot rally; do not use independent-shot confidence intervals'),indent=2))
    findings=dict(human_attempts=len(human),human_valid_landings=int(human.valid_landing.sum()),
        incoming_context_attempts=int(human[['p1_own_lateral','p2_own_lateral']].notna().all(axis=1).sum()),
        incoming_context_valid_landings=int(human[human.valid_landing][['p1_own_lateral','p2_own_lateral']].notna().all(axis=1).sum()),
        incoming_valid_before_pose_filter=int(human[human.valid_landing][['p1_incoming_own_lateral','p2_incoming_own_lateral']].notna().all(axis=1).sum()),
        max_abs_position_normalized=args.max_abs_position,
        classification=classification,robot_audit=audit,primary_region=asdict(primary),primary_band_m=args.primary_band,shared_density_vmax=mass_max,
        saved_nearest_prototype_mismatches=baseline_mismatch)
    provenance=[]
    for path in [args.human_dir/'shot_events.csv',args.human_dir/'all_shot_events.csv',args.nash_decisions,args.ibr_decisions]:
        if path.exists():
            h=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
            provenance.append(dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest()))
    (out/'input_provenance.json').write_text(json.dumps(provenance,indent=2))
    (out/'summary.json').write_text(json.dumps(findings,indent=2))
    report(out,args,findings,pd.concat(sensitivities))
    print(json.dumps({k:v for k,v in findings.items() if k not in ('classification','robot_audit')},indent=2))


def compare_human_conditionals(human,regions,primary,prototypes,band,out):
    """Same-cohort comparison to nearest targets at each available pose time."""
    class NearestTarget:
        name='preserved_nearest'
        categories=CATEGORIES[:5]
        def classify(self,depth,lateral):
            points=np.column_stack([depth,lateral])
            return np.array(SKILL_NAMES)[np.argmin(((points[:,None,:]-prototypes)**2).sum(axis=2),axis=1)]
    comparisons=[];long_tables=[]
    for time in ('incoming','outgoing','pre_contact'):
        frame=human.copy()
        # Incoming uses the quality-screened primary poses. Keep historical
        # outgoing/pre-contact cohorts intact, as in the preceding analysis.
        if time!='incoming':
            for p in (1,2):frame[f'p{p}_own_lateral']=frame[f'p{p}_{time}_own_lateral']
        tables={}
        for method in [NearestTarget(),*regions]:
            table=conditional_statistics(frame,method,band)
            table['method']=method.name;table['position_time']=time;long_tables.append(table)
            tables[method.name]=table
            rows=[]
            for pair,cell in table.groupby(['p1_position','p2_position'],sort=False):
                row=dict(p1_position=pair[0],p2_position=pair[1],n=int(cell.n_valid_landings.iloc[0]),n_attempts=int(cell.n_attempts.iloc[0]))
                for category in getattr(method,'categories',CATEGORIES):
                    value=cell[cell.region==category].iloc[0]
                    row['count_'+category]=int(value.region_count)
                    row['probability_'+category]=value.probability_valid
                row['probability_vector_json']=json.dumps([row['probability_'+c] if pd.notna(row['probability_'+c]) else None for c in getattr(method,'categories',CATEGORIES)])
                rows.append(row)
            pd.DataFrame(rows).to_csv(out/f'human_conditional_distributions_{time}_{method.name}.csv',index=False)
        fig,axes=plt.subplots(3,3,figsize=(13,10),sharey=True)
        for ax,pair in zip(axes.flat,PAIRS):
            old=tables['preserved_nearest'];new=tables[primary.name]
            old=old[(old.p1_position==pair[0])&(old.p2_position==pair[1])].set_index('region').reindex(CATEGORIES[:5])
            new=new[(new.p1_position==pair[0])&(new.p2_position==pair[1])].set_index('region').reindex(CATEGORIES[:5])
            sub=context(frame,pair,band);valid=sub[sub.valid_landing]
            changed=int((primary.classify(valid.depth,valid.lateral)!=valid.human_skill_proxy).sum())
            for category in CATEGORIES[:5]:
                comparisons.append(dict(position_time=time,p1_position=pair[0],p2_position=pair[1],region=category,
                    n=len(valid),changed_from_nearest=changed,
                    nearest_probability=old.loc[category,'probability_valid'],
                    primary_probability=new.loc[category,'probability_valid'],
                    probability_difference=new.loc[category,'probability_valid']-old.loc[category,'probability_valid']))
            x=np.arange(5)
            ax.bar(x-.18,old.probability_valid,width=.36,label='Nearest-target',color='gray')
            ax.bar(x+.18,new.probability_valid,width=.36,label=primary.name,color='teal')
            ax.set_xticks(x,['L','LS','C','RS','R']);ax.set_ylim(0,1)
            ax.set_title(f'{pair[0]}/{pair[1]} | n={len(valid)} | changed={changed}',fontsize=9)
        axes[0,0].legend(fontsize=6);fig.supylabel('Conditional probability among valid landings')
        fig.suptitle(f'Nearest-target versus {primary.name} | {time} positions')
        save(fig,out/f'human_conditional_nearest_comparison_{time}.png')
    pd.concat(long_tables,ignore_index=True).to_csv(out/'human_conditional_sensitivity_all_methods.csv',index=False)
    comparison=pd.DataFrame(comparisons)
    comparison.to_csv(out/'human_conditional_nearest_comparison.csv',index=False)
    summaries=[]
    for (time,p1,p2),cell in comparison.groupby(['position_time','p1_position','p2_position'],sort=False):
        n=int(cell.n.iloc[0]);changed=int(cell.changed_from_nearest.iloc[0])
        delta=cell.probability_difference
        # The sixth legacy outcome, when present, is the complement of the
        # five named regions. For the complete partition this term is zero.
        tv=.5*(delta.abs().sum()+abs(delta.sum())) if n else np.nan
        center=cell[cell.region=='center_safe'].iloc[0]
        summaries.append(dict(position_time=time,p1_position=p1,p2_position=p2,n=n,
            changed_from_nearest=changed,changed_fraction=changed/n if n else np.nan,
            total_variation_distance=tv,nearest_center_probability=center.nearest_probability,
            primary_center_probability=center.primary_probability))
    pd.DataFrame(summaries).to_csv(out/'human_conditional_distribution_changes.csv',index=False)


def report(out,args,findings,sensitivity):
    ll=sensitivity[(sensitivity.dataset=='human_p1')&(sensitivity.cohort=='accepted_landings')&(sensitivity.p1_position=='left')&(sensitivity.p2_position=='left')]
    ll_table='| Position time | Center band (m) | LL count | LL fraction |\n|---|---:|---:|---:|\n'+'\n'.join(f'| {r.position_time} | {r.band_m:g} | {int(r.n_attempts)} | {r.context_attempt_frequency:.4f} |' for r in ll.itertuples())
    changes=pd.read_csv(out/'human_conditional_distribution_changes.csv')
    changes=changes[changes.position_time.isin(['incoming','outgoing'])]
    change_table='| Pose time | P1/P2 context | n | Changed labels | Total variation | Center: nearest → primary |\n|---|---|---:|---:|---:|---:|\n'+'\n'.join(
        f'| {r.position_time} | {r.p1_position}/{r.p2_position} | {r.n} | {r.changed_from_nearest} | {r.total_variation_distance:.4f} | {r.nearest_center_probability:.3f} → {r.primary_center_probability:.3f} |' for r in changes.itertuples())
    text=['# Human/robot realized spatial comparison v2',
        '**Robot realized landings are absent in the original decision logs. UNAVAILABLE panels mean unmeasured, not zero landings. No commanded target or ballistic prediction is used as an observed bounce.**',
        '## Inputs and coordinate audit',
        f"Read {findings['human_attempts']} saved P1 contact attempts, preserving all {findings['human_valid_landings']} accepted reconstructed landings. No LATTE-MV arrays were reread. Original files remain untouched. Robot source/configuration audit is in robot_landing_audit.json.",
        'Human coordinates are already meters: depth=x/(450×0.003048), lateral=y/(250×0.003048) for P1, with both signs reversed for P2. Robot coordinates normalize actual ball center as depth=(x−1.5)/1.37 and lateral=y/0.7625 for P1, reversing signs for P2. Player own-lateral position is +world-y for P1 and −world-y for P2, normalized by the corresponding table half-width. All context pairs are ordered physical P1/P2.',
        '## Landing regions and sensitivity',
        'The previous nearest-prototype method assigns every interior landing to one of five points. The center prototype has no explicit longitudinal bounds; its Voronoi cell extends through central lateral positions over a large depth range. This makes center_safe broad by construction, not evidence of an intentional safe human skill.',
        f"Primary method: **{findings['primary_region']['name']}**. Exact configuration: `{json.dumps(findings['primary_region'],sort_keys=True)}`. Full sensitivity definitions are in region_definitions.json and may be supplied with --regions-json. No boundaries are optimized for human/robot similarity.",
        'four_quadrants_center_override is a complete five-region partition: y >= lateral_split is left, y < lateral_split is right; d < depth_split is short, d >= depth_split is deep. The inclusive center rectangle always overrides the underlying quadrants. All finite in-table coordinates, including table edges, receive exactly one of five labels; invalid/outside coordinates remain excluded. Defaults are lateral_split=0, depth_split=0.51, center depth [0.20,0.70], center lateral [-0.35,0.35]. Every split and rectangle edge is configurable in JSON.',
        'Balanced, compact and broad preserve their historical bounded center, side/depth gaps and other/unmapped labels. Their rules and saved earlier outputs are unchanged. When a legacy rule has coincident short/deep depth boundaries, the exact shared boundary is assigned deep.',
        'classification_sensitivity.csv gives all category percentages and switch counts; transition_nearest_to_*.csv preserves the old-to-new transitions. The existing human_skill_proxy baseline is unchanged. All geometric labels describe realized placement, not intention.',
        f"Nearest-prototype reconstruction disagrees with {findings['saved_nearest_prototype_mismatches']} saved baseline labels; the baseline background uses saved prototype coordinates when available.",
        'human_conditional_distributions_{time}_{method}.csv gives one row per P1/P2 context: n, attempts, ordered category counts/probabilities and probability_vector_json. The new method has exactly five columns in category order left, left_short, center_safe, right_short, right. Empty contexts have undefined probabilities. human_conditional_nearest_comparison.csv reports per-context probability changes and changed-label counts using the same shots for both methods; corresponding nine-panel bar charts show n explicitly. Incoming, outgoing and pre-contact comparisons are separate. The outgoing table covers all 8,510 accepted shots; the primary incoming table covers only recoverable quality-screened contexts.',
        'The table below compares nearest-target and primary-method distributions within identical position contexts. Total variation is half the sum of absolute category-probability changes (including the unmapped complement for legacy methods); it describes classification sensitivity, not human/robot behavioral similarity. Small n limits interpretation.',
        change_table,
        '```json\n'+json.dumps(findings['classification'],indent=2)+'\n```',
        '## Defensive positioning and timing',
        'raw_lateral_distributions.png and player_match_lateral_imbalance.csv separate P1/P2 signed actual positions by time, match and cohort. position_threshold_sensitivity.csv recomputes all nine contexts at the same normalized center bands for human and robot. Bands ±0.05, ±0.15, ±0.25 m are divided by robot half-width .7625; humans use those same dimensionless bounds, rather than independently chosen thresholds. Human feet and robot gantry bases remain different physical measurements. A persistent sign imbalance cannot by itself distinguish stance behavior from camera/homography reconstruction bias.',
        f"Raw human pose outliers include physically suspect multi-meter lateral estimates; see raw_lateral_global_summary.csv (including all tails) and the separately labeled central view. Primary context assignments exclude |own lateral / half-width|>{findings['max_abs_position_normalized']:g} in either domain, with raw coordinates retained and a quality flag. This is a configurable review threshold, not a calibrated court-reach bound. It leaves {findings['incoming_context_valid_landings']} accepted incoming-context landings from {findings['incoming_valid_before_pose_filter']} before screening. The old outgoing/pre-contact threshold sensitivity tables intentionally preserve unscreened baseline cohorts.",
        ll_table,
        'The LL concentration decreases with wider center bands and is also present in pre-contact poses. Thus it is not solely an outgoing-time effect or the narrow original center bin. Player-relative P2 signs are reversed correctly in both time series; persistence does not resolve actual stance preference versus systematic reconstruction-side bias. The per-match table is required to distinguish this from domination by a few long matches; no athlete identity is inferred from the reconstruction slot.',
        f"Preferred incoming association is recoverable for {findings['incoming_context_attempts']} attempts and {findings['incoming_context_valid_landings']} valid landings. It requires the immediately preceding P2 shot, its interval ending at this P1 hit, a recorded directed crossing before the hit, and a validated preceding opponent-side landing. Its crossing poses are copied, never interpolated anew. Missing incoming contexts remain unknown in the primary comparison; pre-contact and outgoing analyses are separate sensitivity figures, not silently mixed fallbacks.",
        'The old outgoing 8,510-shot baseline remains available. Context counts in the preferred aligned human figure may be lower. Robot poses remain at incoming policy selection. This improves temporal alignment but human crossing coordinates still originate from inferred/interpolated reconstruction.',
        '## Outcome availability and logging',
        'Existing robot files retain command, incoming 74D state, episode/decision IDs, and inferred miss/OOB/contact flags. They have no first observed bounce coordinates. Planner actual_landing_prediction and printed successful-return x/y values are ballistic extrapolations, not measured landings. Sparse decision states cannot identify contact and bounce timing between decisions.',
        json.dumps(findings['robot_audit'],indent=2),
        'Measured robot landings are optional external inputs. This analysis can read realized_shot_outcome_json when supplied, but never generates physical measurements or substitutes predictions for them. Separate simulator instrumentation must validate contact/action association and noninterference before producing measured-landings datasets. See ROBOT_LOGGING_VALIDATION.md for the supported input fields. Missing measured outcomes remain unknown in legacy standardized events.',
        'This package does not run simulator evaluations, load model checkpoints, or change policy behavior. Checkpoint paths in source metadata are provenance only; source files may have moved or be unavailable locally. Robot landing maps require genuine measured coordinates in the input logs. Selected-skill comparisons require no new simulations or checkpoints.',
        '## Statistical interpretation',
        'context_landing_statistics.csv distinguishes attempts, measured outcomes, valid landings, mapped shots, primary-region probabilities conditional on valid landings, mapped-only probabilities, context visitation and unknown-context counts. The complete partition has five regions; legacy gap classifiers retain a sixth other/unmapped category. Human valid reconstruction coverage is not true shot success: excluded human shots may have succeeded. Robot unknown legacy outcomes are not failures. Density CSV retains absolute counts and per-context probability mass; all maps use common 20×20 bins, common color maximum and no smoothing. Contexts with fewer than 20 valid landings use scatter instead of a density image.',
        'Attempt bases differ: human rows are detected racket-contact onsets, while robot rows are high-level selections (some never reach a racket contact). attempt_basis and actual_racket_contact_count retain this distinction. Robot success fractions condition on measured, resolved outcomes; all-attempt lower bounds and unknown counts are separate. Truncation before a physical outcome is resolved remains unknown, not a failed landing.',
        'Skill-outcome tables retain selected robot skills separately. Without actual robot bounces, landing success, nominal-region fraction and realized confusion cells are blank. They must not be inferred from successful_return_observed. Context-specific Jensen–Shannon divergence and match/rally-cluster uncertainty are deferred until measured robot outcomes and region definitions are validated. High placement similarity would not establish shared intentions.',
        '## Reproduction',
        '```bash\npython3 v3/human_comparison/analyze_spatial_v2.py --human-dir '+str(args.human_dir)+' --nash-decisions '+str(args.nash_decisions)+' --ibr-decisions '+str(args.ibr_decisions)+' --regions-json '+str(out/'region_definitions.json')+' --primary-method '+findings['primary_region']['name']+' --center-bands '+' '.join(map(str,args.center_bands))+' --primary-band '+str(args.primary_band)+' --max-abs-position '+str(args.max_abs_position)+' --output-dir '+str(out)+'\n```',
        'Input SHA-256 hashes are in input_provenance.json. No original experiments are overwritten. ROBOT_LOGGING_VALIDATION.md describes optional measured-landing inputs; simulator instrumentation is separate from this analysis package.',
        '## Figures']
    text += [f'![{p.stem}]({p.name})' for p in sorted(out.glob('*.png'))]
    (out/'methodology_report.md').write_text('\n\n'.join(text)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--human-dir',type=Path,default=Path('v3/human_comparison/reports/10k_p1'))
    p.add_argument('--nash-decisions',type=Path,default=Path('skill_eval/v3/nash_vs_random_10000/decisions.csv'))
    p.add_argument('--ibr-decisions',type=Path,default=Path('skill_eval/v3/potential_ibr_vs_random_10000/decisions.csv'))
    p.add_argument('--output-dir',type=Path,default=Path('v3/human_comparison/reports/spatial_comparison_v2'))
    p.add_argument('--regions-json',type=Path)
    p.add_argument('--primary-method',help='Primary region name; defaults to the first JSON/default definition. Use four_quadrants_center_override with a new output directory.')
    p.add_argument('--center-bands',nargs='+',type=float,default=[.05,.15,.25])
    p.add_argument('--primary-band',type=float,default=.05)
    p.add_argument('--max-abs-position',type=float,default=3.,help='Same normalized pose-quality review bound for both datasets; raw values retained')
    args=p.parse_args()
    if args.max_abs_position<=0 or args.primary_band<0 or any(b<0 for b in args.center_bands):
        p.error('Pose-quality bound must be positive; center bands nonnegative')
    run(args)


if __name__=='__main__':main()
