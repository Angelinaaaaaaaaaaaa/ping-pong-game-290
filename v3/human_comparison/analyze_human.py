#!/usr/bin/env python3
"""Run a deterministic small LATTE-MV pilot directly from ZIP or extracted files."""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from v3.human_comparison.data import RallyStore, digest
from v3.human_comparison.events import Config, extract, UNIT_M, HALF_WIDTH, TABLE_Z, starts, body_positions, target_prototypes
from v3.human_comparison.plots import shot_plot, spatial_plots, video_sheet, comparison_plot
from v3.analyze_conditional_skill_selection import cell_rows, plot_grid
from v3.constants import DEFENSE_POSITIONS
from nash_skills.skills import SKILL_NAMES
import itertools

REFERENCE_COMMIT = 'bcada1f744af3e1c39fd5adb5ee5bb534f0d1ba2'


def conditional_tables(events, players):
    joint, opponent, frequencies, vectors = [], [], [], []
    for player in players:
        valid=events[(events.attacker_id==player)&events.classified].copy()
        valid['attack_skill']=valid.human_skill_proxy
        valid['policy_vector']=None  # Humans have no logged stochastic policy vector.
        for p1,p2 in itertools.product(DEFENSE_POSITIONS,repeat=2):
            sub=valid[(valid.p1_position==p1)&(valid.p2_position==p2)]
            cells=cell_rows(sub,'latte_pilot','human_landing_proxy_provisional',f'P{player}',{'p1_position':p1,'p2_position':p2})
            joint.extend(cells)
            frequencies.append(dict(player=f'P{player}',p1_position=p1,p2_position=p2,n_decisions=len(sub),
                condition_frequency=len(sub)/len(valid) if len(valid) else None))
            vectors.append(dict(player=f'P{player}',p1_position=p1,p2_position=p2,n=len(sub),
                empirical_probability_vector=[c['empirical_probability'] for c in cells] if len(sub) else None))
        for pos in DEFENSE_POSITIONS:
            field='p2_position' if player==1 else 'p1_position'
            sub=valid[valid[field]==pos]
            opponent.extend(cell_rows(sub,'latte_pilot','human_landing_proxy_provisional',f'P{player}',{'opponent_position':pos}))
    return pd.DataFrame(joint),pd.DataFrame(opponent),pd.DataFrame(frequencies),vectors


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('v3/LATTE-MV'))
    parser.add_argument('--sample-rallies',type=int,default=20)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--sample-manifest',type=Path,help='Replay exact member IDs from a previous sample_manifest.json')
    parser.add_argument('--output-dir',type=Path,default=Path('v3/human_comparison/reports'))
    parser.add_argument('--player',choices=['p1','p2','both'],default='p1')
    parser.add_argument('--position-time',choices=['crossing','pre_contact'],default='crossing')
    parser.add_argument('--position-landmark',choices=['feet_midpoint','pelvis'],default='feet_midpoint')
    parser.add_argument('--center-normalized',type=float,default=.05/.7625)
    parser.add_argument('--robot-profile',choices=['current','moderate_aggressive','aggressive'],default='aggressive')
    parser.add_argument('--edge-margin-m',type=float,default=.005)
    parser.add_argument('--net-clearance-m',type=float,default=0.)
    parser.add_argument('--net-height-m',type=float,default=.1524,help='Nominal height above reconstructed table; source net drawing is inconsistent')
    parser.add_argument('--video-examples',type=int,default=3)
    parser.add_argument('--robot-summary',nargs='*',type=Path)
    parser.add_argument('--verbose',action='store_true')
    args=parser.parse_args(argv)
    if args.sample_rallies<1 or args.center_normalized<0 or args.edge_margin_m<0 or args.video_examples<0 or args.net_height_m<0 or args.net_clearance_m<0:
        parser.error('Sample size must be positive; margins/counts nonnegative')
    out=args.output_dir
    if out.resolve().is_relative_to(args.data_root.resolve()):
        parser.error('Output directory must be outside the original dataset')
    out.mkdir(parents=True,exist_ok=True)
    config=Config(args.center_normalized,args.position_time,args.position_landmark,args.robot_profile,args.edge_margin_m,args.net_clearance_m,args.net_height_m)
    store=RallyStore(args.data_root)
    selected=([s['member'] for s in json.loads(args.sample_manifest.read_text())['samples']]
              if args.sample_manifest else store.sample(args.sample_rallies,args.seed))
    if any(n not in store.name_set for n in selected):
        raise ValueError('Replay manifest contains missing reconstruction members')
    players=[1,2] if args.player=='both' else [1 if args.player=='p1' else 2]
    inventory, rows, videos, failures=[],[],[],[]
    plotted=0
    try:
        for member in selected:
            rally=Path(member).parent.name;blob=store.read(member)
            item=dict(rally_id=rally,member=member,match_id=Path(member).parent.parent.name,sha256=digest(blob),uncompressed_bytes=len(blob))
            try:
                scene=np.load(io.BytesIO(blob),allow_pickle=False)
                item.update(shape=list(scene.shape),dtype=str(scene.dtype),metadata_first_three_frames=scene[:3,0].tolist(),
                    finite_ball_frames=int(np.isfinite(scene[:,90]).all(axis=1).sum()),
                    ball_nan_frames=int(np.isnan(scene[:,90]).any(axis=1).sum()),
                    all_zero_ball_frames=int((scene[:,90]==0).all(axis=1).sum()),
                    contact_active_frames=scene[:,1].sum(axis=0).astype(int).tolist(),
                    contact_onsets=[len(starts(scene[:,1,k])) for k in range(3)])
                for suffix in ('_metadata.json','_ball.csv','_table.csv','_table.npy'):
                    name=member.replace('_recon.npy',suffix)
                    if name not in store.name_set:continue
                    content=store.read(name)
                    if suffix.endswith('csv'):
                        frame=pd.read_csv(io.BytesIO(content));item[suffix]=dict(rows=len(frame),columns=list(frame.columns),sha256=digest(content))
                    elif suffix.endswith('json'):
                        md=json.loads(content);item[suffix]=dict(records=len(md),first_record=md[0] if isinstance(md,list) and md else None,sha256=digest(content))
                    else:
                        array=np.load(io.BytesIO(content),allow_pickle=False);item[suffix]=dict(shape=list(array.shape),dtype=str(array.dtype))
                events=extract(scene,rally,config)
                for row in events:row['source_reconstruction']=member
                rows.extend(events);item['loaded']=True
                good=[r for r in events if r['classified'] and r['attacker_id'] in players]
                if good and plotted<3:
                    shot_plot(scene[:,90]*UNIT_M,body_positions(scene,config.landmark),good[0],out,TABLE_Z+config.net_height_m);plotted+=1
                if good and len(videos)<args.video_examples:
                    videos.append(video_sheet(store,member,good,out))
                if args.verbose:print(rally,len(events),'contacts',len(good),'selected valid shots',flush=True)
            except (ValueError,KeyError,OSError,IndexError) as exc:
                item.update(loaded=False,error=str(exc));failures.append(dict(rally_id=rally,error=str(exc)))
            inventory.append(item)
    finally:store.close()
    (out/'sample_manifest.json').write_text(json.dumps(dict(seed=args.seed,numpy_version=np.__version__,reference_commit=REFERENCE_COMMIT,selection='manifest replay' if args.sample_manifest else 'sorted catalogue; shuffled matches, independently shuffled rally lists; round robin matches',
        catalogue_recon_count=len(store.recons),catalogue_sha256=digest('\n'.join(store.recons).encode()),samples=inventory),indent=2))
    if not rows:raise RuntimeError('No shot events loaded; inspect sample_manifest.json')
    events=pd.DataFrame(rows)
    # Include explicit missing fields for completely empty successful subsets.
    for column in ['p1_position','p2_position','landing_depth_normalized','landing_lateral_normalized']:
        if column not in events:events[column]=np.nan
    chosen=events[events.attacker_id.isin(players)].copy()
    events.to_csv(out/'all_shot_events.csv',index=False);chosen.to_csv(out/'shot_events.csv',index=False)
    joint,opponent,frequency,vectors=conditional_tables(chosen,players)
    joint.to_csv(out/'joint_position_skill_summary.csv',index=False)
    opponent.to_csv(out/'opponent_position_skill_summary.csv',index=False)
    frequency.to_csv(out/'position_combination_frequency.csv',index=False)
    for player in players:
        sub=joint[joint.player==f'P{player}']
        plot_grid(sub,'p1_position','p2_position',out/f'human_P{player}_empirical_probability.png','empirical_probability',
                  f'Human P{player}: PROVISIONAL landing proxy (not commanded skill)\n{args.position_time} positions; {len(selected)} sampled rallies')
    spatial_plots(chosen,config,out)
    sources=args.robot_summary if args.robot_summary is not None else list(Path('reports/v3').glob('conditional_skill*/joint_position_skill_summary.csv'))
    comparison_sources=comparison_plot(joint,[p for p in sources if p.exists()],out)
    summary=dict(total_sampled_rallies=len(selected),successfully_loaded_rallies=sum(i.get('loaded',False) for i in inventory),
        dataset=dict(input_mode='zip' if store.zip else 'extracted',catalogue_members=len(store.names),
            reconstruction_members=len(store.recons),distinct_matches=len({Path(n).parent.parent.name for n in store.recons}),
            archive_bytes=Path(store.zip.filename).stat().st_size if store.zip else None),
        selected_player=args.player,position_time=args.position_time,config=vars(config),
        all_players=dict(detected_racket_contacts=len(events),valid_net_crossings=int(events.valid_net_crossing.sum()),
            valid_opponent_landings=int(events.valid_opponent_landing.sum()),classified_shots=int(events.classified.sum()),
            excluded_by_reason=dict(Counter(events.loc[~events.classified,'exclusion_reason']))),
        selected_players=dict(detected_racket_contacts=len(chosen),valid_net_crossings=int(chosen.valid_net_crossing.sum()),
            valid_opponent_landings=int(chosen.valid_opponent_landing.sum()),classified_shots=int(chosen.classified.sum()),
            excluded_by_reason=dict(Counter(chosen.loc[~chosen.classified,'exclusion_reason']))),
        condition_vectors=vectors,robot_comparison_sources=comparison_sources,video_checks=videos,load_failures=failures,
        classification='provisional_nearest_v3_target',reference_commit=REFERENCE_COMMIT,target_prototypes_normalized=dict(zip(SKILL_NAMES,target_prototypes(args.robot_profile).tolist())))
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    write_reports(out,args,inventory,summary)
    print(json.dumps({k:summary[k] for k in ['total_sampled_rallies','successfully_loaded_rallies','all_players','selected_players']},indent=2))
    print('Saved pilot to',out)


def write_reports(out,args,inventory,summary):
    sources='''Official evidence (consulted directly; the pipeline does not require network access):

- [Dataset README](https://huggingface.co/datasets/ember-lab-berkeley/LATTE-MV/blob/main/README.md): scene layout and metadata.
- [Reconstruction serialization and feet anchoring](https://github.com/sastry-group/LATTE-MV/blob/main/RallyReconstructor/processed.py): feet landmarks 20/23; hit/bounce anchors; NaN serialization; table-bounce clipping.
- [Table geometry](https://github.com/sastry-group/LATTE-MV/blob/main/RallyReconstructor/utils/table.py): 900×500 planar frame and centered ground homography.
- [Official data loader](https://github.com/sastry-group/LATTE-MV/blob/main/GenerativeModel/loaders/youtube_loader.py): conversion factor 0.003048 and root joint 8.
- [Hit and bounce inference](https://github.com/sastry-group/LATTE-MV/blob/main/RallyReconstructor/utils/ball.py): image-motion/hand-distance heuristics, alternating hits, parabola fitting, and clipping.
- [Official visualization](https://github.com/sastry-group/LATTE-MV/blob/main/Visualize/visualize_recon.py): table corners (±450,±250,250) and body skeleton topology; net drawing has inconsistent vertical offsets.
'''
    sources=sources.replace('/blob/main/','/blob/'+REFERENCE_COMMIT+'/')
    investigation=['# LATTE-MV dataset investigation (small-sample pilot)',sources,
        'Reference-code revision: `'+REFERENCE_COMMIT+'` (2025-04-14). The archive manifest records input hashes and NumPy version for replay.',
        '## Local structure and sampling',
        'Observed input catalogue: '+json.dumps(summary['dataset'])+'. Only the sampled members were read. Typical member layout:\n\n```text\nrelease_data/match<ID>/match<ID>_<rally>/\n  match<ID>_<rally>_recon.npy\n  match<ID>_<rally>_metadata.json\n  match<ID>_<rally>_ball.csv\n  match<ID>_<rally>_table.csv\n  match<ID>_<rally>_paddle.csv\n  match<ID>_<rally>_keypoints_2d.npy\n  match<ID>_<rally>_keypoints_3d.npy\n  match<ID>_<rally>.mp4\n```',
        f'Input: `{args.data_root}`. Only the ZIP central directory or extracted file names are inventoried. {len(inventory)} selected reconstruction arrays are read, one at a time, plus their small auxiliary files and at most {args.video_examples} selected video clips. No full extraction or dataset download is performed. Seed={args.seed}. Match-balanced shuffled sampling chooses one rally per different match before revisiting a match; this is a validation sample, not a rally-uniform population sample.',
        'Archive observations: the downloaded release contains *_table.csv in the sample, not the README’s advertised *_table.npy. Table CSV coordinates are image pixels, not reconstructed 3D table corners. *_ball.csv has Frame/Visibility/X/Y (2D). PHALP *_metadata.json maps frame/object_id/keypoints_idx/rescale_factor, not ground-truth athlete identities or shot labels. Full per-rally shapes, columns, hashes, available files, finite-ball counts, contact counts and first metadata rows are in sample_manifest.json.',
        '## Confirmed scene frame and indices',
        '(frames,91,3) float64 arrays. Index 0 stores reconstruction metadata, not a body/ball location: first frame FPS/frame count/usable count; frame 1 scale factors/hand ID; frame 2 other hand ID. Index 1 is P1 racket/P2 racket/table flags. P1 body indices 2:46, P2 46:90, ball 90. Code validates shape, finite player locations and binary flags. Player pose estimates may be forward-filled by the source; usable count is not identical to finite-ball count.',
        'Source coordinate units are table-normalized reconstruction units, not centimeters or meters. Multiply xyz by 0.003048 (official loader). Table x=±450, y=±250, surface z=250 in raw units: length 2.7432 m, width 1.524 m, height 0.762 m. Net plane is x=0. x is longitudinal, y lateral, z up. This frame is fixed by the source homography, not fitted from human bounce samples. Physical scale is conventional and approximate, not per-video surveyed calibration.',
        f'**Unresolved net-height inconsistency:** visualize_recon.py draws net vertices at raw z=300..350 although the table is z=250; source MuJoCo visualization uses yet another vertical offset. Reconstructions contain no measured net geometry. This pilot uses an explicit nominal net height {args.net_height_m:g} m above the table (--net-height-m), plus {args.net_clearance_m:g} m center clearance (--net-clearance-m). The default nominal 0.1524 m gives top z=0.9144 m, raw z=300. These gates are plausibility assumptions, not verified per-video net calibration, and may explain some rejected trajectories. No claim of legal net clearance is established by the reconstruction alone.',
        'P1 occupies negative x and faces +x; P2 positive x faces −x. The sample validates median body positions on those halves without silently swapping identities. Own-left is +y for P1, −y for P2. IDs denote reconstruction slots per rally; they do not identify a consistent real athlete across matches or camera changes.',
        '## Player location and timing',
        'Default position is the projected midpoint of local landmarks 20/23 within each 44-point body block, the same foot landmarks used by processed.py to anchor skeletons to the ground. This is closer to a robot gantry base than a leaning head/hand. --position-landmark pelvis selects local joint 8 (official loader root) as a sensitivity alternative. Foot-midpoint is still a proxy for human base location; it is not a robot gantry measurement.',
        'Positions at the crossing are linearly interpolated between two adjacent pose frames; fractional crossing frame is saved. Positions at the immediately preceding frame contact_frame−1 are also retained (no future look-back substitution). --position-time selects which feeds the conditional tables. Crossing-time positions occur after the shot decision and can confound decision interpretation; pre-contact sensitivity should be checked before scientific comparison.',
        f'Own-side lateral position is normalized by table half-width. Center is inclusive |y/half-width|≤{args.center_normalized:.8g}; otherwise sign gives own-left/right. Default is the robot 0.05 m band divided by robot table half-width 0.7625 m; for humans the equivalent band is {args.center_normalized*HALF_WIDTH:.6g} m. It is not an equal-third court partition. Pair ordering is (physical P1 own-side position, physical P2 own-side position), matching the robotic grids.',
        '## Event association and reconstruction quality',
        'Consecutive contact flags collapse to their first active frame. Every racket event is bounded by the next racket event. We require an own-half hit with finite incoming/outgoing samples, reversal of longitudinal velocity toward the opponent, one directed adjacent-frame crossing of x=0 over the net span/top, exactly one subsequent table-bounce onset before the next hit, a fully finite hit-to-bounce path, an opponent-half interior landing at table height, and downward/upward vertical bounce velocities. Serves/self-half bounces and simultaneous or incomplete events are excluded. A valid crossing counter means a shot passed earlier gates as well; it is not the number of arbitrary sign changes in all raw frames.',
        f'An edge margin of {args.edge_margin_m:g} m rejects reconstructed landings near boundaries because the source clips bounces to x=±450/y=±250. This prevents treating clipped out-of-bounds candidates as proven successes, but also removes real edge-near shots. Contact positions are anchored to inferred racket-hand positions; bounces are anchored to the table plane; intervening trajectories are synthetic parabolic interpolation. Therefore geometric consistency checks are not independent ground truth. Missing ball xyz are NaN, frequently near the start/end and sometimes internally. No new missing contacts or trajectories are imputed.',
        '## Human landing categories: explicitly provisional',
        f'V3 profile `{args.robot_profile}` supplies five commanded target points through skill_target_xy, not five validated outcome regions or learned landing labels. Robot table_top in mujoco_env_comp.py has half-length 1.37 m and half-width 0.7625 m; net x=TABLE_SHIFT. Convert each target to ((x−TABLE_SHIFT)/1.37, y/0.7625). Convert human landing to (hitter-oriented x/human half-length, hitter-oriented y/human half-width). Assign the nearest prototype in Euclidean normalized coordinates, with first SKILL_NAMES entry winning exact ties. The resulting Voronoi cells are a proposed comparison convention, not official LATTE-MV or robot-controller decision boundaries. Short prototypes are toward the net relative to the corresponding long prototypes; the center-safe prototype competes over both coordinates.',
        'Every retained candidate has observed normalized landing coordinates and every assigned label is human_skill_proxy with classification_status=provisional_nearest_v3_target. Landing_proxy_regions.png displays exact proposed boundaries, target prototypes, accepted points and rejected candidate landings. Those rejected before a landing could be associated have no fabricated landing coordinates. Validation of this proxy requires reviewing actual robotic landing distributions under each commanded skill and human annotations; commanded actions and observed outcomes are different measurements.',
        '## Robotic format audit and reuse',
        '**Timing mismatch:** robot receiver_from_crossing makes the receiver the attacker and chooses its next commanded shot at the incoming crossing. This human pilot follows the requested outgoing crossing after the hitter’s contact. These are different temporal conditions, even when both are called net-crossing positions. Align human positions to the preceding incoming crossing or compare at pre-contact before claiming decision-rule equivalence. --position-time pre_contact is available now as a first sensitivity check.',
        'v3/analyze_conditional_skill_selection.py reads decisions.csv with attacker_policy, role IDs, attack_skill, corrected 74D state_json and optional logged policy vectors and defender snapshot. Canonical attacker xy are state[0:2], defender xy state[18:20]; undo the P2-attacker reflection to get physical coordinates. P1/P2 positions are pre-commit actual gantries. Five skill names/order come from SKILL_NAMES; v3.skill_semantics uses acting-player-relative signs and skill-profile depth targets. Counts condition only on rows where the named strategy controls the attacker.',
        'Human summaries reuse the robot cell_rows and plot_grid directly without changing robot code. joint_position_skill_summary.csv has analysis_run,strategy,player,p1_position,p2_position,attack_skill,n_decisions,n_policy_vectors,selected_count,empirical_probability,mean_policy_probability. attack_skill is a compatibility key for the explicitly named human landing proxy; original human event data use human_skill_proxy. There is no known human policy vector: n_policy_vectors=0 and mean_policy_probability is blank. The separate empirical five-vector is derived from category counts; it is not a logged policy distribution. Empty cells have n=0, zero counts and undefined probabilities. n is successful classified shot count, not frames or rallies; vectors in the reused figure titles mean logged policy vectors, so are always zero for humans.',
        'opponent_position_skill_summary.csv preserves the same fields with opponent_position instead of the two-player pair. position_combination_frequency.csv reports the nine visited-state frequencies within each player’s accepted shots, explicitly conditional on successful usable reconstruction. Summary JSON includes empirical five-vectors (null for empty cells), diagnostics, configuration, source pointers and video checks. Human shot/all-shot event CSVs additionally preserve contact/crossing/bounce timing, both pose timings, raw normalized landings, velocity checks, source ID and exclusion reason.',
        '## Selected rallies', '| Rally | Match | Shape | Finite ball frames | Loaded |', '| --- | --- | --- | ---: | --- |']
    investigation += [f"| {i['rally_id']} | {i['match_id']} | {i.get('shape')} | {i.get('finite_ball_frames')} | {i.get('loaded')} |" for i in inventory]
    table_start=next(i for i,text in enumerate(investigation) if text.startswith('| Rally |'))
    investigation=investigation[:table_start]+['\n'.join(investigation[table_start:])]
    (out/'dataset_investigation.md').write_text('\n\n'.join(investigation)+'\n')
    selected=summary['selected_players']
    report=['# Human conditional shot-selection pilot',
        '**Pilot only. All five-way human categories are provisional landing proxies. These are reconstructed successful-return outcomes, whereas the robot plots show selected commands. No claim of statistical similarity is supported.**',
        '## Execution and diagnostics','```json\n'+json.dumps({k:summary[k] for k in ['total_sampled_rallies','successfully_loaded_rallies','all_players','selected_players']},indent=2)+'\n```',
        'Default primary analysis is P1 only. Both-player contact diagnostics are retained, and --player p2/both uses the same orientation normalization. Excluded observations stay in shot_events.csv/all_shot_events.csv, rather than being counted as successful categories. Counts are cumulative through validation gates; reasons describe the first failed gate.',
        '## Validation evidence',
        'Top-down shot figures show recorded players/ball positions and ball height versus event timing for up to three accepted examples. Position_zones.png and landing_proxy_regions.png show both coordinate conventions and exact proposed category boundaries. Original-video contact/crossing/landing frame sheets are generated for up to three selected clips, without assuming 2D tracker coordinates are native-video pixels. Video frame count/FPS and inspected indices are recorded in summary.json; these sheets are visual validation aids, not an independent human-labeled contact accuracy score.',
        '## Output compatibility and side-by-side comparison',
        '**Temporal alignment remains unresolved:** the human outgoing-crossing condition follows the hit; robotic incoming-crossing conditions precede the chosen hit. Matching CSV schemas and coordinate signs does not resolve this temporal difference.',
        json.dumps(summary['robot_comparison_sources'],indent=2),
        'The comparison uses available P1 Nash-Q and Potential IBR CSVs, choosing the most recently modified compatible report for each method (paths/run IDs recorded). No robot evaluation is launched. The same 9 physical-player own-side conditions, five ordering keys, count normalization and colors are used. Human n_policy_vectors=0 is intentional. State occupancy and sample sizes are different; successful human landings are selected while robot commands may include failed outcomes. Do not interpret the plotted differences as behavioral or statistical similarity/dissimilarity.',
        '## Next-step recommendation',
        f"Do not scale to the full archive yet. The selected pilot retains {selected['classified_shots']} classified shots from {selected['detected_racket_contacts']} selected-player contacts. Resolve (1) independent manual video checks of contact/bounce association and left/right for more camera views, (2) clipped-edge and invalid-net trajectory handling, (3) validation of provisional regions against actual robot landing distributions, (4) pre-contact versus crossing and pelvis versus foot-midpoint sensitivity, and (5) sparse-cell uncertainty and match-level clustering. Reconstruction checks are partly circular because contacts/bounces anchor the generated trajectories. A balanced multi-match sample and fixed manifest establish reproducibility, not population representativeness. Keep landing coordinates even if proxy regions change.",
        '## Reproduction',
        '```bash\npython3 v3/human_comparison/analyze_human.py --data-root '+str(args.data_root)+' --sample-rallies '+str(args.sample_rallies)+' --seed '+str(args.seed)+' --player '+args.player+' --position-time '+args.position_time+' --output-dir '+str(out)+'\n```',
        'See dataset_investigation.md, sample_manifest.json and summary.json for exact input paths, hashes, available events and all numerical settings.',
        'For the seed-42 pilot, [manual review notes](manual_review.md) record the sparse-frame inspection and its limits. Those notes identify the reviewed rally IDs and do not establish ground-truth event accuracy for other samples.',
        '## Figures']
    report += [f'![{p.stem}]({p.name})' for p in sorted(out.glob('*.png'))]
    (out/'pilot_report.md').write_text('\n\n'.join(report)+'\n')


if __name__=='__main__':main()
