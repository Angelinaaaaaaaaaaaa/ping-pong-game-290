"""Pilot-only diagnostic figures; the conditional grid uses the robot plotter."""
import io
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from v3.human_comparison.events import HALF_LENGTH, HALF_WIDTH, TABLE_Z, NET_TOP, target_prototypes
from nash_skills.skills import SKILL_NAMES
from v3.constants import DEFENSE_POSITIONS


def save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches='tight'); plt.close(fig)


def draw_table(ax):
    ax.plot([-HALF_LENGTH, HALF_LENGTH, HALF_LENGTH, -HALF_LENGTH, -HALF_LENGTH],
            [-HALF_WIDTH, -HALF_WIDTH, HALF_WIDTH, HALF_WIDTH, -HALF_WIDTH], color='black')
    ax.plot([0,0],[-HALF_WIDTH,HALF_WIDTH],color='gray',ls='--',label='Net')
    ax.set(xlabel='Table longitudinal x (m)',ylabel='World lateral y (m)'); ax.set_aspect('equal')


def shot_plot(ball, positions, row, out, net_top=NET_TOP):
    c, b = int(row['contact_frame']), int(row['bounce_frame'])
    t = float(row['crossing_frame']); first = int(t); fraction = t-first
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    draw_table(axes[0]); segment=ball[c:b+1]
    axes[0].plot(segment[:,0],segment[:,1],'-o',ms=2,color='green',label='Reconstructed ball')
    axes[0].scatter(ball[[c,b],0],ball[[c,b],1],c=['red','black'],marker='x',s=60)
    for p in (1,2):
        pos=positions[p][first]+fraction*(positions[p][first+1]-positions[p][first])
        axes[0].scatter(pos[0],pos[1],s=80,label=f'P{p} at crossing')
        axes[0].annotate(f'P{p}',pos[:2],xytext=(5,5),textcoords='offset points')
    axes[0].legend(fontsize=7)
    ts=np.arange(max(0,c-2),min(len(ball),b+3));axes[1].plot(ts,ball[ts,2],'.-')
    for frame,label in [(c,'hit'),(t,'net crossing'),(b,'bounce')]:axes[1].axvline(frame,ls='--',label=label)
    axes[1].axhline(TABLE_Z,color='gray',label='Table surface');axes[1].axhline(net_top,color='black',ls=':',label='Configured nominal net top')
    axes[1].set(xlabel='Frame',ylabel='Ball z (m)');axes[1].legend(fontsize=7)
    fig.suptitle(f"{row['rally_id']} shot {row['shot_index']}: P{row['attacker_id']}, provisional {row['human_skill_proxy']}")
    save(fig,out/f"shot_{row['rally_id']}_{row['shot_index']}.png")


def spatial_plots(events, config, out):
    valid=events[events.classified]
    fig,ax=plt.subplots(figsize=(9,5));draw_table(ax)
    boundary=config.center_normalized*HALF_WIDTH
    for y in (-boundary,boundary):ax.axhline(y,color='gray',ls=':')
    for p in (1,2):
        if f'p{p}_crossing_x_m' in valid:
            ax.scatter(valid[f'p{p}_crossing_x_m'],valid[f'p{p}_crossing_y_m'],label=f'P{p} {config.landmark}',s=25)
        x=(-1 if p==1 else 1)*1.8
        for y,label in [(HALF_WIDTH*.65,'left' if p==1 else 'right'),(0,'center'),(-HALF_WIDTH*.65,'right' if p==1 else 'left')]:ax.text(x,y,f'P{p} {label}',ha='center',fontsize=8)
    ax.set_title(f'Player-relative positioning zones: |own y / half-width| ≤ {config.center_normalized:.4f} is center')
    ax.legend(fontsize=8);save(fig,out/'position_zones.png')
    depth,lateral=np.meshgrid(np.linspace(0,1,250),np.linspace(-1,1,300))
    prototypes=target_prototypes(config.robot_profile)
    assignment=np.argmin(np.sum((np.stack([depth,lateral],axis=-1)[...,None,:]-prototypes)**2,axis=-1),axis=-1)
    colors=plt.get_cmap('tab10')(np.arange(5));fig,ax=plt.subplots(figsize=(8,6))
    ax.pcolormesh(depth,lateral,assignment,cmap=ListedColormap(colors),vmin=0,vmax=4,alpha=.25,shading='auto')
    for i,(d,l) in enumerate(prototypes):ax.scatter(d,l,color=colors[i],marker='s');ax.annotate(SKILL_NAMES[i],(d,l),fontsize=8)
    if 'landing_depth_normalized' in events:
        for flag,marker,color,label in [(True,'o','black','Accepted proxy shots'),(False,'x','red','Rejected candidate landings')]:
            sub=events[events.classified==flag];ax.scatter(sub.landing_depth_normalized,sub.landing_lateral_normalized,marker=marker,color=color,label=label,s=25)
    ax.set(xlim=(0,1),ylim=(-1,1),xlabel='Opponent-half depth (0=net, 1=endline)',ylabel='Hitter-relative lateral y (positive=left)',title='PROVISIONAL nearest-V3-target regions; observed human landings')
    ax.legend(fontsize=8);save(fig,out/'landing_proxy_regions.png')


def video_sheet(store, member, rows, out):
    """Read one selected clip into a temporary file, inspect native frames only."""
    import cv2
    video=member.replace('_recon.npy','.mp4')
    if video not in store.name_set:return {'video':video,'status':'unavailable'}
    record=rows[0]; frames=[max(0,int(record['contact_frame'])-1),int(np.ceil(record['crossing_frame'])),int(record['bounce_frame'])]
    with tempfile.TemporaryDirectory(prefix='latte_pilot_video_') as temp:
        path=Path(temp)/'clip.mp4';path.write_bytes(store.read(video));cap=cv2.VideoCapture(str(path))
        result={'video':video,'video_frames':int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 'video_fps':float(cap.get(cv2.CAP_PROP_FPS)), 'inspected_frames':frames}
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for ax,f,label in zip(axes,frames,['Before hit','Net crossing','Landing']):
            cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,image=cap.read()
            if ok:ax.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            else:ax.text(.5,.5,'Frame unavailable',ha='center')
            ax.set_title(f'{label}: frame {f}');ax.axis('off')
        cap.release();fig.suptitle(record['rally_id']+' original video (annotation-selected frames; manual validation aid)');save(fig,out/f"video_{record['rally_id']}.png")
        return result


def comparison_plot(human, sources, out):
    player='P1' if (human.player=='P1').any() else 'P2'
    groups=[(f'Human {player}: PROVISIONAL landing proxy',human[human.player==player])]
    used=[]
    for policy in ('nash_q','potential_ibr'):
        candidates=[]
        for path in sources:
            frame=pd.read_csv(path);sub=frame[(frame.strategy==policy)&(frame.player==player)]
            for run,g in sub.groupby('analysis_run'):
                if len(g)==45:candidates.append((path.stat().st_mtime,g,path,run))
        if candidates:
            _,g,path,run=max(candidates,key=lambda item:item[0]);groups.append((policy+' '+player+': commanded skill',g));used.append({'policy':policy,'player':player,'file':str(path),'analysis_run':run})
    fig,axes=plt.subplots(len(groups),9,figsize=(23,3.2*len(groups)),squeeze=False)
    for i,(title,frame) in enumerate(groups):
        for j,(p1,p2) in enumerate(__import__('itertools').product(DEFENSE_POSITIONS,repeat=2)):
            ax=axes[i,j];cell=frame[(frame.p1_position==p1)&(frame.p2_position==p2)].set_index('attack_skill').reindex(SKILL_NAMES)
            n=int(cell.n_decisions.iloc[0]) if len(cell) and cell.n_decisions.notna().any() else 0
            if n:ax.bar(range(5),cell.empirical_probability,color=plt.get_cmap('tab10')(np.arange(5)))
            else:ax.text(.5,.5,'Undefined',transform=ax.transAxes,ha='center',fontsize=7)
            ax.set_ylim(0,1);ax.set_title(f'{p1}/{p2}\nn={n}',fontsize=8);ax.set_xticks(range(5),['L','LS','C','RS','R'],fontsize=7)
            if j==0:ax.set_ylabel(title,fontsize=8)
    fig.suptitle('Output compatibility only: physical P1/P2 own-side bins; human outcomes versus robot commands')
    save(fig,out/'human_robot_side_by_side.png');return used
