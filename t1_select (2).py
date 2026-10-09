#!/usr/bin/env python3
"""T1 OMOMO selection for Rudra. Python 3.11+, numpy, scipy, torch, matplotlib.

Commands (run from ~/holosoma in hsretargeting):
  python t1/t1_select.py selftest
  python t1/t1_select.py scan
  python t1/t1_select.py review --task lift
  python t1/t1_select.py review --task carry
  python t1/t1_select.py review --task push
  python t1/t1_select.py review --task pull
  python t1/t1_select.py status
  python t1/t1_select.py finalize

Review: Space pause; left/right scrub; R replay; A accept; X reject;
N skip; C confirm coordinate/geometry conventions for this object; Q quit.
Acceptance requires one complete playback with every frame displayed. C is a
human assertion: confirm mesh fits the recorded hands, floor is z=0, skeleton
is upright, mesh scale is plausible in metres, and orange arrow faces forward.
If these are wrong, reject/stop and investigate; do not press C to bypass it.
A also asserts this is one clean action of the displayed class, with appropriate
hand interaction, not a throw, kick, incidental drift, or unrelated movement.

Data mapping is from the user's installed Holosoma load_intermimic_data:
 human joints 162:318 -> (T,52,3); object XYZ 318:321; XYZW 321:325.
Z-up, metre units and floor z=0 are hypotheses checked numerically and manually.
FPS is NOT known from the tensor. --playback-fps defaults to 30 for viewing only;
all segmentation durations below are FRAME COUNTS, never inferred seconds.

Thresholds are conservative selection heuristics, not supplied ground truth:
 floor +/-0.04 m, airborne >=0.10 m for >=12 consecutive frames;
 lift rise-and-hold excursion <=0.35 m; airborne carry excursion >1.0 m.
 Carries use horizontal excursion during the off-floor episode, not a sum of
 noisy frame displacements. Lift must start grounded, rise, and remain raised.
 Ground movement >=0.30 m, >=12 frames, >=80% direction consistency,
 cosine with subject facing >=0.5, object predominantly in front of subject.
 Push follows forward facing; pull opposes facing. Person/object separation
 alone is not used: walking together can keep that separation constant.
 Hand-to-object convex-hull-vertex proximity <=0.65 m is a loose plausibility
 filter, not proof of physical contact. Manual review checks actual interaction.

Mixed recordings: lift prefixes are selected independently of later carrying;
 off-floor episodes are split at sustained ground contact;
 ground motions split on pauses, floor departure or direction reversal.
 At most one candidate per source file per class; duplicate tensor contents are
 excluded. No overlapping sliding windows are manufactured to satisfy counts.
 Crops retain ALL 591 columns and use [start, end) source-frame indexing.
 Raw data and Holosoma code are never modified. A fresh output directory is
 required for each scan; reviews are bound to source and geometry SHA256 hashes.

Finalization: >=15 non-rejected candidates/class, >=10 accepted complete
 reviews/class, conventions confirmed for every selected object, no input read
 errors. Four lists include all eligible candidates, with review state in JSON.
 Unreviewed candidates are explicitly listed as such, as T1 requires 10 reviews
 per class, not every clip. --require-all-reviewed provides a stricter gate.
 Finalization exports source crops to final/clips for later T2 retargeting.
 final/manifest.json retains source IDs, hashes, frame ranges and thresholds.
 T2 still must replay EVERY retargeted motion and reject physical artifacts.

Mesh provenance: downloaded by user from
 https://github.com/Sirui-Xu/InterMimic
 isaacgym/src/intermimic/data/assets/objects/objects/OBJECT/OBJECT.obj
 Meshes are read directly. The downloaded URDF relative mesh paths need not
 resolve in the flattened t1/assets layout and are not used by this script.

If counts are short, inspect audit.csv and review candidates; do not loosen
 thresholds simply to claim completion. Re-scan with a new --out after justified
 changes. Floor or coordinate errors require correction, not relaxed thresholds.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from scipy.ndimage import median_filter

VERSION = '1.1.0'
TASKS = ('lift', 'carry', 'push', 'pull')
BOXES = ('largebox', 'smallbox', 'plasticbox')
GROUND_OBJECTS = ('smalltable', 'woodchair', 'whitechair')
OBJECTS = BOXES + GROUND_OBJECTS
DEFAULTS = dict(floor_tolerance=0.04, airborne_height=0.10,
                minimum_air_frames=12, ground_reset_frames=6,
                lift_hold_height_range=0.04,
                minimum_ground_frames=12, lift_max_travel=0.35,
                carry_min_travel=1.0, ground_min_travel=0.30,
                direction_cosine=0.5, direction_consistency=0.80,
                motion_step=0.002, hand_distance=0.65)
# Zero-based indices from installed SMPLH_DEMO_JOINTS.
BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
         (0,9),(9,10),(10,11),(11,12),(12,13),
         (11,14),(14,15),(15,16),(16,17),
         (11,33),(33,34),(34,35),(35,36)]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def load_tensor(path):
    import torch
    data = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(data, torch.Tensor) or data.ndim != 2 or data.shape[1] != 591:
        raise ValueError('Expected a (frames, 591) tensor')
    if len(data) < 3 or not bool(torch.isfinite(data).all()):
        raise ValueError('Too few frames or non-finite tensor values')
    return data.detach()


def read_obj(path):
    vertices, triangles = [], []
    for line in Path(path).read_text().splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == 'v':
            vertices.append([float(x) for x in tokens[1:4]])
        elif tokens[0] == 'f':
            indices = [int(x.split('/')[0]) for x in tokens[1:]]
            indices = [i-1 if i > 0 else len(vertices)+i for i in indices]
            for j in range(1, len(indices)-1):
                triangles.append([indices[0], indices[j], indices[j+1]])
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(triangles, dtype=int)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) < 4 or not np.isfinite(v).all():
        raise ValueError(f'Invalid mesh: {path}')
    if f.ndim != 2 or f.shape[1] != 3 or f.min() < 0 or f.max() >= len(v):
        raise ValueError(f'Invalid OBJ faces: {path}')
    return v, f


def mesh_vertices(path):
    # Only OBJ vertex positions are needed; the hull preserves exact extrema
    # under arbitrary rigid rotations, unlike a bounding box approximation.
    v, _ = read_obj(path)
    v = np.unique(v, axis=0)
    return v[ConvexHull(v).vertices]


def geometry(data, vertices):
    human = np.asarray(data[:,162:318], float).reshape(-1,52,3)
    pos = np.asarray(data[:,318:321], float)
    quat = np.asarray(data[:,321:325], float)
    norms = np.linalg.norm(quat, axis=1)
    if np.any(np.abs(norms-1) > 0.05):
        raise ValueError('Object quaternion norm differs from 1 by >0.05')
    matrices = Rotation.from_quat(quat).as_matrix()
    bottom = np.empty(len(pos))
    hand = np.empty(len(pos))
    for s in range(0,len(pos),64):
        e = min(s+64,len(pos))
        world = np.einsum('tij,vj->tvi', matrices[s:e], vertices) + pos[s:e,None,:]
        bottom[s:e] = world[:,:,2].min(axis=1)
        delta = world[:,:,None,:] - human[s:e,None,[17,36],:]
        hand[s:e] = np.linalg.norm(delta,axis=-1).min(axis=(1,2))
    # Anatomical left axis crossed with world up gives forward in a
    # right-handed Z-up system. Must be confirmed visually, never auto-flipped.
    left = (human[:,15]-human[:,34]) + (human[:,1]-human[:,5])
    facing = np.cross(left, np.array([0.,0.,1.]))[:,:2]
    lengths = np.linalg.norm(facing,axis=1)
    valid_facing = lengths > 0.10
    facing = facing / np.maximum(lengths[:,None],1e-9)
    head_above = human[:,13,2] - human[:,0,2]
    foot_z = human[:,[4,8],2].min(axis=1)
    valid = ((head_above > 0.15) & (head_above < 1.1)
             & (foot_z > -0.20) & (foot_z < 0.45))
    return dict(human=human,pos=pos,quat=quat,matrices=matrices,
                bottom=bottom,hand=hand,facing=facing,
                valid_facing=valid_facing,valid=valid)


def excursion(xy):
    # Maximum distance from episode's initial horizontal position.
    return float(np.linalg.norm(xy-xy[0],axis=1).max())


def candidates(g, name, cfg, diagnostics=None):
    h,p,b = g['human'],g['pos'],g['bottom']
    n=len(p); found=[]
    diag = diagnostics if diagnostics is not None else {}
    def count(reason):
        diag[reason] = diag.get(reason, 0) + 1
    grounded = np.abs(b) <= cfg['floor_tolerance']
    smoothed = median_filter(p[:,:2],size=(5,1),mode='nearest')
    def add(task,s,e,travel,quality):
        if e-s < 3: return
        if np.mean(g['valid'][s:e]) < 0.95:
            count(task+'_invalid_skeleton'); return
        if np.min(b[s:e]) < -cfg['floor_tolerance']:
            count(task+'_floor_penetration'); return
        if np.mean(g['hand'][s:e] <= cfg['hand_distance']) < 0.5:
            count(task+'_hand_proximity'); return
        found.append(dict(task=task,start=int(s),end=int(e),
                          travel_m=float(travel),bottom_min_m=float(b[s:e].min()),
                          bottom_max_m=float(b[s:e].max()),score=float(quality)))
    if name in BOXES:
        # Stable ground intervals delimit complete airborne episodes.
        stable = [(s,e) for s,e in runs(grounded)
                  if e-s >= cfg['ground_reset_frames']]
        if not stable: count('no_stable_ground_interval')
        for i,(gs,ge) in enumerate(stable):
            stop = stable[i+1][0] if i+1 < len(stable) else n
            if stop-ge < cfg['minimum_air_frames']:
                count('departure_too_short'); continue
            high = [(s+ge,e+ge) for s,e in runs(b[ge:stop] >= cfg['airborne_height'])
                    if e-s >= cfg['minimum_air_frames']]
            if not high:
                count('no_sustained_airborne_interval'); continue
            s=max(gs,ge-cfg['ground_reset_frames'])
            # The lift ends after the FIRST sustained, height-stable raised
            # interval. Later walking or lowering cannot invalidate that crop.
            # Do not search later intervals after a failed prefix: this would
            # hide intervening motion or manufacture a lift after a carry.
            hs,he = high[0]
            hold = cfg['minimum_air_frames']
            lift_end = None
            for end in range(hs+hold, he+1):
                if np.ptp(b[end-hold:end]) <= cfg['lift_hold_height_range']:
                    lift_end = end
                    break
            if lift_end is None:
                count('lift_no_stable_raised_hold')
            else:
                lift_travel = excursion(smoothed[s:lift_end])
                if lift_travel <= cfg['lift_max_travel']:
                    add('lift',s,lift_end,lift_travel,hold/(1+lift_travel))
                else:
                    count('lift_prefix_horizontal_travel')
            # Carry remains a separate candidate from the same source.
            # Its >1m threshold must be met while continuously clearly raised.
            high_travel = max(excursion(smoothed[a:z]) for a,z in high)
            if high_travel > cfg['carry_min_travel']:
                e=min(n,stop+cfg['ground_reset_frames'])
                add('carry',s,e,high_travel,high_travel)
            else:
                count('carry_below_travel_threshold')
    else:
        step=np.diff(smoothed,axis=0)
        length=np.linalg.norm(step,axis=1)
        forward=g['facing'][:-1]
        cosine=np.sum(step*forward,axis=1)/np.maximum(length,1e-9)
        front=np.sum((p[:,:2]-h[:,0,:2])*g['facing'],axis=1) > 0.05
        good=(grounded[:-1]&grounded[1:]&g['valid_facing'][:-1]&front[:-1]
              &(length>=cfg['motion_step']))
        for task,sign in [('push',1),('pull',-1)]:
            mask=good & (sign*cosine >= cfg['direction_cosine'])
            # Bridge <=3 noisy/paused intervals only if grounded and no
            # substantial reverse motion. Never bridge an airborne interval.
            for s,e in runs(~mask):
                if s>0 and e<len(mask) and e-s<=3:
                    safe=(grounded[s:e+1].all() and front[s:e].all()
                          and g['valid_facing'][s:e].all()
                          and np.all(sign*cosine[s:e] >= -0.25))
                    if safe: mask[s:e]=True
            for s,e in runs(mask):
                end=e+1
                if end-s < cfg['minimum_ground_frames']: continue
                travel=float(np.linalg.norm(smoothed[end-1]-smoothed[s]))
                weights=length[s:e]
                consistency=float(weights[sign*cosine[s:e]>=cfg['direction_cosine']].sum()
                                  /max(weights.sum(),1e-9))
                if travel>=cfg['ground_min_travel'] and consistency>=cfg['direction_consistency']:
                    add(task,s,end,travel,travel*consistency)
    # Prevent inflation by returning only the best episode/class/source.
    return [max([c for c in found if c['task']==t],key=lambda c:c['score'])
            for t in TASKS if any(c['task']==t for c in found)]


def scan(args):
    out=args.out
    if out.exists(): raise ValueError(f'{out} exists. Use a fresh --out to protect reviews.')
    cfg=DEFAULTS.copy()
    if args.config:
        overrides=json.loads(args.config.read_text())
        if set(overrides)-set(cfg): raise ValueError('Unknown threshold keys')
        cfg.update(overrides)
    if any(not isinstance(v,(float,int)) or not math.isfinite(v) or v<=0 for v in cfg.values()):
        raise ValueError('Thresholds must be positive finite numbers')
    if cfg['direction_cosine'] > 1 or cfg['direction_consistency'] > 1:
        raise ValueError('Direction cosine and consistency must be <=1')
    if not cfg['floor_tolerance'] < cfg['airborne_height']:
        raise ValueError('Floor tolerance must be below airborne threshold')
    if not cfg['lift_max_travel'] < cfg['carry_min_travel'] or cfg['carry_min_travel'] < 1:
        raise ValueError('Carry threshold must be >=1m and above lift threshold')
    for key in ('minimum_air_frames','ground_reset_frames','minimum_ground_frames'):
        if int(cfg[key]) != cfg[key]: raise ValueError(f'{key} must be integer')
        cfg[key]=int(cfg[key])
    meshes={}; mesh_info={}
    for name in OBJECTS:
        path=args.assets/name/(name+'.obj')
        meshes[name]=mesh_vertices(path)
        mesh_info[name]=dict(path=str(path.resolve()),sha256=digest(path))
    files=sorted(args.data.rglob('*.pt'))
    if not files: raise ValueError(f'No .pt files under {args.data}')
    diagnostic_totals={}
    records=[]; audit=[]; hashes=set(); stems=set(); inventory={o:0 for o in OBJECTS}
    for i,path in enumerate(files):
        matches=[o for o in OBJECTS if o in path.stem.lower().split('_')]
        if len(matches)!=1: continue
        name=matches[0]; inventory[name]+=1
        row=dict(source=str(path.resolve()),object=name,status='',detail='')
        try:
            if path.stem in stems: raise ValueError('Duplicate source stem; IDs would be ambiguous')
            stems.add(path.stem)
            raw=load_tensor(path); array=raw.numpy()
            content=hashlib.sha256(array.tobytes()).hexdigest()
            if content in hashes:
                row.update(status='duplicate',detail='Identical tensor already processed')
                audit.append(row); continue
            hashes.add(content)
            g=geometry(array,meshes[name]); diagnostics={}
            cs=candidates(g,name,cfg,diagnostics)
            for reason, count in diagnostics.items():
                diagnostic_totals[reason]=diagnostic_totals.get(reason,0)+count
            source_hash=digest(path)
            for c in cs:
                c.update(id=f'{path.stem}_t1_{c["task"]}_{c["start"]:06d}_{c["end"]:06d}',
                         source=str(path.resolve()),source_id=path.stem,
                         sha256=source_hash,object=name,review='pending')
                records.append(c)
            row.update(status='candidate' if cs else 'unclassified',
                       detail=json.dumps(dict(frames=len(array),
                           bottom_percentiles=np.percentile(g['bottom'],[0,5,50,95,100]).tolist(),
                           upright_fraction=float(g['valid'].mean()),
                           tasks=[c['task'] for c in cs], diagnostics=diagnostics)))
        except Exception as exc:
            row.update(status='error',detail=f'{type(exc).__name__}: {exc}')
        audit.append(row)
        if sum(inventory.values())%100==0:
            print(f'Processed {sum(inventory.values())} target recordings; {len(records)} candidates',flush=True)
    out.mkdir(parents=True)
    manifest=dict(version=VERSION,script_sha256=digest(__file__),created=datetime.now(timezone.utc).isoformat(),
                  data=str(args.data.resolve()),thresholds=cfg,meshes=mesh_info,
                  conventions=dict(up='z',floor=0,units='metres',quaternion='xyzw',
                                   fps=None,facing='anatomical_left cross world_up'),
                  conventions_confirmed={},inventory=inventory,
                  errors=[r for r in audit if r['status']=='error'],
                  diagnostic_totals=diagnostic_totals,clips=records)
    save_json(out/'manifest.json',manifest)
    with (out/'audit.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['source','object','status','detail'])
        writer.writeheader();writer.writerows(audit)
    for task in TASKS:
        ids=[r['id'] for r in records if r['task']==task]
        (out/f'{task}_candidates.txt').write_text(''.join(x+'\n' for x in ids))
    print('Diagnostic counts (episodes, not distinct recordings):')
    for reason,count in sorted(diagnostic_totals.items()):
        print(f'  {reason}: {count}')
    print('Scan saved:',out)
    status(args)


def read_manifest(args):
    return json.loads((args.out/'manifest.json').read_text())


def verify_inputs(m, clip=None):
    names={clip['object']} if clip else {c['object'] for c in m['clips'] if c['review']!='rejected'}
    for name in names:
        item=m['meshes'][name]
        if digest(item['path'])!=item['sha256']: raise ValueError(f'Mesh changed: {name}')
    clips=[clip] if clip else [c for c in m['clips'] if c['review']!='rejected']
    seen=set()
    for c in clips:
        if c['source'] in seen: continue
        seen.add(c['source'])
        if digest(c['source'])!=c['sha256']: raise ValueError(f'Source changed: {c["source"]}')


def status(args):
    m=read_manifest(args)
    for t in TASKS:
        rows=[c for c in m['clips'] if c['task']==t]
        counts={s:sum(c['review']==s for c in rows) for s in ('pending','accepted','rejected')}
        print(f'{t:5}: {counts}; eligible={counts["pending"]+counts["accepted"]}')
    print('Object conventions confirmed:',', '.join(sorted(m['conventions_confirmed'])) or 'none')
    print('Input errors:',len(m['errors']))
    print('Candidate lists are provisional. Only finalize creates final lists.')


def review(args):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    if plt.get_backend().lower() in {'agg','pdf','svg','ps','cairo','template'}:
        raise ValueError('Review needs an interactive Matplotlib backend. Run from a desktop terminal; try MPLBACKEND=TkAgg.')
    m=read_manifest(args)
    rows=[c for c in m['clips'] if c['task']==args.task and
          (args.revisit or c['review']=='pending')]
    # Cycle through objects to encourage diverse reviews rather than one mesh.
    queues={o:sorted([c for c in rows if c['object']==o],key=lambda c:-c['score']) for o in OBJECTS}
    rows=[]
    while any(queues.values()):
        for q in queues.values():
            if q: rows.append(q.pop(0))
    for c in rows:
        verify_inputs(m,c)
        v=mesh_vertices(m['meshes'][c['object']]['path'])
        display_vertices, faces = read_obj(m['meshes'][c['object']]['path'])
        raw=load_tensor(c['source']).numpy()[c['start']:c['end']]
        g=geometry(raw,v); n=len(raw)
        fig=plt.figure(figsize=(12,7)); ax=fig.add_subplot(121,projection='3d')
        chart=fig.add_subplot(222); chart.plot(g['bottom'],label='object bottom')
        chart.axhline(0,color='black');chart.axhline(0.04,color='gray',ls=':')
        chart.axhline(0.10,color='green',ls=':');chart.set_ylabel('Height (m)');chart.legend()
        marker=chart.axvline(0,color='red')
        top=fig.add_subplot(224);top.plot(g['pos'][:,0],g['pos'][:,1],label='object')
        top.plot(g['human'][:,0,0],g['human'][:,0,1],label='pelvis')
        top.axis('equal');top.legend();top.set_xlabel('World X (m)');top.set_ylabel('World Y (m)')
        allpoints=np.concatenate([g['human'].reshape(-1,3),g['pos']],axis=0)
        low=allpoints.min(axis=0)-0.6;high=allpoints.max(axis=0)+0.6
        low[2]=min(-0.1,low[2]); center=(low+high)/2;span=max(high-low)
        ax.set_xlim(center[0]-span/2,center[0]+span/2)
        ax.set_ylim(center[1]-span/2,center[1]+span/2)
        ax.set_zlim(low[2],low[2]+span);ax.set_box_aspect((1,1,1))
        ax.set_xlabel('X');ax.set_ylabel('Y');ax.set_zlabel('Z')
        floor=np.array([[low[0],low[1],0],[high[0],low[1],0],
                        [high[0],high[1],0],[low[0],high[1],0]])
        ax.add_collection3d(Poly3DCollection([floor],alpha=.12,color='gray'))
        body=Line3DCollection([g['human'][0,[a,b]] for a,b in BONES],colors='royalblue',linewidths=2);ax.add_collection3d(body)
        initial_world=display_vertices@g['matrices'][0].T+g['pos'][0]
        mesh=Poly3DCollection(initial_world[faces],alpha=.35,facecolor='green',edgecolor='none');ax.add_collection3d(mesh)
        facing,=ax.plot([],[],[],color='darkorange',linewidth=3,marker='>')
        title=fig.suptitle('')
        notice=fig.text(.04,.02,'Space pause | arrows scrub | R replay | C conventions | A accept | X reject | N next | Q quit',fontsize=9)
        accepted_count=sum(x['task']==args.task and x['review']=='accepted' for x in m['clips'])
        state=dict(frame=0,paused=False,seen=set(),decision=None,quit=False)
        def draw(i):
            body.set_segments([g['human'][i,[a,b]] for a,b in BONES])
            world=display_vertices@g['matrices'][i].T+g['pos'][i]
            mesh.set_verts(world[faces])
            origin=g['human'][i,0];tip=origin+np.r_[0.5*g['facing'][i],0]
            facing.set_data_3d([origin[0],tip[0]],[origin[1],tip[1]],[origin[2],tip[2]])
            marker.set_xdata([i,i]);state['seen'].add(i)
            title.set_text(f'{c["task"]}: {c["id"]}\nFrame {i+1}/{n} | travel {c["travel_m"]:.2f}m | '
                           f'watched {len(state["seen"])}/{n} | accepted {accepted_count}/10 | playback {args.playback_fps:g} FPS (setting)')
        def tick(_):
            draw(state['frame'])
            if not state['paused']: state['frame']=(state['frame']+1)%n
        def key(event):
            k=event.key
            if k==' ': state['paused']=not state['paused']
            elif k in ('left','right'):
                state['paused']=True
                state['frame']=int(np.clip(state['frame']+(1 if k=='right' else -1),0,n-1))
                draw(state['frame'])
            elif k=='r': state.update(frame=0,paused=False)
            elif k=='c':
                if len(state['seen'])<n:
                    notice.set_text('Watch every frame before confirming conventions.')
                else:
                    m['conventions_confirmed'][c['object']]=dict(clip=c['id'],time=datetime.now(timezone.utc).isoformat())
                    save_json(args.out/'manifest.json',m)
                    notice.set_text('Conventions confirmed: metres, z=0 floor, geometry fit, upright human, forward arrow.')
            elif k in ('a','x'):
                if k=='a' and (len(state['seen'])<n or c['object'] not in m['conventions_confirmed']):
                    notice.set_text('Acceptance blocked: watch all frames and confirm object conventions with C.')
                else:
                    state['decision']='accepted' if k=='a' else 'rejected';plt.close(fig)
            elif k=='n': plt.close(fig)
            elif k=='q': state['quit']=True;plt.close(fig)
            fig.canvas.draw_idle()
        fig.canvas.mpl_connect('key_press_event',key)
        # Keep a live reference. No frames are skipped to catch up to wall time.
        animation=FuncAnimation(fig,tick,interval=1000/args.playback_fps,cache_frame_data=False)
        plt.show()
        if state['decision']:
            c.update(review=state['decision'],reviewed_at=datetime.now(timezone.utc).isoformat(),
                     frames_seen=len(state['seen']),playback_fps=args.playback_fps)
            save_json(args.out/'manifest.json',m)
            print(c['id'],state['decision'],flush=True)
        if state['quit']: break
    status(args)


def finalize(args):
    m=read_manifest(args)
    selected=[c for c in m['clips'] if c['review']!='rejected']
    failures=[]
    if m['errors']: failures.append('Input read/processing errors: inspect audit.csv and fix/re-scan')
    for t in TASKS:
        rows=[c for c in selected if c['task']==t]
        accepted=[c for c in rows if c['review']=='accepted' and c.get('frames_seen',0)>=c['end']-c['start']]
        if len(rows)<15: failures.append(f'{t}: {len(rows)}/15 eligible clips')
        if len(accepted)<10: failures.append(f'{t}: {len(accepted)}/10 complete accepted reviews')
        if args.require_all_reviewed and len(accepted)!=len(rows): failures.append(f'{t}: pending reviews remain')
    missing={c['object'] for c in selected}-set(m['conventions_confirmed'])
    if missing: failures.append('Unconfirmed object conventions: '+', '.join(sorted(missing)))
    if failures: raise ValueError('T1 NOT COMPLETE:\n'+'\n'.join(failures))
    verify_inputs(m)
    final=args.out/'final'
    if final.exists(): raise ValueError('final already exists; keep it or use a new run directory')
    import torch
    # Publish final folder atomically only after all crops and lists are saved.
    with tempfile.TemporaryDirectory(prefix='.final-',dir=args.out) as temp:
        stage=Path(temp)/'final';(stage/'clips').mkdir(parents=True)
        for c in selected:
            raw=load_tensor(c['source'])
            torch.save(raw[c['start']:c['end']].clone(),stage/'clips'/(c['id']+'.pt'))
        for t in TASKS:
            (stage/f'{t}.txt').write_text(''.join(c['id']+'\n' for c in selected if c['task']==t))
        m['clips']=selected;m['t1_complete']=True
        m['completion_note']='>=15 candidates and >=10 accepted visual reviews per class; pending clips remain identified.'
        save_json(stage/'manifest.json',m)
        os.rename(stage,final)
    print('T1 COMPLETE under the recorded conventions and review decisions.')
    print('Lists, source-frame manifest, and cropped .pt files:',final)


def selftest(args):
    # No torch, graphics, network, or local dataset required.
    v=np.array([[x,y,z] for x in (-.2,.2) for y in (-.2,.2) for z in (-.2,.2)])
    def fixture(n=100):
        a=np.zeros((n,591));h=a[:,162:318].reshape(n,52,3)
        h[:,:,2]=1;h[:,13,2]=1.6;h[:,[4,8],2]=.05
        h[:,15,1]=.2;h[:,34,1]=-.2;h[:,1,1]=.1;h[:,5,1]=-.1
        a[:,318]=.4;a[:,320]=.2;a[:,324]=1
        h[:,17,:]=a[:,318:321];h[:,36,:]=a[:,318:321]
        return a
    a=fixture();g=geometry(a,v)
    assert np.allclose(g['bottom'],0)
    assert np.allclose(g['facing'],[1,0])
    a[:,321:325]=Rotation.from_euler('y',45,degrees=True).as_quat()
    assert np.allclose(geometry(a,v)['bottom'],.2-.2*np.sqrt(2))
    a=fixture();a[20:80,320]=.5
    assert [x['task'] for x in candidates(geometry(a,v),'largebox',DEFAULTS)]==['lift']
    a=fixture();a[20:80,320]=.5;a[20:80,318]=np.linspace(.4,1.7,60);a[80:,318]=1.7
    assert [x['task'] for x in candidates(geometry(a,v),'largebox',DEFAULTS)]==['lift','carry']
    a=fixture();a[:,320]=.5
    assert not candidates(geometry(a,v),'largebox',DEFAULTS), 'Initially held object must not invent a lift'
    for task,sign in [('push',1),('pull',-1)]:
        a=fixture();shift=sign*np.linspace(0,.6,len(a));a[:,318]+=shift
        h=a[:,162:318].reshape(-1,52,3);h[:,:,0]+=shift[:,None]
        assert [x['task'] for x in candidates(geometry(a,v),'woodchair',DEFAULTS)]==[task]
        a[:,320]=.4
        assert not candidates(geometry(a,v),'woodchair',DEFAULTS), 'Raised chair must not count as sliding'
    a=fixture();a[:,319]=np.linspace(0,.6,len(a))
    assert not candidates(geometry(a,v),'woodchair',DEFAULTS), 'Sideways movement is ambiguous'
    a=fixture();a[20:80,320]=.5;a[20:80,318]=np.linspace(.4,1.05,60);a[80:,318]=1.05
    cs=candidates(geometry(a,v),'largebox',DEFAULTS)
    assert [c['task'] for c in cs]==['lift']
    assert cs[0]['end'] < 80, 'Only the lift prefix, not the intermediate-distance transfer, qualifies'
    a=fixture();a[:,318]+=np.r_[np.linspace(0,.6,50),np.linspace(.6,0,50)]
    h=a[:,162:318].reshape(-1,52,3);h[:,:,0]+= (a[:,318]-.4)[:,None]
    assert {x['task'] for x in candidates(geometry(a,v),'woodchair',DEFAULTS)}=={'push','pull'}
    regression_tests(fixture, v)
    print('Self-tests passed: rotated bottom, facing, lift, carry, push, pull, initially-held, raised-chair, sideways, ambiguous-distance, reversal segmentation, lift-then-carry, lift-then-lower, moving-rise rejection, no-hold rejection.')


def regression_tests(fixture, vertices):
    # Gradual lift, hold, then walk 1.3m and lower. A source can supply two
    # distinct task crops, but a lift crop must end before the walk starts.
    a=fixture(150)
    a[20:40,320]=np.linspace(.2,.5,20)
    a[40:130,320]=.5
    a[130:,320]=np.linspace(.5,.2,20)
    a[70:130,318]=np.linspace(.4,1.7,60)
    a[130:,318]=1.7
    h=a[:,162:318].reshape(-1,52,3)
    h[:,:,0]+=(a[:,318]-.4)[:,None]
    h[:,[17,36],:]=a[:,None,318:321]
    cs=candidates(geometry(a,vertices),'largebox',DEFAULTS)
    assert {c['task'] for c in cs}=={'lift','carry'}
    lift=next(c for c in cs if c['task']=='lift')
    assert lift['end']<=70 and lift['travel_m']<=.35
    # A later horizontal repositioning during lowering must not erase a lift.
    a[70:,318]=.4
    a[130:,318]=np.linspace(.4,1.0,20)
    cs=candidates(geometry(a,vertices),'largebox',DEFAULTS)
    assert any(c['task']=='lift' and c['end']<=70 for c in cs)
    # A rise that already travels too far is not a stationary lift.
    a=fixture(100)
    a[20:70,320]=np.linspace(.2,.5,50); a[70:,320]=.5
    a[20:70,318]=np.linspace(.4,1.2,50); a[70:,318]=1.2
    h=a[:,162:318].reshape(-1,52,3);h[:,[17,36],:]=a[:,None,318:321]
    assert not any(c['task']=='lift' for c in candidates(geometry(a,vertices),'largebox',DEFAULTS))
    # Fast uninterrupted ascent has no sustained raised hold.
    a=fixture(60);a[20:,320]=.2+np.arange(40)*.02
    assert not candidates(geometry(a,vertices),'largebox',DEFAULTS)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command',choices=['scan','review','status','finalize','selftest'])
    home=Path.home()/'holosoma'
    parser.add_argument('--data',type=Path,default=home/'src/holosoma_retargeting/holosoma_retargeting/demo_data/OMOMO_new')
    parser.add_argument('--assets',type=Path,default=home/'t1/assets')
    parser.add_argument('--out',type=Path,default=home/'t1/run01')
    parser.add_argument('--config',type=Path,help='JSON overrides of documented heuristic thresholds')
    parser.add_argument('--task',choices=TASKS)
    parser.add_argument('--playback-fps',type=float,default=30,help='Display setting only; source FPS is unverified')
    parser.add_argument('--revisit',action='store_true')
    parser.add_argument('--require-all-reviewed',action='store_true')
    args=parser.parse_args()
    if args.command=='review' and not args.task: parser.error('review requires --task')
    if not math.isfinite(args.playback_fps) or args.playback_fps<=0: parser.error('playback FPS must be positive')
    try: globals()[args.command](args)
    except (ValueError,FileNotFoundError,ImportError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);return 1
    return 0

if __name__=='__main__':
    raise SystemExit(main())
