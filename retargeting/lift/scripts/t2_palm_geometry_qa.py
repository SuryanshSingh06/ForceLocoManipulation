#!/usr/bin/env python3
"""Prepare 50 Hz lift references and measure calibrated palm-to-collider gaps.

Commands: prepare, inventory, qa. Read README_CONTACT_QA.md first.
No simulation, robot policy, source review, collider, or input motion is modified.
Geometry passes are screening results, not proof of force-feasible lifting.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import pickle
import re
import xml.etree.ElementTree as ET

import numpy as np


THRESHOLDS = dict(contact_gap_m=.02, allowed_penetration_m=.01,
    min_contact_fraction=.9, max_p95_gap_m=.03, max_p95_drift_m=.03,
    reject_gap_m=.05, reject_far_fraction=.1, reject_penetration_m=.02,
    max_gap_jump_m=.03, reject_gap_jump_m=.05, min_window_s=.2,
    airborne_clearance_m=.05, side_tolerance_m=.005, max_wrong_side_fraction=.1)


def save(path, value):
    path=Path(path); temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ValueError(f'Cannot import {path}')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def safe_id(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+',value) or value in ('.','..'):
        raise ValueError('Unsafe clip ID.')
    return value


def asset_path(root, value):
    p=Path(value)
    return p if p.is_absolute() else root/p


def vector(value, size, label):
    if value is None: raise ValueError(f'Calibration needed: {label}')
    result=np.asarray(value,dtype=float)
    if result.shape!=(size,) or not np.isfinite(result).all():
        raise ValueError(f'{label} must contain {size} finite values.')
    return result


def unit_quat(q):
    q=np.asarray(q,dtype=float)
    if q.shape[-1:]!=(4,) or not np.isfinite(q).all(): raise ValueError('Invalid WXYZ quaternions.')
    norm=np.linalg.norm(q,axis=-1,keepdims=True)
    if np.any(np.abs(norm-1)>.02): raise ValueError('Quaternion is not unit length.')
    return q/norm


def rotate(q,v):
    q=unit_quat(q); v=np.asarray(v,float)
    t=2*np.cross(q[...,1:],v)
    return v+q[...,:1]*t+np.cross(q[...,1:],t)


def inverse_rotate(q,v):
    q=np.array(q,dtype=float,copy=True); q[...,1:]*=-1
    return rotate(q,v)


def sdf_box(points,half):
    delta=np.abs(points)-half
    return np.linalg.norm(np.maximum(delta,0),axis=-1)+np.minimum(delta.max(axis=-1),0)


def collider_primitives(config):
    if config.get('verified') is not True:
        raise ValueError('Collider geometry has not been verified in the intended simulation asset.')
    primitives=config.get('boxes')
    if not isinstance(primitives,list) or not primitives: raise ValueError('Supply measured box collider primitives.')
    result=[]
    for primitive in primitives:
        half=vector(primitive.get('half_extents_m'),3,'box half extents')
        if np.any(half<=0): raise ValueError('Collider half extents must be positive.')
        center=vector(primitive.get('center_object_m'),3,'collider center in object frame')
        quat=unit_quat(vector(primitive.get('quat_object_wxyz'),4,'collider rotation'))
        result.append((half,center,quat))
    return result


def collider_gap(points,primitives):
    # Outside a union, the minimum primitive distance is exact. Inside overlapping
    # primitives this is a penetration proxy, not exact union-boundary distance.
    return np.min([sdf_box(inverse_rotate(q,points-c),h) for h,c,q in primitives],axis=0)


def collider_bottom(object_pos,object_q,primitives):
    result=np.full(len(object_pos),np.inf)
    signs=np.asarray(list(itertools.product([-1,1],repeat=3)))
    for half,center,q in primitives:
        corners=rotate(q,signs*half)+center
        world=rotate(object_q[:,None,:],corners[None,:,:])+object_pos[:,None,:]
        result=np.minimum(result,world[:,:,2].min(axis=1))
    return result


def object_array(value,n,width,name):
    a=np.asarray(value,dtype=float)
    if a.shape==(n,1,width): a=a[:,0,:]
    if a.shape!=(n,width) or not np.isfinite(a).all(): raise ValueError(f'Invalid {name} shape/values: {a.shape}')
    return a


def read_npz(path):
    with np.load(path,allow_pickle=False) as source:
        data={k:source[k] for k in source.files}
    if 'quaternion_order' not in data or str(data['quaternion_order'].item())!='wxyz':
        raise ValueError('Expected explicit quaternion_order=wxyz; verify legacy archives before adapting.')
    positions=np.asarray(data['body_pos_w'],float)
    quats=np.asarray(data['body_quat_w'],float)
    if positions.ndim!=3 or positions.shape[2]!=3 or quats.shape!=(*positions.shape[:2],4):
        raise ValueError('Invalid body pose dimensions.')
    if not np.isfinite(positions).all(): raise ValueError('Non-finite body positions.')
    n=len(positions)
    if n<2: raise ValueError('Need at least two frames.')
    names=[x.decode() if isinstance(x,bytes) else str(x) for x in data['body_names'].tolist()]
    if len(names)!=positions.shape[1] or len(set(names))!=len(names): raise ValueError('Invalid body names.')
    fps=float(np.asarray(data['fps']).item())
    if not np.isfinite(fps) or fps<=0: raise ValueError('Invalid FPS.')
    return data,positions,unit_quat(quats),names,fps


def flag_vector(value,n,label):
    a=np.asarray(value)
    if a.shape==(n,1): a=a[:,0]
    if a.shape!=(n,) or not np.isfinite(a).all() or not np.isin(a,[0,1]).all():
        raise ValueError(f'Invalid {label}.')
    return a.astype(bool)


def evaluate(data,positions,quats,names,fps,calibration,clip):
    cfg=dict(THRESHOLDS); custom=calibration.get('thresholds',{})
    if set(custom)-set(cfg): raise ValueError('Unknown QA threshold.')
    cfg.update(custom)
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) or x<=0 for x in cfg.values()):
        raise ValueError('Thresholds must be finite positive numbers.')
    if calibration.get('quaternion_order')!='wxyz' or calibration.get('robot_geometry_verified') is not True:
        raise ValueError('Verify robot geometry and WXYZ calibration first.')
    objcfg=calibration['objects'][clip['object']]
    primitives=collider_primitives(objcfg)
    scale=float(np.asarray(data['object_scale']).item())
    expected=objcfg.get('reference_object_scale')
    if expected is None or not np.isclose(scale,float(expected),rtol=1e-6,atol=1e-8):
        raise ValueError('Calibration object scale does not match the reference; do not apply scaling twice.')
    n=len(positions); op=object_array(data['object_pos_w'],n,3,'object position')
    oq=unit_quat(object_array(data['object_quat_w'],n,4,'object rotation'))
    bottom=collider_bottom(op,oq,primitives)
    floor=float(calibration.get('floor_z_m',0))
    if not np.isfinite(floor): raise ValueError('Invalid floor height.')
    annotations=calibration.get('contact_windows',{}).get(clip['id'])
    if annotations is not None:
        window=np.zeros(n,bool)
        for start,end in annotations:
            if not isinstance(start,int) or not isinstance(end,int) or not 0<=start<end<=n:
                raise ValueError('Contact windows must be valid end-exclusive frame intervals in this NPZ.')
            window[start:end]=True
        window_source='manually_calibrated_frame_intervals'
    else:
        if 'object_contact_sequence' not in data:
            raise ValueError('Missing contact annotation. Supply explicit contact_windows frame intervals.')
        global_contact=flag_vector(data['object_contact_sequence'],n,'object contact annotation')
        window=global_contact & (bottom>floor+cfg['airborne_clearance_m'])
        window_source='object_contact_annotation_and_collider_airborne'
    if window.sum()<max(2,int(np.ceil(cfg['min_window_s']*fps))):
        raise ValueError('Too few intended lift/contact frames; inspect collider calibration and annotations.')
    curves=dict(time=np.arange(n)/fps,window=window,bottom=bottom)
    metrics={}; reject=[]; correct=[]; contact_masks=[]
    for side in ('left','right'):
        hand=calibration['hands'][side]
        if hand.get('calibrated') is not True: raise ValueError(f'{side} palm offset is not calibrated.')
        offset=vector(hand.get('offset_link_m'),3,f'{side} inner palm offset')
        if np.linalg.norm(offset)<1e-6:
            raise ValueError('Zero palm offsets are placeholders; calibrate a surface point, not the link origin.')
        link=hand['link_name']
        if link not in names: raise ValueError(f'Missing calibrated link {link}.')
        idx=names.index(link)
        palm_world=positions[:,idx]+rotate(quats[:,idx],offset)
        local=inverse_rotate(oq,palm_world-op)
        gap=collider_gap(local,primitives)
        selected=gap[window]; sites=local[window]
        drift=np.linalg.norm(sites-np.median(sites,axis=0),axis=1)
        adjacent=window[1:]&window[:-1]
        jump=float(np.abs(np.diff(gap))[adjacent].max()) if adjacent.any() else 0.
        within=(selected>=-cfg['allowed_penetration_m'])&(selected<=cfg['contact_gap_m'])
        contact_masks.append(within)
        m=dict(median_gap_cm=float(np.median(selected)*100),
            p95_gap_cm=float(np.percentile(selected,95)*100),max_gap_cm=float(selected.max()*100),
            contact_fraction_2cm=float(within.mean()),
            outside_within_1cm_fraction=float(((selected>=-cfg['allowed_penetration_m'])&(selected<=.01)).mean()),
            max_penetration_cm=float(max(0.,-selected.min())*100),
            p95_drift_cm=float(np.percentile(drift,95)*100),
            median_drift_cm=float(np.median(drift)*100),max_gap_jump_cm=jump*100,
            far_fraction=float((selected>cfg['reject_gap_m']).mean()))
        axis=objcfg.get('grasp_axis')
        if axis is not None:
            if axis not in (0,1,2) or isinstance(axis,bool): raise ValueError('grasp_axis must be 0, 1 or 2.')
            expected_side=hand.get('expected_box_side')
            if expected_side not in (-1,1): raise ValueError('Specify expected_box_side -1/+1 for the calibrated grasp axis.')
            wrong=sites[:,axis]*expected_side < -cfg['side_tolerance_m']
            m['wrong_side_fraction']=float(wrong.mean())
            if wrong.mean()>cfg['max_wrong_side_fraction']: reject.append(side+'_wrong_box_side')
        if m['far_fraction']>cfg['reject_far_fraction']: reject.append(side+'_large_or_frequent_gap')
        if -selected.min()>cfg['reject_penetration_m']: reject.append(side+'_deep_penetration')
        if jump>cfg['reject_gap_jump_m']: reject.append(side+'_erratic_gap')
        if within.mean()<cfg['min_contact_fraction']: correct.append(side+'_insufficient_contact_coverage')
        if np.percentile(selected,95)>cfg['max_p95_gap_m']: correct.append(side+'_p95_gap')
        if -selected.min()>cfg['allowed_penetration_m']: correct.append(side+'_penetration')
        if np.percentile(drift,95)>cfg['max_p95_drift_m']: correct.append(side+'_contact_site_drift')
        if jump>cfg['max_gap_jump_m']: correct.append(side+'_gap_jump')
        metrics[side]=m
        curves[side+'_gap']=gap;curves[side+'_sites']=local
        curves[side+'_drift']=np.linalg.norm(local-np.median(sites,axis=0),axis=1)
    bilateral_fraction=float(np.logical_and(*contact_masks).mean())
    if bilateral_fraction<cfg['min_contact_fraction']: correct.append('insufficient_simultaneous_two_palm_contact')
    status='reject' if reject else ('correct' if correct else 'accept_geometry')
    return dict(clip=clip['id'],qa_status=status,qa_reason=reject+correct,metrics=metrics,
                window_source=window_source,contact_frames=int(window.sum()),fps=fps,thresholds=cfg,
                bilateral_contact_fraction=bilateral_fraction,
                scope='Calibrated point-to-collider geometry only. Not a physics/AMP-training approval.'),curves


def graph_and_csv(folder,report,curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (folder/'frames.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['time_s','contact_window','bottom_m','left_gap_m','right_gap_m',
            'left_drift_m','right_drift_m','left_x_box','left_y_box','left_z_box','right_x_box','right_y_box','right_z_box'])
        for i,t in enumerate(curves['time']):
            writer.writerow([t,int(curves['window'][i]),curves['bottom'][i],curves['left_gap'][i],curves['right_gap'][i],
                curves['left_drift'][i],curves['right_drift'][i],*curves['left_sites'][i],*curves['right_sites'][i]])
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    t=curves['time'];mask=curves['window']; cfg=report['thresholds']
    for side in ('left','right'):
        axes[0].plot(t,100*curves[side+'_gap'],label=side+' signed gap')
        axes[1].plot(t,100*curves[side+'_drift'],label=side+' box-frame drift')
    for limit in (cfg['contact_gap_m'],-cfg['allowed_penetration_m']): axes[0].axhline(limit*100,ls='--',color='k',alpha=.5)
    axes[1].axhline(cfg['max_p95_drift_m']*100,ls='--',color='k',alpha=.5)
    axes[2].plot(t,curves['bottom'],label='collider bottom (m)')
    axes[2].plot(t,mask.astype(int),label='intended lift/contact window')
    for ax in axes:
        ax.grid(alpha=.2);ax.legend()
        ax.fill_between(t,0,1,where=mask,transform=ax.get_xaxis_transform(),color='green',alpha=.08)
    axes[0].set_ylabel('gap / penetration (cm)');axes[1].set_ylabel('drift (cm)');axes[2].set_xlabel('time (s)')
    fig.suptitle(report['clip']+'\n'+report['qa_status']+': '+', '.join(report['qa_reason']),fontsize=10)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(folder/'geometry_qa.png',dpi=130);plt.close(fig)


def prepare(args):
    import mujoco
    exporter=load_module(args.helpers/'t2_export_holosoma.py','qa_exporter')
    sampler=load_module(args.helpers/'t2_resample_hoi.py','qa_sampler')
    batch=args.batch.resolve(); manifest=json.loads((batch/'passed_lift_manifest.json').read_text())
    clips=manifest['clips']
    if not clips or any(c['task']!='lift' for c in clips): raise ValueError('Need a nonempty lift-only passed manifest.')
    ids=[safe_id(c['id']) for c in clips]
    if len(set(ids))!=len(ids): raise ValueError('Duplicate clip IDs.')
    settings=dict(batch=str(batch),manifest_sha256=digest(batch/'passed_lift_manifest.json'),fps=50,
        package=str(args.package.resolve()),exporter_sha256=digest(args.helpers/'t2_export_holosoma.py'),
        resampler_sha256=digest(args.helpers/'t2_resample_hoi.py'),version=1)
    marker=args.out/'prepare_settings.json'
    if args.out.exists():
        if not marker.is_file() or json.loads(marker.read_text())!=settings:
            raise ValueError('Output/settings differ. Use a fresh --out.')
    else:
        args.out.mkdir(parents=True);save(marker,settings)
    lock=(args.out/'.prepare.lock').open('a')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise ValueError('Another preparation process is using this output.')
    models=args.out/'models';models.mkdir(exist_ok=True)
    records=[];failures=[]
    oldpath=args.out/'manifest.json'
    old=json.loads(oldpath.read_text()) if oldpath.exists() else {}
    oldclips={c['id']:c for c in old.get('clips',[])}
    for i,c in enumerate(clips,1):
        cid=c['id']; source=batch/'lift'/cid/'contact_window.pkl'
        try:
            state=json.loads((source.parent/'status.json').read_text())
            if state.get('status')!='completed': raise ValueError('Retargeted clip is not completed.')
            sha=digest(source)
            prior=oldclips.get(cid)
            if prior and prior.get('source_pickle_sha256')==sha:
                target=asset_path(args.out,prior['npz'])
                if (target.is_file() and prior.get('npz_sha256')==digest(target)
                    and asset_path(args.out,prior['model_xml']).is_file()
                    and prior.get('model_xml_sha256')==digest(asset_path(args.out,prior['model_xml']))):
                    records.append(prior);print(f'[{i}/{len(clips)}] RESUME {cid}',flush=True);continue
            with source.open('rb') as f: motion=pickle.load(f)
            if float(motion['fps'])!=50:
                motion=sampler.resample(motion,50)
                # Exporter recomputes robot body kinematics in the destination model.
                # No HOI body FK is needed here: avoid mixing robot body layouts.
            clip_models=models/cid;clip_models.mkdir(exist_ok=True)
            model,xml,urdf,scale=exporter.make_model(args.package,motion,c['object'],clip_models)
            arrays=exporter.export_motion(motion,model,c['object'])
            target=args.out/'motions'/'lift'/(cid+'.npz');target.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target,**arrays);read_npz(target)
            record=dict(c,npz=str(target.relative_to(args.out)),model_xml=str(xml.relative_to(args.out)),
                object_urdf=str(urdf.relative_to(args.out)),object_geometry_scale=float(scale),fps=50,
                frames=len(arrays['joint_pos']),source_pickle=str(source),source_pickle_sha256=sha,
                npz_sha256=digest(target),model_xml_sha256=digest(xml))
            records.append(record);print(f'[{i}/{len(clips)}] OK {cid}',flush=True)
        except Exception as exc:
            failures.append(dict(id=cid,error=f'{type(exc).__name__}: {exc}'))
            print(f'[{i}/{len(clips)}] FAIL {cid}: {exc}',flush=True)
        save(oldpath,dict(clips=records,failures=failures,fps=50,quaternion_order='wxyz',mujoco_version=mujoco.__version__))
    save(oldpath,dict(clips=records,failures=failures,fps=50,quaternion_order='wxyz',mujoco_version=mujoco.__version__))
    print(f'Prepared {len(records)}/{len(clips)} at {args.out}',flush=True)
    if failures: raise RuntimeError('Preparation incomplete; rerun the same command to retry.')


def inventory(args):
    manifest=json.loads((args.root/'manifest.json').read_text())
    result=[]; objects={}
    for c in manifest['clips']:
        path=asset_path(args.root,c['npz']);data,p,q,names,fps=read_npz(path)
        model=asset_path(args.root,c['model_xml']); tree=ET.parse(model).getroot()
        hand_nodes=[]
        for node in tree.findall('.//body'):
            name=node.get('name','')
            if any(word in name.lower() for word in ('wrist','hand','palm')):
                hand_nodes.append(dict(body=name,attributes=node.attrib,geoms=[g.attrib for g in node.findall('geom')]))
        urdf=asset_path(args.root,c['object_urdf'])
        objxml=ET.parse(urdf).getroot()
        result.append(dict(clip=c['id'],object=c['object'],fps=fps,frames=len(p),npz=str(path),
            model_xml=str(model),model_sha256=digest(model),object_urdf=str(urdf),
            reference_object_scale=float(data['object_scale'].item()),
            hand_body_names=[n for n in names if any(x in n.lower() for x in ('wrist','palm','hand'))],
            model_hand_geoms=hand_nodes,object_collision_xml=[ET.tostring(n,encoding='unicode') for n in objxml.findall('.//collision')],
            contact_fields={k:list(data[k].shape) for k in ('object_contact_sequence','per_link_contact_flags','contact_link_names') if k in data}))
        objects.setdefault(c['object'],dict(verified=False,reference_object_scale=float(data['object_scale'].item()),
            grasp_axis=None,boxes=[dict(center_object_m=None,quat_object_wxyz=[1,0,0,0],half_extents_m=None)]))
    args.out.mkdir(parents=True,exist_ok=True)
    save(args.out/'inventory.json',dict(clips=result,failures=manifest.get('failures',[])))
    template=dict(quaternion_order='wxyz',robot_geometry_verified=False,floor_z_m=0.,
        hands={side:dict(calibrated=False,link_name=side+'_wrist_yaw_link',offset_link_m=None,expected_box_side=None)
               for side in ('left','right')},objects=objects,thresholds=THRESHOLDS,contact_windows={})
    calibration=args.out/'calibration.json'
    if not calibration.exists(): save(calibration,template)
    print('Inventory:',args.out/'inventory.json')
    print('Calibration template (never overwritten):',calibration)
    print('Measure palm offsets/collider dimensions in the matching assets. Null/unverified values cannot pass QA.')


def qa(args):
    manifest=json.loads((args.root/'manifest.json').read_text())
    if manifest.get('failures'): raise ValueError('NPZ export is incomplete; fix preparation failures first.')
    calibration=json.loads(args.calibration.read_text())
    if args.out.exists(): raise ValueError('QA output exists. Use a fresh --out to preserve prior decisions.')
    args.out.mkdir(parents=True);reports=[];passed=[]
    for c in manifest['clips']:
        cid=safe_id(c['id']);folder=args.out/cid;folder.mkdir()
        try:
            data,p,q,names,fps=read_npz(asset_path(args.root,c['npz']))
            report,curves=evaluate(data,p,q,names,fps,calibration,c)
            graph_and_csv(folder,report,curves)
            if report['qa_status']=='accept_geometry': passed.append(c)
        except Exception as exc:
            report=dict(clip=cid,qa_status='needs_data_or_calibration',qa_reason=[f'{type(exc).__name__}: {exc}'])
        save(folder/'report.json',report);reports.append(report)
        print(cid,report['qa_status'],', '.join(report['qa_reason']),flush=True)
    rows=[]
    for r in reports:
        row=dict(clip=r['clip'],qa_status=r['qa_status'],qa_reason='; '.join(r['qa_reason']))
        if 'bilateral_contact_fraction' in r: row['bilateral_contact_fraction']=r['bilateral_contact_fraction']
        for side in ('left','right'):
            for key,value in r.get('metrics',{}).get(side,{}).items(): row[side+'_'+key]=value
        if 'metrics' in r:
            row['max_penetration_cm']=max(r['metrics'][s]['max_penetration_cm'] for s in ('left','right'))
        rows.append(row)
    fields=['clip']+sorted({k for row in rows for k in row if k not in ('clip','qa_status','qa_reason')})+['qa_status','qa_reason']
    with (args.out/'contact_qa.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    save(args.out/'summary.json',dict(reports=reports,calibration=str(args.calibration),calibration_sha256=digest(args.calibration)))
    save(args.out/'geometry_pass_manifest.json',dict(clips=passed,source_root=str(args.root),
        note='Geometry screening only; collider visualization and dynamic PhysX validation still required.'))
    print('Summary CSV:',args.out/'contact_qa.csv')
    print('Geometry passes:',len(passed),'/',len(reports))


def inspect_project(args):
    import importlib.metadata
    lines=['Installed package versions (run this in hssim for the simulation environment):']
    for name in ('isaacsim','isaaclab','holosoma','torch'):
        try: lines.append(name+': '+importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError: lines.append(name+': not registered in this environment')
    root=args.repo/'src'
    lines.append('\nReplay entry points:')
    lines.extend(str(p) for p in sorted(root.rglob('replay.py')))
    pattern=re.compile(r'ContactSensorCfg|activate_contact_sensors|force_matrix_w|g1-29dof-wbt-w-object')
    lines.append('\nRelevant installed source locations (not patched):')
    count=0
    for path in sorted(root.rglob('*.py')):
        if count>=50: break
        for i,line in enumerate(path.read_text(errors='replace').splitlines()):
            if pattern.search(line):
                lines.append(f'{path}:{i+1}: {line.strip()}');count+=1
                if count>=50: break
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text('\n'.join(lines)+'\n')
    print('Simulation integration inventory:',args.out)


def main():
    home=Path.home()/'holosoma/t2';parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare',help='Resample and export only passing HOI lift clips; safe to resume.')
    p.add_argument('--batch',type=Path,default=home/'hoi_all_lift_candidates01')
    p.add_argument('--out',type=Path,default=home/'lift24_npz_contactqa')
    p.add_argument('--helpers',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--package',type=Path,default=Path.home()/'holosoma/src/holosoma_retargeting/holosoma_retargeting')
    p=sub.add_parser('inventory',help='List real body/asset names and create a calibration template.')
    p.add_argument('--root',type=Path,default=home/'lift24_npz_contactqa')
    p.add_argument('--out',type=Path,default=home/'lift24_contact_calibration')
    p=sub.add_parser('qa',help='Generate calibrated point/collider gap graphs and clip decisions.')
    p.add_argument('--root',type=Path,default=home/'lift24_npz_contactqa')
    p.add_argument('--calibration',type=Path,default=home/'lift24_contact_calibration/calibration.json')
    p.add_argument('--out',type=Path,default=home/'lift24_contact_qa01')
    p=sub.add_parser('inspect-project',help='Read installed simulator versions, replay paths and existing contact hooks.')
    p.add_argument('--repo',type=Path,default=Path.home()/'holosoma')
    p.add_argument('--out',type=Path,default=home/'lift24_contact_calibration/sim_project_inventory.txt')
    args=parser.parse_args()
    for k,v in vars(args).items():
        if isinstance(v,Path): setattr(args,k,v.expanduser().resolve())
    try: {'prepare':prepare,'inventory':inventory,'qa':qa,'inspect-project':inspect_project}[args.command](args)
    except (ValueError,RuntimeError,FileNotFoundError,KeyError) as exc: parser.exit(1,f'ERROR: {exc}\n')


if __name__=='__main__': main()
