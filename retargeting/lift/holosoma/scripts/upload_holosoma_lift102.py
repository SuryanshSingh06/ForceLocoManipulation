#!/usr/bin/env python3
"""Publish the completed Holosoma lift batch from the laptop, in a new Git branch."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

REPO='SuryanshSingh06/ForceLocoManipulation'
BASE='lift-retargeting-assets-20261025-195739'
DEST=Path('retargeting/lift/holosoma')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())

def write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,indent=2)+'\n')

def command(args,cwd=None,capture=False):
    return subprocess.run([str(a) for a in args],cwd=cwd,check=True,
                          text=True,stdout=subprocess.PIPE if capture else None).stdout

def safe(cid):
    if not isinstance(cid,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',cid):
        raise ValueError(f'Unsafe clip ID: {cid!r}')
    return cid

def verify_npz(path,frames=None,fps50=False):
    import numpy as np
    with np.load(path,allow_pickle=False) as data:
        q=data['qpos'];fps=float(np.asarray(data['fps']).item())
        if q.ndim!=2 or q.shape[1]!=43 or len(q)<2 or not np.isfinite(q).all():
            raise ValueError(f'Invalid G1/object motion: {path}')
        if frames is not None and len(q)!=frames:raise ValueError(f'Frame mismatch: {path}')
        if not np.isfinite(fps) or fps<=0 or (fps50 and fps!=50):raise ValueError(f'Invalid FPS: {path}')
        for section in (q[:,3:7],q[:,39:43]):
            if np.max(abs(np.linalg.norm(section,axis=1)-1))>.02:raise ValueError(f'Invalid quaternion: {path}')
        return dict(frames=len(q),fps=fps,qpos_shape=list(q.shape))

def plan(run,selection):
    rows=read(run/'manifest.json')['clips']
    if len(rows)!=104 or len({safe(c['id']) for c in rows})!=104 or any(c['task']!='lift' for c in rows):
        raise ValueError('Expected 104 distinct lift attempts.')
    completed=[];failed=[]
    for row in rows:
        folder=run/'lift'/row['id'];state=read(folder/'status.json')
        if state['status']!=row['status']:raise ValueError(f'Manifest/status mismatch: {row["id"]}')
        if state['status']=='completed':
            for name,key in [('motion_original.npz','output_sha256'),('motion_fps50.npz','fps50_sha256')]:
                path=folder/name
                if sha(path)!=state.get(key):raise ValueError(f'Hash mismatch: {path}')
            original=verify_npz(folder/'motion_original.npz',int(row['end'])-int(row['start']))
            resampled=verify_npz(folder/'motion_fps50.npz',fps50=True)
            for name in ('selection.json','retarget.log'):
                if not (folder/name).is_file():raise FileNotFoundError(folder/name)
            completed.append(dict(row,original=original,resampled=resampled))
        else:
            failed.append(dict(row))
    if len(completed)!=102 or len(failed)!=2:raise ValueError(f'Expected 102 completed / 2 failed; got {len(completed)} / {len(failed)}.')
    selected=read(selection)
    picks=selected['clips'] if isinstance(selected,dict) else selected
    ids={safe(c['id']) for c in picks}
    if len(picks)!=24 or len(ids)!=24 or not ids.issubset({c['id'] for c in completed}):
        raise ValueError('Expected exactly 24 selected IDs, all completed.')
    by_id={c['id']:c for c in completed}
    for c in picks:
        for key in ('task','start','end','object'):
            if key in c and c[key]!=by_id[c['id']].get(key):raise ValueError(f'Selected metadata mismatch: {c["id"]}/{key}')
    return completed,failed,ids

def copy(source,target):
    if source.is_symlink():raise ValueError(f'Refusing symlink: {source}')
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)

def portable_models(source,target,objects):
    # Copy the complete standard-G1 asset directory and each used object directory.
    for name in ['g1',*sorted(objects)]:
        directory=source/name
        if not directory.is_dir():raise FileNotFoundError(directory)
        for p in directory.rglob('*'):
            if p.is_symlink():raise ValueError(f'Model symlink: {p}')
        shutil.copytree(directory,target/name)
    for p in target.rglob('*'):
        if p.suffix not in ('.xml','.urdf'):continue
        tree=ET.parse(p);compiler=tree.find('compiler')
        meshdir=compiler.get('meshdir','') if compiler is not None else ''
        # Generated combined XMLs have absolute private-runtime mesh paths.
        # Normalize references against their actual loader base directory.
        for node in tree.findall('.//mesh'):
            attr='filename' if p.suffix=='.urdf' else 'file'
            value=node.get(attr)
            if not value:continue
            base=p.parent/(meshdir if p.suffix=='.xml' else '')
            if str(source.resolve()) in value:
                value=value.replace(str(source.resolve()),str(target.resolve()))
            if value.startswith('file://'):value=value[7:]
            candidate=Path(value) if Path(value).is_absolute() else base/value
            if not candidate.is_file():
                # Raw object URDFs may still reference objects/<name>/<name>.obj.
                matches=[f for f in target.rglob(Path(value).name) if f.is_file()]
                if not matches or len({sha(f) for f in matches})!=1:
                    raise FileNotFoundError(f'Cannot resolve model mesh {p}: {value}')
                candidate=matches[0]
            candidate=candidate.resolve()
            if not candidate.is_relative_to(target.resolve()):raise ValueError(f'Mesh outside publication: {candidate}')
            node.set(attr,os.path.relpath(candidate,base.resolve()))
        tree.write(p,encoding='utf-8',xml_declaration=True)
    for obj in objects:
        model=target/'g1'/f'g1_29dof_w_{obj}.xml'
        if not model.is_file():raise FileNotFoundError(model)
        # Loading the actual portable combined model catches missing MJCF assets.
        import mujoco
        if mujoco.MjModel.from_xml_path(str(model)).nq!=43:raise ValueError(f'Wrong model nq: {model}')

def collect(run,selection,package,target,completed,failed,picked):
    if target.exists():raise FileExistsError(target)
    target.mkdir(parents=True)
    manifest=[]
    for c in completed:
        cid=c['id'];folder=run/'lift'/cid;out=target/'clips'/cid
        for name in ('motion_original.npz','motion_fps50.npz','status.json','selection.json','retarget.log',
                     'recovery_events.json','constraint_audit.json','preserved_status.json'):
            if (folder/name).exists():copy(folder/name,out/name)
        manifest.append(dict(id=cid,task='lift',object=c['object'],source_id=c.get('source_id'),
                             source_original=c['source'],start=c['start'],end=c['end'],
                             source_sha256=c.get('sha256') or c.get('fingerprint',{}).get('source_sha256'),
                             t1_review=c.get('review'),robot_motion_review='pending',
                             selected24=cid in picked,backend='holosoma_object_interaction',
                             original=dict(path=f'clips/{cid}/motion_original.npz',sha256=sha(out/'motion_original.npz'),**c['original']),
                             fps50=dict(path=f'clips/{cid}/motion_fps50.npz',sha256=sha(out/'motion_fps50.npz'),**c['resampled']),
                             constraint_audit=f'clips/{cid}/constraint_audit.json' if (out/'constraint_audit.json').exists() else None))
    for row in failed:
        cid=row['id']
        for name in ('status.json','retarget.log','recovery_events.json','constraint_audit.json'):
            source=run/'lift'/cid/name
            if source.exists():copy(source,target/'failures'/cid/name)
    counts=dict(Counter(c['object'] for c in completed))
    write(target/'manifest.json',dict(backend='holosoma',task_type='object_interaction',robot='g1',
          attempted=104,completed=102,failed=2,object_counts=counts,clips=manifest,
          schema='qpos_playback_npz_not_training_arrays',quaternion_order='wxyz',
          note='Numerical completion is not robot-motion approval or physical grasp validation.'))
    write(target/'selected24/manifest.json',dict(expected=24,completed=24,
          clips=[dict(c,original=dict(c['original'],path='../'+c['original']['path']),
                      fps50=dict(c['fps50'],path='../'+c['fps50']['path']),
                      constraint_audit='../'+c['constraint_audit'] if c['constraint_audit'] else None)
                 for c in manifest if c['id'] in picked]))
    (target/'selected24/README.md').write_text('# Selected 24\n\nThe manifest identifies the same 24 selected lift IDs. Paths are relative to\nthis directory and reference the shared clips/ files one level up. Files are\nnot duplicated. Selection does not certify grasp quality.\n')
    write(target/'failures/manifest.json',dict(count=2,clips=failed))
    for name in ('manifest.json','failures.json','recovery_settings.json','settings.json'):
        if (run/name).exists():copy(run/name,target/'provenance'/name)
    copy(selection,target/'provenance/selected_source_manifest.json')
    recovery=run.parent/'holosoma_lift_recovery'
    batch=run.parent/'holosoma_lift_batch'
    for directory,label in ((recovery,'recovery'),(batch,'initial_batch')):
        for p in directory.glob('*'):
            if p.is_file() and p.suffix in ('.py','.sh','.md'):copy(p,target/'scripts'/label/p.name)
    if not (target/'scripts/recovery/t2_recover_holosoma_lifts.py').exists():
        raise FileNotFoundError('Missing local recovery script snapshot.')
    # Python source only. No demo recordings, body-model weights or environments.
    for p in package.rglob('*.py'):
        rel=p.relative_to(package)
        if any(part.startswith('.') or part in ('models','demo_data','__pycache__','build','dist') for part in rel.parts):continue
        copy(p,target/'scripts/installed_holosoma_retargeting'/rel)
    copy(Path(__file__),target/'scripts'/Path(__file__).name)
    metadata=run/'runtime/demo_data/height_dict.pkl'
    if metadata.is_file():copy(metadata,target/'scripts/runtime_metadata/height_dict.pkl')
    revision=command(['git','rev-parse','HEAD'],package,True).strip()
    dirty=command(['git','status','--porcelain'],package,True)
    write(target/'provenance/installed_source_revision.json',dict(commit=revision,working_tree_status=dirty,
          note='Tracked Python files are snapshotted under scripts/installed_holosoma_retargeting.'))
    portable_models(run/'runtime/models',target/'models',{c['object'] for c in completed})
    (target/'environment.txt').write_text(command([sys.executable,'-m','pip','freeze'],capture=True))
    failed_ids='\n'.join('- '+c['id'] for c in failed)
    object_rows='\n'.join(f'| {obj} | {n} |' for obj,n in sorted(counts.items()))
    (target/'README.md').write_text(f'''# Holosoma G1 lift references: 102 completed clips

This folder contains **Holosoma retargeting outputs**, not HOI-Retarget outputs.
The task was object_interaction, robot g1, data format smplh. Of 104 lift
attempts, 102 completed and two failed. All 24 previously selected lift IDs are
included in the 102. No carry or push/pull motions are added by this upload.

| Object | Completed clips |
| --- | ---: |
{object_rows}

## File structure

```text
manifest.json                         Portable index of all 102 completed clips
clips/<clip-id>/motion_original.npz    Native frame-rate robot/object qpos
clips/<clip-id>/motion_fps50.npz       50 Hz quaternion-aware resampling
clips/<clip-id>/status.json            Original execution/provenance details
clips/<clip-id>/selection.json         T1 source recording and crop interval
clips/<clip-id>/retarget.log           Optimization log
clips/<clip-id>/recovery_events.json   Recovery events, when available
clips/<clip-id>/constraint_audit.json  Nonlinear audit, when available
selected24/manifest.json              References to the matching 24 shared clips
models/g1/                            Matching standard G1 model/geometry
models/<object>/                      Object URDF/mesh used by the runtime
failures/                             Two failed attempts: logs and metadata only
provenance/                           Original run/selection manifests and settings
scripts/                              Runner, recovery and installed Python snapshots
environment.txt                       Exporting environment package versions
checksums.json                        SHA256 and size of published files
```

Paths in manifest.json are relative to this folder. Paths in selected24/manifest.json
are relative to selected24/. Original status/provenance files retain historical
laptop paths as evidence; those are not portable loader paths.

## Motion format and use

qpos has 43 columns: root XYZ [0:3], root quaternion WXYZ [3:7], 29 G1 joint
positions [7:36], object XYZ [36:39], object quaternion WXYZ [39:43]. Read fps
from each NPZ. All motion_fps50 files have fps=50. Quaternion-aware interpolation
uses a uniform 50 Hz grid without extrapolating beyond the native final timestamp.
These are **qpos playback NPZs, not Holosoma joint_pos/body_pos training-schema
exports**. Training-array conversion and physical validation remain separate work.
Source human trajectories may also be present inside the NPZs. Original source
.pt recordings, licensed body-model weights, checkpoints and videos are not
added by this uploader. This is not a complete standalone source-data archive.
Preserve the recorded source filenames, checksums and [start:end] frame intervals
when obtaining source recordings to reproduce retargeting.

Robot/object geometry is copied from the private runtime without deliberate
rescaling. Mesh references in copied XML/URDF files are made relative; original
installed models are not edited. Some combined object MJCFs were generated from
the largebox template and retain its inertial parameters: these are kinematic
reference assets, not calibrated physical mass/inertia models.

## Generation and recovery

The native runner used --task-type object_interaction --robot g1 --data-format
smplh with object nonpenetration, joint limits and foot sticking enabled.
Recovery first enlarged the pose-update trust region to at most 0.8, then
allowed foot-sticking XY tolerances up to 10 mm if needed. The final recovery
also tried 4/8/16/32 internal object-motion subdivisions, preserving each output
frame's requested object endpoint and retaining previous-output-frame foot
anchoring. See script snapshots and per-clip events for exact behavior.

The final run preserved 88 earlier results and newly recovered 14. Only newly
solved clips received that version's frame-by-frame nonlinear constraint audit.
Absence of an audit is not a pass. Checks cover the retargeter's configured
collision masks/pairs and joint limits; they are not physical support tests.
Per-frame foot tolerances can accumulate drift. Larger updates can cause jumps.
Every clip remains pending robot-motion review, regardless of its T1 review.
Completion does not certify hand contact, realistic grasp forces, stability,
collision-free training behavior, or suitability as an AMP reference.

## The two failures

{failed_ids}

They have logs/metadata under failures/ and are excluded from the motion index.

## Download

Clone this published branch with Git LFS, then run git lfs pull and git lfs fsck.
checksums.json can verify the hydrated asset bytes. Existing HOI selected-24
artifacts remain elsewhere in this branch under retargeting/lift/artifacts/selected24;
they use a different backend and are not replaced by these Holosoma results.
The model/software licensing terms of the upstream projects still apply.
''')
    (target/'.gitattributes').write_text('*.npz filter=lfs diff=lfs merge=lfs -text\n*.obj filter=lfs diff=lfs merge=lfs -text\n*.stl filter=lfs diff=lfs merge=lfs -text\n*.png filter=lfs diff=lfs merge=lfs -text\n*.jpg filter=lfs diff=lfs merge=lfs -text\n*.dae filter=lfs diff=lfs merge=lfs -text\n')
    inventory=[dict(path=str(p.relative_to(target)),sha256=sha(p),bytes=p.stat().st_size)
               for p in sorted(target.rglob('*')) if p.is_file()]
    write(target/'checksums.json',dict(files=inventory))

def main():
    home=Path.home()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=home/'ForceLocoManipulation')
    parser.add_argument('--run',type=Path,default=home/'holosoma/t2/holosoma_lift_all04')
    parser.add_argument('--selection',type=Path,default=home/'holosoma/t2/hoi_all_lift_candidates01/passed_lift_manifest.json')
    parser.add_argument('--package',type=Path,default=home/'holosoma/src/holosoma_retargeting/holosoma_retargeting')
    parser.add_argument('--push',action='store_true')
    args=parser.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.expanduser().resolve())
    completed,failed,picked=plan(args.run,args.selection)
    print('Validated:',len(completed),'completed,',len(failed),'failed,',len(picked),'selected.',flush=True)
    remote=command(['git','remote','get-url','origin'],args.repo,True).strip()
    if not re.fullmatch(r'(https://github\.com/|git@github\.com:)'+re.escape(REPO)+r'(\.git)?/?',remote):
        raise ValueError(f'Unexpected repository origin: {remote}')
    command(['git','lfs','version'],args.repo)
    command(['git','fetch','origin',f'refs/heads/{BASE}'],args.repo)
    branch='holosoma-lift102-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    temporary=Path(tempfile.mkdtemp(prefix='holosoma_lift102_upload_'))
    checkout=temporary/'checkout'
    # Prevent checkout of inherited LFS payloads merely to add this new package.
    env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1')
    subprocess.run(['git','-C',str(args.repo),'worktree','add','-b',branch,str(checkout),'FETCH_HEAD'],check=True,env=env)
    target=checkout/DEST
    collect(args.run,args.selection,args.package,target,completed,failed,picked)
    command(['git','lfs','install','--local'],checkout)
    command(['git','config','user.name','Rudra Hazari'],checkout)
    command(['git','config','user.email','Rudra@hazari.com'],checkout)
    command(['git','add','-f','--',str(DEST)],checkout)
    command(['git','diff','--cached','--stat'],checkout)
    command(['git','commit','-m','Add 102 Holosoma G1 lift references, selected 24 index and recovery documentation'],checkout)
    print('Prepared worktree:',checkout,flush=True)
    if args.push:
        command(['git','push','-u','origin',branch],checkout)
        local=command(['git','rev-parse','HEAD'],checkout,True).strip()
        remote_head=command(['git','ls-remote','origin',f'refs/heads/{branch}'],checkout,True).split()[0]
        if local!=remote_head:raise RuntimeError('Remote branch does not match the local commit.')
        print(f'Published: https://github.com/{REPO}/tree/{branch}/{DEST}',flush=True)
    else:
        print('Not pushed. Publish with:',flush=True)
        print(f'git -C {checkout} push -u origin {branch}',flush=True)

if __name__=='__main__':main()
