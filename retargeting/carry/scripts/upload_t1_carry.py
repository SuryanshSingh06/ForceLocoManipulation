#!/usr/bin/env python3
"""Publish end-of-T1 carry inputs, not retargeted robot motions.
Default collects locally; --push creates and pushes a new branch using local Git.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

REPO='SuryanshSingh06/ForceLocoManipulation'
PREFIX=Path('retargeting/carry')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def safe(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',value):raise ValueError(f'Unsafe ID/object name: {value!r}')
    return value


def build_plan(manifest_path,scope):
    original=json.loads(manifest_path.read_text())
    clips=[dict(c) for c in original['clips'] if c.get('task')=='carry' and (c.get('review')=='accepted' if scope=='accepted' else c.get('review')!='rejected')]
    if not clips:raise ValueError('No carry clips in the requested scope.')
    seen=set();sources={};assets={};hashes={}
    for c in clips:
        cid=safe(c['id']);obj=safe(c['object'])
        if cid in seen:raise ValueError(f'Duplicate clip ID: {cid}')
        seen.add(cid)
        if not 0<=int(c['start'])<int(c['end']):raise ValueError(f'Invalid frame range: {cid}')
        source=Path(c['source']).expanduser()
        if not source.is_absolute():source=manifest_path.parent/source
        source=source.resolve()
        if not source.is_file() or source.suffix.lower()!='.pt':raise FileNotFoundError(f'Original .pt recording missing: {source}')
        if source not in hashes:hashes[source]=sha(source)
        value=hashes[source]
        if c.get('sha256') and c['sha256']!=value:raise ValueError(f'T1 source checksum mismatch: {cid}')
        relative=Path('source_recordings')/value[:16]/source.name
        if relative in sources and sources[relative]!=source and hashes[sources[relative]]!=value:raise ValueError('Source destination collision.')
        sources[relative]=source
        c.update(source_original=str(source),source=relative.as_posix(),sha256=value)
        if obj not in assets:
            metadata=original.get('meshes',{}).get(obj)
            if not isinstance(metadata,dict) or not metadata.get('path'):raise ValueError(f'Missing T1 object mesh metadata: {obj}')
            mesh=Path(metadata['path']).expanduser()
            if not mesh.is_absolute():mesh=manifest_path.parent/mesh
            mesh=mesh.resolve()
            if not mesh.is_file():raise FileNotFoundError(mesh)
            value=sha(mesh)
            if metadata.get('sha256') and metadata['sha256']!=value:raise ValueError(f'T1 object mesh checksum mismatch: {obj}')
            assets[obj]=(mesh,value)
    portable={k:original[k] for k in ('version','script_sha256','thresholds','conventions') if k in original}
    portable.update(task='carry',scope=scope,clips=clips,data='source_recordings',
                    meshes={obj:{'path':(Path('object_assets')/obj/mesh.name).as_posix(),'sha256':value} for obj,(mesh,value) in assets.items()},
                    conventions_confirmed={k:v for k,v in original.get('conventions_confirmed',{}).items() if k in assets},
                    created=datetime.now(timezone.utc).isoformat(),
                    source_manifest_sha256=sha(manifest_path),
                    note='Original full human recordings with original start/end intervals, before retargeting. Pending reviews remain pending.')
    return portable,sources,assets


RESOLVER='''#!/usr/bin/env python3
"""Resolve portable carry source/mesh paths before invoking local retargeters."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest',type=Path,default=Path(__file__).resolve().parents[1]/'t1/manifest.json')
p.add_argument('--out',type=Path,default=Path.home()/'Downloads/t1_carry_manifest_local.json')
a=p.parse_args();src=a.manifest.expanduser().resolve();data=json.loads(src.read_text())
for c in data['clips']:
    if c['task']!='carry':raise ValueError('Expected carry only.')
    c['source']=str((src.parent/c['source']).resolve())
    if not Path(c['source']).is_file():raise FileNotFoundError(c['source'])
for m in data['meshes'].values():m['path']=str((src.parent/m['path']).resolve())
data['data']=str(src.parent/'source_recordings')
out=a.out.expanduser();out.parent.mkdir(parents=True,exist_ok=True)
if out.exists():raise FileExistsError(f'{out} exists; choose a fresh --out.')
out.write_text(json.dumps(data,indent=2)+'\\n');print('Local retargeting manifest:',out)
'''


def collect(args,destination):
    data,sources,assets=build_plan(args.manifest,args.scope)
    if destination.exists():raise FileExistsError(f'Destination exists: {destination}. Choose a fresh --out.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.carry_t1_',dir=destination.parent))
    try:
        t1=temp/'t1';t1.mkdir();records=[]
        def copy_file(source,target):
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
            value=sha(target)
            if value!=sha(source):raise ValueError(f'Source changed during collection: {source}')
            records.append(dict(path=target.relative_to(temp).as_posix(),sha256=value,bytes=target.stat().st_size))
        for relative,source in sorted(sources.items(),key=lambda x:str(x[0])):copy_file(source,t1/relative)
        suffixes={'.obj','.stl','.ply','.mtl','.png','.jpg','.jpeg','.json','.urdf','.xml'}
        for obj,(mesh,_) in assets.items():
            for source in mesh.parent.rglob('*'):
                if source.is_file() and source.suffix.lower() in suffixes:
                    if source.is_symlink():raise ValueError(f'Unexpected asset symlink: {source}')
                    copy_file(source,t1/'object_assets'/obj/source.relative_to(mesh.parent))
        (t1/'manifest.json').write_text(json.dumps(data,indent=2)+'\n')
        scripts=temp/'scripts';scripts.mkdir()
        copy_file(Path(__file__).resolve(),scripts/'upload_t1_carry.py')
        selector=args.holosoma_root/'t1/t1_select.py'
        if selector.is_file():copy_file(selector,scripts/'t1_select.py')
        (scripts/'resolve_manifest.py').write_text(RESOLVER)
        (temp/'.gitattributes').write_text('t1/source_recordings/**/*.pt filter=lfs diff=lfs merge=lfs -text\nt1/object_assets/**/*.obj filter=lfs diff=lfs merge=lfs -text\nt1/object_assets/**/*.stl filter=lfs diff=lfs merge=lfs -text\nt1/object_assets/**/*.STL filter=lfs diff=lfs merge=lfs -text\nt1/object_assets/**/*.png filter=lfs diff=lfs merge=lfs -text\nt1/object_assets/**/*.jpg filter=lfs diff=lfs merge=lfs -text\n')
        (temp/'README.md').write_text('''# Carry inputs at the end of T1, before retargeting

This collection contains original human-motion `.pt` recordings, carry-only clip labels and frame intervals, raw object meshes, and available local T1 selection code. These are the inputs used to view the human skeleton and to retarget again, not HOI/robot motion outputs or policy checkpoints.

The scope and accepted/pending counts are recorded in `t1/upload_inventory.json`. Pending clips have not been manually approved. Full source recordings are copied once per unique recording/name; `[start:end]` selects each carry segment, with end exclusive. A full recording can contain actions outside the selected interval. Source filenames and bytes are preserved for subject-dependent retargeting and checksum verification.

Motion `.pt` tensors contain skeleton/object trajectories; they are not triangle meshes of a skinned human. Object meshes are supplied separately in `t1/object_assets`. SMPL-family body-model weights and complete third-party tool installations are not included.

## Download and use

Clone this branch with Git LFS and run `git lfs pull`. The committed manifest uses paths relative to its own directory. Convert it to local absolute paths before using older scripts that resolve paths against the working directory:

```bash
python3 retargeting/carry/scripts/resolve_manifest.py
```

That writes `~/Downloads/t1_carry_manifest_local.json`. Use a fresh `--out` if it already exists. Retarget with the separately installed HOI/Holosoma tool and the appropriate G1/object assets. The newer lift-only batch script intentionally rejects carry inputs, so use a carry-capable runner.

The earlier already-retargeted 10 carry clips remain under `github_lift_carry/` on the `t3-carry` branch. This folder supplies the pre-retargeting inputs and preserves review state. Checksums verify copied files; they do not certify motion or physical grasp quality.
''')
        report=dict(created_at=datetime.now(timezone.utc).isoformat(),scope=args.scope,clip_count=len(data['clips']),
                    object_counts=dict(Counter(c['object'] for c in data['clips'])),review_counts=dict(Counter(c.get('review','unknown') for c in data['clips'])),
                    original_recording_count=len(sources),files=records,source_manifest=str(args.manifest),source_manifest_sha256=data['source_manifest_sha256'])
        (t1/'upload_inventory.json').write_text(json.dumps(report,indent=2)+'\n')
        temp.rename(destination)
    except BaseException:
        shutil.rmtree(temp,ignore_errors=True);raise
    print('Carry clips:',report['clip_count'],'recordings:',report['original_recording_count'],flush=True)
    print('Review counts:',report['review_counts'],'objects:',report['object_counts'],flush=True)
    print('Collected:',destination,flush=True)


def git(repo,*args,capture=False):
    result=subprocess.run(['git','-C',str(repo),*args],check=True,text=True,stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else None


def publish(args):
    repo=args.repo
    origin=git(repo,'remote','get-url','origin',capture=True).removesuffix('.git')
    if origin not in (f'https://github.com/{REPO}',f'git@github.com:{REPO}'):raise ValueError(f'Wrong origin: {origin}')
    git(repo,'lfs','version');build_plan(args.manifest,args.scope)
    git(repo,'fetch','origin','refs/heads/t3-carry:refs/remotes/origin/t3-carry')
    branch='carry-t1-inputs-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    parent=Path(tempfile.mkdtemp(prefix='carry_t1_upload_'));worktree=parent/'checkout'
    git(repo,'worktree','add','-b',branch,str(worktree),'origin/t3-carry')
    target=worktree/PREFIX
    collect(args,target)
    git(worktree,'lfs','install','--local')
    git(worktree,'add','-f','--',str(PREFIX))
    git(worktree,'commit','-m','Add pre-retargeting T1 carry recordings, labels and object meshes')
    git(worktree,'push','-u','origin',branch)
    print(f'Published: https://github.com/{REPO}/tree/{branch}/{PREFIX}',flush=True)
    print('Existing branches and retargeted clips were not replaced. The upload worktree is:',worktree,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--holosoma-root',type=Path,default=Path.home()/'holosoma')
    p.add_argument('--manifest',type=Path)
    p.add_argument('--repo',type=Path,default=Path.home()/'ForceLocoManipulation')
    p.add_argument('--scope',choices=['nonrejected','accepted'],default='nonrejected')
    p.add_argument('--out',type=Path,default=Path.home()/'Downloads/t1_carry_pre_retarget')
    p.add_argument('--push',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();a.holosoma_root=a.holosoma_root.expanduser().resolve()
    a.manifest=(a.manifest or a.holosoma_root/'t1/run05/manifest.json').expanduser().resolve()
    a.repo=a.repo.expanduser().resolve();a.out=a.out.expanduser().resolve()
    try:
        if a.dry_run:
            data,sources,assets=build_plan(a.manifest,a.scope)
            print('Carry clips:',len(data['clips']),'recordings:',len(sources),'objects:',list(assets))
            print('Review counts:',dict(Counter(c.get('review','unknown') for c in data['clips'])))
        elif a.push:publish(a)
        else:collect(a,a.out)
    except (ValueError,FileNotFoundError,FileExistsError,subprocess.CalledProcessError) as exc:p.exit(1,f'ERROR: {exc}\n')


if __name__=='__main__':main()
