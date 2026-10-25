#!/usr/bin/env python3
"""Holosoma object-interaction retargeting: all nonrejected T1 lifts plus a copied selected-24 subset.

Run in hsretargeting. This invokes Holosoma, not HOI. Existing HOI data is read
only to identify the same 24 IDs; no HOI motion is reused for optimization.
Outputs are qpos NPZ references, NOT the final Holosoma training-array schema.
"""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


def stamp():return datetime.now(timezone.utc).isoformat()

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def json_save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)

def read(path):return json.loads(Path(path).read_text())

def safe(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',value):raise ValueError(f'Unsafe ID/object: {value!r}')
    return value


def selections(manifest,selected,include_rejected=False,expected=24):
    rows=manifest['clips'];all_lifts=[c for c in rows if c.get('task')=='lift' and (include_rejected or c.get('review')!='rejected')]
    by_id={}
    for c in all_lifts:
        cid=safe(c['id']);safe(c['object'])
        if cid in by_id:raise ValueError(f'Duplicate T1 lift ID: {cid}')
        if not 0<=int(c['start'])<int(c['end']):raise ValueError(f'Invalid frame range: {cid}')
        by_id[cid]=c
    picks=selected['clips'] if isinstance(selected,dict) else selected
    if any(c.get('task')!='lift' for c in picks):raise ValueError('The selected-24 manifest must contain lift only.')
    pick_ids=[safe(c['id']) for c in picks]
    if len(set(pick_ids))!=expected or len(pick_ids)!=expected:raise ValueError(f'Expected exactly {expected} distinct selected lift IDs, got {len(pick_ids)}.')
    for c in picks:
        if c['id'] not in by_id:raise ValueError(f"Selected ID absent from the eligible T1 lifts: {c['id']}. Check rejection status/manifest; --include-rejected is an explicit override.")
        for key in ('start','end','object','source'):
            if key in c and c[key]!=by_id[c['id']].get(key):raise ValueError(f"Selected/T1 {key} mismatch: {c['id']}")
    if not all_lifts:raise ValueError('No eligible lift clips.')
    return sorted(all_lifts,key=lambda c:c['id']),set(pick_ids)


def object_assets(models,obj,fallback):
    """Resolve raw meshes through their URDF, including nested assets and STL visuals."""
    directory=models/obj
    if (directory/(obj+'.urdf')).is_file() and (directory/(obj+'.obj')).is_file():return
    roots=[models,fallback]
    candidates=[]
    for base in roots:
        if not base.is_dir():continue
        for p in base.rglob('*'):
            if not p.is_file() or p.suffix.lower()!='.urdf':continue
            if p.stem.lower()==obj.lower() or obj.lower() in [part.lower() for part in p.parent.parts]:
                candidates.append(p)
    # Exact object filename is preferred; never infer a differently sized object.
    candidates=sorted(set(candidates),key=lambda p:(p.stem.lower()!=obj.lower(),len(p.parts),str(p)))
    diagnostics=[];chosen=None
    for urdf in candidates:
        try:
            tree=ET.parse(urdf)
            if len(tree.findall('link'))!=1:raise ValueError('expected one object link')
            visual=tree.find('.//visual/geometry/mesh')
            if visual is None:raise ValueError('expected a mesh visual')
            resolved=[]
            for node in tree.findall('.//mesh'):
                filename=node.get('filename','').replace('\\','/')
                if filename.startswith('file://'):filename=filename[7:]
                direct=(urdf.parent/filename).resolve()
                if direct.is_file():source=direct
                else:
                    basename=filename.rsplit('/',1)[-1]
                    hits=[]
                    for base in [urdf.parent,*roots]:
                        if base.is_dir():hits.extend(p for p in base.rglob('*') if p.is_file() and p.name.lower()==basename.lower())
                    hits=sorted(set(p.resolve() for p in hits))
                    if not hits:raise FileNotFoundError(f'URDF mesh not found: {filename}')
                    if len({digest(p) for p in hits})>1:raise ValueError(f'ambiguous mesh basename: {basename}')
                    source=hits[0]
                resolved.append((node,source))
            chosen=(urdf,tree,visual,resolved);break
        except (ValueError,FileNotFoundError,ET.ParseError) as exc:diagnostics.append(f'{urdf}: {exc}')
    if chosen is None:
        message='\n'.join(diagnostics[:8]) or 'No matching raw object URDF found, including nested folders.'
        raise FileNotFoundError(f'Cannot resolve raw {obj} assets under {models} or {fallback}.\n{message}\nUse --object-assets /path/to/raw/object/assets; do not substitute the scaled HOI export.')
    urdf,tree,visual,resolved=chosen
    # Stage first; an existing incomplete runtime object is replaced only after resolution.
    temporary=Path(tempfile.mkdtemp(prefix='.'+obj+'_',dir=models))
    try:
        visual_source=next(source for node,source in resolved if node is visual)
        canonical=temporary/(obj+'.obj')
        if visual_source.suffix.lower()=='.obj':shutil.copy2(visual_source,canonical)
        else:
            import trimesh
            mesh=trimesh.load(str(visual_source),force='mesh')
            if not len(mesh.vertices) or not len(mesh.faces):raise ValueError('Empty visual mesh.')
            mesh.export(str(canonical),file_type='obj')
        for node,source in resolved:
            if node is visual:
                node.set('filename',canonical.name)
            else:
                name='mesh_'+digest(source)[:12]+'_'+source.name
                target=temporary/name
                if not target.exists():shutil.copy2(source,target)
                node.set('filename',name)
        # Preserve local OBJ material companions when available.
        if visual_source.suffix.lower()=='.obj':
            for line in visual_source.read_text(errors='replace').splitlines():
                if line.startswith('mtllib '):
                    for name in line.split()[1:]:
                        source=(visual_source.parent/name).resolve()
                        if source.is_file() and Path(name).name==name:shutil.copy2(source,temporary/name)
        tree.write(temporary/(obj+'.urdf'),encoding='utf-8',xml_declaration=True)
        json_save(temporary/'raw_asset_provenance.json',dict(source_urdf=str(urdf),source_visual_mesh=str(visual_source),source_visual_sha256=digest(visual_source),canonical_obj=obj+'.obj',geometry_scale_applied=1.0))
        if directory.exists():shutil.rmtree(directory)
        temporary.rename(directory)
        print(f'Raw asset resolved: {obj} -> {visual_source}',flush=True)
    except BaseException:
        shutil.rmtree(temporary,ignore_errors=True);raise


def make_object_model(models,obj):
    """Build missing small/plastic combined MJCF from the installed standard-G1 template."""
    destination=models/'g1'/f'g1_29dof_w_{obj}.xml'
    if destination.is_file():return destination
    template=models/'g1/g1_29dof_w_largebox.xml'
    tree=ET.parse(template);root=tree.getroot()
    if root.findall('.//include'):raise ValueError('Combined MJCF includes need explicit expansion before adaptation.')
    body=root.find('.//body[@name="largebox_link"]')
    if body is None or body.findall('body') or body.find('freejoint') is None:raise ValueError('Expected the simple free largebox body in the standard-G1 template.')
    geoms=body.findall('geom')
    if not geoms or any(g.get('mesh') is None for g in geoms):raise ValueError('Expected mesh-based largebox template.')
    old_names={g.get('mesh') for g in geoms}
    # Do not silently adapt transformed meshes with a different object pose frame.
    urdf=ET.parse(models/obj/(obj+'.urdf'))
    visual=urdf.find('.//visual')
    if visual is None:raise ValueError('Object needs a mesh visual.')
    umesh=visual.find('geometry/mesh');origin=visual.find('origin')
    if umesh is None:raise ValueError('Expected an object mesh.')
    def floats(value,default):return [float(x) for x in (value or default).split()]
    if any(abs(x)>1e-8 for x in floats(origin.get('xyz') if origin is not None else None,'0 0 0')+floats(origin.get('rpy') if origin is not None else None,'0 0 0')):raise ValueError('Object URDF has a nonzero visual transform; verify its trajectory frame before adapting.')
    if any(abs(x-1)>1e-8 for x in floats(umesh.get('scale'),'1 1 1')):raise ValueError('Expected raw, unscaled source object geometry. Do not reuse the 0.83 HOI-export model.')
    asset=root.find('asset')
    if asset is None:raise ValueError('MJCF has no asset section.')
    for node in list(asset):
        if node.tag=='mesh' and node.get('name') in old_names:asset.remove(node)
    name=obj+'_mesh'
    compiler=root.find('compiler')
    if compiler is None:compiler=ET.SubElement(root,'compiler')
    compiler.set('strippath','false')
    # Absolute path is within this private runtime, so compiler meshdir cannot redirect it.
    ET.SubElement(asset,'mesh',name=name,file=str((models/obj/(obj+'.obj')).resolve()),scale='1 1 1')
    body.set('name',obj+'_link')
    for i,g in enumerate(geoms):
        g.set('mesh',name)
        if g.get('name'):g.set('name',obj if i==0 else f'{obj}_{i}')
        g.set('pos','0 0 0');g.set('quat','1 0 0 0')
        for key in ('euler','axisangle','xyaxes','zaxis'):g.attrib.pop(key,None)
    # This is kinematic retargeting. Retain template inertial parameters and disclose them.
    tree.write(destination,encoding='utf-8',xml_declaration=True)
    return destination


def prepare_runtime(package,run,objects,fallback):
    runtime=run/'runtime';ready=runtime/'ready.json'
    if ready.is_file():
        if read(ready)['objects']!=sorted(objects):raise ValueError('Runtime object list changed; use fresh output folders.')
        return runtime
    if runtime.exists():raise FileExistsError(f'Incomplete runtime exists at {runtime}; use fresh output folders.')
    temp=Path(tempfile.mkdtemp(prefix='.runtime_',dir=run))
    try:
        shutil.copytree(package/'models',temp/'models',symlinks=False)
        for obj in objects:
            object_assets(temp/'models',obj,fallback)
            make_object_model(temp/'models',obj)
        # Generated absolute paths must refer to the final runtime path, not temp.
        for p in (temp/'models/g1').glob('g1_29dof_w_*.xml'):
            content=p.read_text()
            p.write_text(content.replace(str(temp),str(runtime)))
        json_save(temp/'ready.json',dict(objects=sorted(objects),note='Private copies of standard retargeting models. Missing combined object MJCFs adapt the largebox template; object inertias remain template values.'))
        temp.rename(runtime)
    except BaseException:
        shutil.rmtree(temp,ignore_errors=True);raise
    return runtime


def verify_motion(path,frames=None):
    import numpy as np
    with np.load(path,allow_pickle=False) as f:
        q=np.asarray(f['qpos']);fps=float(np.asarray(f['fps']).item())
        if q.ndim!=2 or q.shape[1]!=43 or len(q)<2 or not np.isfinite(q).all():raise ValueError('Expected finite (T,43) standard G1 + free object qpos.')
        if frames is not None and len(q)!=frames:raise ValueError(f'Expected {frames} frames, got {len(q)}.')
        if not np.isfinite(fps) or fps<=0:raise ValueError('Invalid output FPS.')
        for section in (q[:,3:7],q[:,-4:]):
            if np.max(np.abs(np.linalg.norm(section,axis=1)-1))>.02:raise ValueError('Invalid quaternion norm.')
        return dict(frames=len(q),fps=fps,qpos_shape=list(q.shape))


def resample50(source,target):
    import numpy as np
    from scipy.spatial.transform import Rotation,Slerp
    with np.load(source,allow_pickle=False) as f:data={k:f[k] for k in f.files}
    q=np.asarray(data['qpos']);fps=float(np.asarray(data['fps']).item());t=np.arange(len(q))/fps
    new=np.arange(int(np.floor(t[-1]*50+1e-8))+1)/50
    result=np.stack([np.interp(new,t,q[:,i]) for i in range(q.shape[1])],axis=1)
    for start in (3,39):
        rotations=Slerp(t,Rotation.from_quat(q[:,start:start+4][:,[1,2,3,0]]))(new).as_quat()
        result[:,start:start+4]=rotations[:,[3,0,1,2]]
    data['qpos']=result;data['fps']=np.asarray(50)
    if 'human_joints' in data:
        original=data['human_joints'];flat=original.reshape(len(q),-1)
        data['human_joints']=np.stack([np.interp(new,t,flat[:,i]) for i in range(flat.shape[1])],axis=1).reshape(len(new),*original.shape[1:])
    data['retargeting_backend']=np.asarray('holosoma_object_interaction')
    data['quaternion_order']=np.asarray('wxyz')
    with target.open('wb') as f:np.savez_compressed(f,**data)
    verify_motion(target)


def sync_subset(folder,destination):
    if destination.exists():
        old=read(destination/'status.json')
        current=read(folder/'status.json')
        if old.get('fingerprint')!=current.get('fingerprint'):raise ValueError(f'Selected output differs: {destination}. Use a fresh --selected-out.')
    else:destination.mkdir(parents=True)
    for name in ('selection.json','status.json','retarget.log','motion_original.npz','motion_fps50.npz'):
        source=folder/name;target=destination/name
        if not source.is_file():raise FileNotFoundError(source)
        if target.exists() and digest(target)!=digest(source):raise ValueError(f'Existing selected copy differs: {target}')
        if not target.exists():shutil.copy2(source,target)
    return digest(destination/'motion_original.npz')


def run_child(command,cwd,log):
    with log.open('w') as f:
        child=subprocess.Popen(command,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:return child.wait()
        except KeyboardInterrupt:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    home=Path.home()/'holosoma'
    parser.add_argument('--manifest',type=Path,default=home/'t1/run05/manifest.json')
    parser.add_argument('--selected-manifest',type=Path,default=home/'t2/hoi_all_lift_candidates01/passed_lift_manifest.json')
    parser.add_argument('--package',type=Path,default=home/'src/holosoma_retargeting/holosoma_retargeting')
    parser.add_argument('--object-assets',type=Path,default=Path.home()/'hoi-retarget/assets/objects')
    parser.add_argument('--all-out',type=Path,default=home/'t2/holosoma_lift_all01')
    parser.add_argument('--selected-out',type=Path,default=home/'t2/holosoma_lift_selected24_01')
    parser.add_argument('--include-rejected',action='store_true',help='Also retarget explicitly rejected T1 lifts; default includes pending and accepted only.')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.expanduser().resolve())
    if args.all_out==args.selected_out or args.all_out.is_relative_to(args.selected_out) or args.selected_out.is_relative_to(args.all_out):parser.error('All and selected output folders must be separate siblings, not nested.')
    clips,picked=selections(read(args.manifest),read(args.selected_manifest),args.include_rejected)
    for c in clips:
        if not Path(c['source']).is_file():raise FileNotFoundError(c['source'])
    entry=args.package/'examples/robot_retarget.py'
    if not entry.is_file():raise FileNotFoundError(entry)
    print('Holosoma lift clips:',len(clips),'objects:',dict(Counter(c['object'] for c in clips)),flush=True)
    print('Separate subset:',len(picked),'same IDs as the HOI selected 24.',flush=True)
    print('Task: object_interaction | robot: g1 | format: smplh | raw and 50 Hz playback qpos NPZs',flush=True)
    if args.dry_run:return
    # Fail before expensive work if required installed environment/dependencies are absent.
    import numpy as np
    import torch
    import mujoco
    import scipy
    help_result=subprocess.run([sys.executable,str(entry),'--help'],cwd=args.package,capture_output=True,text=True)
    if help_result.returncode:raise RuntimeError('Holosoma CLI help failed. Activate hsretargeting and check installed dependencies.\n'+help_result.stderr[-2000:])
    flags=['--retargeter.activate-obj-non-penetration','--retargeter.activate-joint-limits','--retargeter.activate-foot-sticking']
    for flag in flags:
        if flag not in re.sub(r'\x1b\[[0-9;]*m','',help_result.stdout):raise ValueError(f'Installed Holosoma help lacks {flag}; inspect its CLI before running.')
    for output in (args.all_out,args.selected_out):
        if output.exists() and not (output/'settings.json').is_file() and any(output.iterdir()):
            raise ValueError(f'Untracked nonempty output folder: {output}. Use a fresh folder.')
        if output.is_relative_to(args.package/'models'):
            raise ValueError('Outputs must be outside the installed model directory.')
    args.all_out.mkdir(parents=True,exist_ok=True)
    lock=(args.all_out/'.batch.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another Holosoma lift batch is using this output.')
    settings=dict(backend='holosoma',task_type='object_interaction',robot='g1',data_format='smplh',
                  manifest_sha256=digest(args.manifest),selected_sha256=digest(args.selected_manifest),runner_sha256=digest(__file__),
                  entry_sha256=digest(entry),all_out=str(args.all_out),selected_out=str(args.selected_out),
                  include_rejected=args.include_rejected,flags=flags)
    settings_path=args.all_out/'settings.json'
    if settings_path.exists() and read(settings_path)!=settings:raise ValueError('Run settings changed. Use fresh output directories to protect prior results.')
    json_save(settings_path,settings)
    runtime=prepare_runtime(args.package,args.all_out,{c['object'] for c in clips},args.object_assets)
    # Validate every combined model before starting a long batch.
    for obj in {c['object'] for c in clips}:
        model=mujoco.MjModel.from_xml_path(str(runtime/'models/g1'/f'g1_29dof_w_{obj}.xml'))
        if model.nq!=43:raise ValueError(f'{obj}: expected nq=43, got {model.nq}.')
    args.selected_out.mkdir(parents=True,exist_ok=True)
    subset_settings=args.selected_out/'settings.json'
    if subset_settings.exists() and read(subset_settings)!=settings:raise ValueError('Selected folder belongs to a different run; use a fresh --selected-out.')
    json_save(subset_settings,settings)
    records=[];counts=Counter();last_path=None;tensor=None
    def summaries():
        json_save(args.all_out/'manifest.json',dict(settings=settings,clips=records,counts=dict(counts),created_at=stamp(),note='All attempted T1 lifts. Successful retargeting is not grasp approval.'))
        selected_rows=[r for r in records if r['id'] in picked]
        json_save(args.selected_out/'manifest.json',dict(settings=settings,expected_ids=sorted(picked),clips=selected_rows,
                    completed=sum(r['status']=='completed' for r in selected_rows),expected=24,
                    note='Independent copies of the matching Holosoma results, not HOI motions.'))
    for index,c in enumerate(clips,1):
        cid=c['id'];folder=args.all_out/'lift'/cid;folder.mkdir(parents=True,exist_ok=True)
        record=dict(c,status='pending',backend='holosoma',selected24=cid in picked)
        try:
            source=Path(c['source']).resolve();fingerprint=dict(source_sha256=digest(source),start=int(c['start']),end=int(c['end']),object=c['object'],settings=settings)
            state=folder/'status.json';original=folder/'motion_original.npz';fps50=folder/'motion_fps50.npz'
            old=read(state) if state.is_file() else None
            if old and old.get('fingerprint')!=fingerprint:raise ValueError('Clip/source/settings changed; use fresh output directories.')
            if old and old.get('status')=='completed' and original.is_file() and fps50.is_file() and old.get('output_sha256')==digest(original) and old.get('fps50_sha256')==digest(fps50):
                verify_motion(original,int(c['end'])-int(c['start']));verify_motion(fps50)
                record.update(old);counts['resumed']+=1
                print(f'[{index}/{len(clips)}] RESUME {cid}',flush=True)
            else:
                if not old and any(folder.glob('*.npz')):raise ValueError('Untracked output exists; use fresh output directories.')
                record.update(status='running',fingerprint=fingerprint,started_at=stamp());json_save(state,record);json_save(folder/'selection.json',c)
                if source!=last_path:
                    try:tensor=torch.load(source,map_location='cpu',weights_only=True)
                    except TypeError:tensor=torch.load(source,map_location='cpu')
                    last_path=source
                if not isinstance(tensor,torch.Tensor) or tensor.ndim!=2 or tensor.shape[1]!=591:raise ValueError('Expected a (frames,591) InterMimic tensor.')
                start,end=int(c['start']),int(c['end'])
                if end>len(tensor) or end-start<2:raise ValueError('Source crop is invalid or too short.')
                input_dir=folder/'input';input_dir.mkdir(exist_ok=True)
                staged=input_dir/source.name
                torch.save(tensor[start:end].detach().cpu().clone(),staged)
                result_dir=folder/'retargeted';result_dir.mkdir(exist_ok=True)
                command=[sys.executable,str(entry),'--task-type','object_interaction','--robot','g1','--data-format','smplh',
                         '--task-name',source.stem,'--data-path',str(input_dir),'--task-config.object-name',c['object'],'--save-dir',str(result_dir),*flags]
                record['command']=command;json_save(state,record)
                print(f'[{index}/{len(clips)}] RUN {cid}',flush=True)
                code=run_child(command,runtime,folder/'retarget.log')
                if code:raise RuntimeError(f'Holosoma exited {code}; see {folder/"retarget.log"}')
                generated=result_dir/(source.stem+'_original.npz')
                shape=verify_motion(generated,end-start)
                shutil.copy2(generated,original);resample50(original,fps50)
                record.update(status='completed',finished_at=stamp(),output_sha256=digest(original),fps50_sha256=digest(fps50),output=shape)
                json_save(state,record);counts['completed']+=1
            if cid in picked:record['selected_copy_sha256']=sync_subset(folder,args.selected_out/'lift'/cid)
        except KeyboardInterrupt:
            record.update(status='interrupted',finished_at=stamp());json_save(folder/'status.json',record);records.append(record);summaries();raise
        except Exception as exc:
            record.update(status='failed',error=f'{type(exc).__name__}: {exc}',finished_at=stamp());counts['failed']+=1
            if not (folder/'status.json').is_file() or read(folder/'status.json').get('status')!='completed':json_save(folder/'status.json',record)
            print(f'[{index}/{len(clips)}] FAIL {cid}: {record["error"]}',flush=True)
        records.append(record);summaries()
    # Separate usable model snapshots make both result directories self-contained on this laptop.
    subset_models=args.selected_out/'runtime/models'
    if not subset_models.exists():
        subset_models.parent.mkdir(parents=True,exist_ok=True)
        temp_models=Path(tempfile.mkdtemp(prefix='.models_',dir=subset_models.parent))
        try:
            shutil.copytree(runtime/'models',temp_models,dirs_exist_ok=True)
            # Rewrite only private absolute runtime paths to the independent selected runtime.
            for p in (temp_models/'g1').glob('g1_29dof_w_*.xml'):
                p.write_text(p.read_text().replace(str(runtime/'models'),str(subset_models)))
            temp_models.rename(subset_models)
        except BaseException:
            shutil.rmtree(temp_models,ignore_errors=True);raise
    print('Finished:',dict(counts),flush=True)
    print('All lifts:',args.all_out,flush=True)
    print('Matching 24:',args.selected_out,'completed',sum(r['status']=='completed' and r['id'] in picked for r in records),'/24',flush=True)
    if counts['failed']:raise SystemExit(1)


if __name__=='__main__':main()
