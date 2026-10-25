#!/usr/bin/env python3
"""Paused IsaacSim reference/collider viewer adapted to Rudra's replay sources.

This is kinematic inspection: NO force-based grasp success is inferred.
No repository, reference, mass, friction, or collider settings are edited.
One environment, a matching already-scaled object URDF, and the installed robot
asset are used. Commands/configuration are built through verified installed
dataclasses, not guessed command-line flags. Run in hssim from ~/holosoma.
"""
import argparse
import dataclasses
import json
import hashlib
from pathlib import Path
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET


def prepare_usd_safe_urdf(source, destination):
    """Copy assets unchanged, giving imported mesh prims valid USD names."""
    source=Path(source).resolve();destination=Path(destination).resolve()
    tree=ET.parse(source)
    meshes=tree.findall('.//mesh')
    resolved=[]
    for mesh in meshes:
        name=mesh.get('filename')
        if not name:raise ValueError('URDF mesh is missing its filename.')
        if '://' in name:raise ValueError(f'Expected a local mesh filename, got {name}')
        asset=(source.parent/name).resolve()
        if not asset.is_file():raise FileNotFoundError(asset)
        if destination.is_relative_to(asset.parent):
            raise ValueError('Debug asset folder must be outside the source mesh directory.')
        resolved.append((mesh,asset))
    destination.mkdir(parents=True,exist_ok=False)
    directories={};mapping=[]
    for mesh,asset in resolved:
        if asset.parent not in directories:
            folder=destination/('assets_'+hashlib.sha256(str(asset.parent).encode()).hexdigest()[:12])
            # Keep OBJ material/texture paths alongside the mesh unchanged.
            shutil.copytree(asset.parent,folder)
            directories[asset.parent]=folder
        folder=directories[asset.parent]
        stem=re.sub(r'[^A-Za-z0-9_]', '_',asset.stem)
        alias=folder/('mesh_'+hashlib.sha256(str(asset).encode()).hexdigest()[:12]+'_'+stem+asset.suffix)
        shutil.copy2(asset,alias)
        old=mesh.get('filename');mesh.set('filename',str(alias.relative_to(destination)))
        digest=hashlib.sha256(asset.read_bytes()).hexdigest()
        if hashlib.sha256(alias.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Debug mesh copy differs from source.')
        mapping.append(dict(original=old,source=str(asset),alias=mesh.get('filename'),sha256=digest))
    # Resolve any explicit URDF texture paths relative to their original file.
    for texture in tree.findall('.//texture'):
        name=texture.get('filename')
        if name and '://' not in name:texture.set('filename',str((source.parent/name).resolve()))
    output=destination/'object_debug.urdf'
    tree.write(output,encoding='utf-8',xml_declaration=True)
    (destination/'mesh_aliases.json').write_text(json.dumps(mapping,indent=2)+'\n')
    return output


def fields(value):
    if not dataclasses.is_dataclass(value): raise TypeError(f'Expected dataclass, got {type(value).__name__}')
    return {f.name for f in dataclasses.fields(value)}


def replace_path(value,path,replacement):
    if not path:return replacement
    key,*rest=path
    if isinstance(value,dict):
        if key not in value: raise KeyError(f'Missing config dictionary key {key}; available={list(value)}')
        result=dict(value);result[key]=replace_path(value[key],rest,replacement);return result
    if key not in fields(value): raise KeyError(f'Missing config field {key}; available={sorted(fields(value))}')
    return dataclasses.replace(value,**{key:replace_path(getattr(value,key),rest,replacement)})


def configure(base,npz_path,urdf):
    training=dataclasses.replace(base.training,num_envs=1,headless=False)
    config=dataclasses.replace(base,training=training)
    config=replace_path(config,['command','setup_terms','motion_command','params','motion_config','motion_file'],str(npz_path))
    scene=config.scene
    objects=scene.rigid_objects
    if scene.scene_files or len(objects)!=1:
        raise ValueError(f'Debug preset must contain one standalone object and no scene files. objects={list(objects)}')
    key,obj=next(iter(objects.items()))
    names=fields(obj)
    urdf_fields=[n for n in names if 'urdf' in n.lower()]
    usd_fields=[n for n in names if 'usd' in n.lower()]
    if len(urdf_fields)!=1 or len(usd_fields)!=1:
        raise ValueError(f'Cannot uniquely identify URDF/USD fields. Object config fields: {sorted(names)}')
    # Uploaded object_spawner chooses USD before URDF. Clear USD to ensure the
    # reference's scaled URDF is actually used, keeping the command's object key.
    changes={urdf_fields[0]:str(urdf),usd_fields[0]:None,'fixed':False}
    if 'scale' in names:
        scale=getattr(obj,'scale')
        if scale is not None:
            if isinstance(scale,(int,float)): changes['scale']=1.
            elif isinstance(scale,(list,tuple)):changes['scale']=type(scale)([1.,1.,1.])
            else:raise ValueError('Unrecognized object scale config; inspect before viewing.')
    obj=dataclasses.replace(obj,**changes)
    config=dataclasses.replace(config,scene=dataclasses.replace(scene,rigid_objects={key:obj}))
    return config,key


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.home()/'holosoma/t2/lift24_npz_contactqa')
    parser.add_argument('--clip',default='sub12_plasticbox_036_t1_lift_000242_000280')
    parser.add_argument('--out',type=Path,default=Path.home()/'holosoma/t2/lift24_contact_calibration/isaac_view01')
    parser.add_argument('--frame',type=int)
    parser.add_argument('--inspect-config',action='store_true',help='Print resolved configuration without starting IsaacSim.')
    args=parser.parse_args();args.root=args.root.expanduser().resolve();args.out=args.out.expanduser().resolve()
    manifest=json.loads((args.root/'manifest.json').read_text())
    matches=[c for c in manifest['clips'] if c['id']==args.clip and c['task']=='lift']
    if len(matches)!=1:parser.error('Select a unique lift clip from the NPZ manifest.')
    clip=matches[0]
    npz_path=args.root/clip['npz'];urdf=args.root/clip['object_urdf']
    if not npz_path.is_file() or not urdf.is_file():parser.error('Missing NPZ or matching object URDF.')
    from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_w_object
    config,object_key=configure(g1_29dof_wbt_w_object,npz_path,urdf)
    print('Clip:',args.clip,'\nObject scene key:',object_key,'\nScaled URDF:',urdf)
    print('Object fixed=False keeps it in the Holosoma motion-object registry. This viewer writes poses and renders without physics stepping; do not use it as a grasp test.')
    if args.inspect_config:
        print(config);return
    if args.out.exists():parser.error('Debug output exists. Use a fresh --out to preserve snapshots.')
    args.out.mkdir(parents=True)
    urdf=prepare_usd_safe_urdf(urdf,args.out/'usd_safe_assets')
    config,object_key=configure(g1_29dof_wbt_w_object,npz_path,urdf)
    print('USD-safe debug URDF:',urdf,flush=True)
    # Follow the uploaded replay.py startup/import ordering. Heavy simulator
    # imports happen after init_sim_imports starts the app.
    from holosoma.utils.eval_utils import init_sim_imports
    from holosoma.utils.sim_utils import close_simulation_app
    # Keep our argparse options away from Isaac's own launch argument parser.
    sys.argv=[sys.argv[0]]
    app=init_sim_imports(config)
    try:
        import numpy as np
        import torch
        import omni.ui as ui
        from holosoma.config_types.env import get_tyro_env_config
        from holosoma.utils.helpers import get_class
        from isaac_palm_contact_tools import dump_stage_colliders
        env=get_class(config.env_class)(get_tyro_env_config(config),device='cuda:0' if torch.cuda.is_available() else 'cpu')
        sim=env.simulator
        robot=sim.scene.articulations['robot']
        box=sim.scene.rigid_objects[object_key]
        with np.load(npz_path,allow_pickle=False) as archive:
            data={k:archive[k] for k in archive.files}
        if str(data['quaternion_order'].item())!='wxyz':raise ValueError('Expected explicit WXYZ reference.')
        joint_names=[str(x) for x in data['joint_names'].tolist()]
        if len(joint_names)!=29 or len(set(joint_names))!=29:raise ValueError('Expected 29 unique G1 joints.')
        ids=[robot.joint_names.index(name) for name in joint_names]
        if data['joint_pos'].shape[1]!=36:raise ValueError('Unexpected reference qpos layout.')
        n=len(data['joint_pos']);fps=float(data['fps'].item())
        frame=args.frame if args.frame is not None else n//2
        if not 0<=frame<n:raise ValueError('Frame out of range.')
        device=robot.data.joint_pos.device
        root=torch.as_tensor(data['joint_pos'][:,:7],device=device,dtype=torch.float32)
        joints=torch.as_tensor(data['joint_pos'][:,7:],device=device,dtype=torch.float32)
        object_poses=torch.as_tensor(np.concatenate([data['object_pos_w'],data['object_quat_w']],axis=-1),device=device,dtype=torch.float32)
        zero_root=torch.zeros((1,6),device=device);zero_joint=torch.zeros((1,29),device=device)
        if not hasattr(sim.sim,'forward'):raise RuntimeError('Installed SimulationContext lacks forward(); inspect API before adapting.')
        # Per-environment origins are added only to translated XYZ.
        origin=sim.scene.env_origins[:1]
        def set_frame(index):
            rp=root[index:index+1].clone();rp[:,:3]+=origin
            bp=object_poses[index:index+1].clone();bp[:,:3]+=origin
            robot.write_root_pose_to_sim(rp)
            robot.write_root_velocity_to_sim(zero_root)
            robot.write_joint_state_to_sim(joints[index:index+1],zero_joint,joint_ids=ids)
            box.write_root_pose_to_sim(bp);box.write_root_velocity_to_sim(zero_root)
            sim.sim.forward()
            sim.scene.update(dt=1/fps)
        set_frame(frame)
        dump_stage_colliders(args.out/'isaac_stage_colliders.json')
        # Record the live asset config too; an MJCF inventory alone cannot
        # establish that the Isaac URDF/USD uses the same collision shapes.
        (args.out/'resolved_assets.txt').write_text('robot='+repr(config.robot.asset)+'\nobject='+repr(config.scene.rigid_objects[object_key])+'\n')
        def snapshot():
            i=frame_model.get_value_as_int()
            set_frame(i)
            bodies=robot.body_names
            bp=robot.data.body_pos_w[0].detach().cpu().tolist()
            bq=robot.data.body_quat_w[0].detach().cpu().tolist()
            payload=dict(clip=args.clip,frame=i,fps=fps,mode='kinematic_geometry_only',
                quaternion_order='wxyz',body_names=bodies,body_pos_w=bp,body_quat_w=bq,
                object_pos_w=box.data.root_pos_w[0].detach().cpu().tolist(),
                object_quat_w=box.data.root_quat_w[0].detach().cpu().tolist(),
                env_origin=origin[0].detach().cpu().tolist(),npz=str(npz_path),object_urdf=str(urdf))
            (args.out/f'pose_frame_{i:05d}.json').write_text(json.dumps(payload,indent=2)+'\n')
            dump_stage_colliders(args.out/'isaac_stage_colliders.json')
            print('Saved geometry snapshot, frame',i,'to',args.out,flush=True)
        window=ui.Window('Lift contact QA: geometry only',width=460,height=170)
        frame_model=ui.SimpleIntModel(frame);play_model=ui.SimpleBoolModel(False)
        with window.frame:
            with ui.VStack():
                ui.Label('Paused by default. Inspect collider outlines, not force readings.')
                ui.IntSlider(model=frame_model,min=0,max=n-1)
                with ui.HStack():
                    ui.Label('Play / loop');ui.CheckBox(model=play_model)
                ui.Button('Save current pose and collider inventory',clicked_fn=snapshot)
        center=object_poses[frame,:3].detach().cpu().numpy()+origin[0].detach().cpu().numpy()
        sim.sim.set_camera_view((center+np.array([1.8,-1.8,1.2])).tolist(),center.tolist())
        print('GUI ready. Enable viewport Physics > Colliders and inspect from front, side and top.',flush=True)
        previous=time.monotonic()
        while app.is_running():
            now=time.monotonic()
            if play_model.get_value_as_bool() and now-previous>=1/fps:
                frame_model.set_value((frame_model.get_value_as_int()+1)%n);previous=now
            index=frame_model.get_value_as_int()
            set_frame(index)
            sim.sim.render()
            time.sleep(.005)
    finally:
        close_simulation_app(app)


if __name__=='__main__':main()
