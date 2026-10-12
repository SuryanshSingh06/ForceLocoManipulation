#!/usr/bin/env python3
"""Train a new PPO lift policy from random weights with a three-second stationary hold.
Creates a separate run. Training completion is not proof of task success.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone

COMMIT='d18d6cc50f872c15e904a22ceac22313cec955c8'
CLIP='sub7_largebox_024_t1_lift_000009_000066'



def settling_cost(env, kind, settling_start, ramp_frames=20):
    """Dense, smooth error cost active near the end of the lift and during hold.

    World position targets come from the motion plus environment origins,
    never the robot-relative reference that can follow a drifting robot.
    Cost has no box-position gate. Velocity targets follow the original
    reference during settling, then become zero in the appended hold.
    """
    import torch
    from holosoma.utils.rotations import quat_rotate_inverse
    command=env.command_manager.get_state('motion_command')
    phase=((command.time_steps.float()-settling_start+1)/ramp_frames).clamp(0.,1.)
    if kind=='tilt':
        up=torch.zeros_like(command.robot_root_pos_w);up[:,2]=1.
        actual=quat_rotate_inverse(command.robot_root_quat_w,up,w_last=True)
        target=quat_rotate_inverse(command.root_quat_w,up,w_last=True)
        torso_actual=quat_rotate_inverse(command.robot_ref_quat_w,up,w_last=True)
        torso_target=quat_rotate_inverse(command.ref_quat_w,up,w_last=True)
        error=((actual-target).square().sum(-1)+
               (torso_actual-torso_target).square().sum(-1))/(2.*0.25**2)
    elif kind=='robot_velocity':
        error=(command.robot_root_lin_vel_w-command.root_lin_vel_w).square().sum(-1)/(0.2**2)
        error+=(command.robot_root_ang_vel_w-command.root_ang_vel_w).square().sum(-1)/(0.8**2)
    elif kind=='feet_position':
        if not hasattr(command,'_fresh_foot_indices'):
            names=command.motion_cfg.body_names_to_track
            command._fresh_foot_indices=[names.index(side+'_ankle_roll_link') for side in ('left','right')]
        indices=command._fresh_foot_indices
        target=command.body_pos_w[:,indices]  # Holosoma already adds environment origins.
        error=(command.robot_body_pos_w[:,indices]-target).square().sum(-1).mean(-1)/(0.1**2)
    elif kind=='robot_position':
        error=(command.robot_root_pos_w[:,:2]-command.root_pos_w[:,:2]).square().sum(-1)/(0.1**2)
    elif kind=='box_velocity':
        state=command._simulator_object_states()
        error=(state[:,7:10]-command.object_lin_vel_w).square().sum(-1)/(0.2**2)
        # This Holosoma revision does not load reference object angular velocity.
        # Penalize spin only after the stationary hold has begun.
        holding=(command.time_steps>=settling_start+ramp_frames).float()
        error+=holding*state[:,10:13].square().sum(-1)/(0.8**2)
    elif kind=='box_position':
        error=(command.simulator_object_pos_w-command.object_pos_w).square().sum(-1)/(0.15**2)
    else:
        raise ValueError('Unknown settling cost: '+kind)
    # Pseudo-Huber form: quadratic near zero, approximately linear at large errors.
    # Unlike exp(-error), it continues distinguishing moderately bad states.
    return phase*(torch.sqrt(1.+error)-1.)


def supported_hold(env, hold_start, collision_bounds, ramp_frames=10):
    """Reward a raised, near-target box while the robot and box settle.

    Height saturates at 15 cm bottom clearance. Extra height earns nothing.
    The conservative rotated mesh AABB bottom prevents center-height shortcuts.
    Wrist-to-box geometry is a proximity proxy, not a contact-force measurement.
    """
    import torch
    from holosoma.utils.rotations import quat_rotate_inverse
    c=env.command_manager.get_state('motion_command')
    state=c._simulator_object_states()
    phase=((c.time_steps.float()-hold_start+1)/ramp_frames).clamp(0.,1.)
    if not hasattr(env,'_fresh_box_bounds'):
        env._fresh_box_bounds=torch.tensor(collision_bounds,device=state.device,dtype=state.dtype)
    lo,hi=env._fresh_box_bounds
    center=(lo+hi)/2.;half=(hi-lo)/2.
    q=state[:,3:7];x,y,z,w=q.unbind(-1)
    row=torch.stack((2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),dim=-1)
    bottom=state[:,2]+(row*center).sum(-1)-(row.abs()*half).sum(-1)-env.simulator.scene.env_origins[:,2]
    airborne=(bottom/0.15).clamp(0.,1.)
    up=torch.zeros_like(c.robot_root_pos_w);up[:,2]=1.
    tilt=(quat_rotate_inverse(c.robot_root_quat_w,up,w_last=True)-
          quat_rotate_inverse(c.root_quat_w,up,w_last=True)).square().sum(-1)/(0.35**2)
    torso_tilt=(quat_rotate_inverse(c.robot_ref_quat_w,up,w_last=True)-
                quat_rotate_inverse(c.ref_quat_w,up,w_last=True)).square().sum(-1)/(0.35**2)
    tilt=(tilt+torso_tilt)/2.
    position=(state[:,:3]-c.object_pos_w).square().sum(-1)/(0.3**2)
    speed=state[:,7:10].square().sum(-1)/(0.4**2)
    speed+=c.robot_root_lin_vel_w.square().sum(-1)/(0.3**2)
    speed+=state[:,10:13].square().sum(-1)/(1.0**2)
    drift=(c.robot_root_pos_w[:,:2]-c.root_pos_w[:,:2]).square().sum(-1)/(0.15**2)
    if not hasattr(c,'_fresh_hand_indices'):
        names=c.motion_cfg.body_names_to_track
        c._fresh_hand_indices=[names.index(side+'_wrist_yaw_link') for side in ('left','right')]
    hands=c._fresh_hand_indices
    actual=c.robot_body_pos_w[:,hands]-state[:,None,:3]
    target=c.body_pos_w[:,hands]-c.object_pos_w[:,None,:]
    hand_error=(actual-target).square().sum(-1).mean(-1)/(0.15**2)
    return phase*airborne/(1.+tilt+position+speed+drift+hand_error)


def check_reward_math(device):
    """Fast real-PyTorch regression checks after IsaacSim initializes.

    These check reward implementation, not physical grasp or balance success.
    """
    import torch
    from types import SimpleNamespace
    def tensor(x):return torch.tensor(x,dtype=torch.float32,device=device)
    origins=tensor([[0.,0.,0.],[3.,2.,0.]])
    positions=origins+tensor([[0.,0.,1.],[0.,0.,1.]])
    zeros=torch.zeros_like(positions)
    quats=tensor([[0.,0.,0.,1.],[0.,0.,0.,1.]])
    bodies=origins[:,None,:].repeat(1,4,1)
    c=SimpleNamespace(
        time_steps=tensor([101.,101.]),robot_root_pos_w=positions.clone(),root_pos_w=positions.clone(),
        robot_root_quat_w=quats.clone(),root_quat_w=quats.clone(),
        robot_ref_quat_w=quats.clone(),ref_quat_w=quats.clone(),
        robot_root_lin_vel_w=zeros.clone(),root_lin_vel_w=zeros.clone(),
        robot_root_ang_vel_w=zeros.clone(),root_ang_vel_w=zeros.clone(),
        object_lin_vel_w=zeros.clone(),object_pos_w=positions.clone(),
        simulator_object_pos_w=positions.clone(),body_pos_w=bodies.clone(),robot_body_pos_w=bodies.clone(),
        motion_cfg=SimpleNamespace(body_names_to_track=[
            'left_ankle_roll_link','right_ankle_roll_link','left_wrist_yaw_link','right_wrist_yaw_link']))
    state=torch.zeros((2,13),device=device);state[:,:3]=positions;state[:,6]=1.
    c._simulator_object_states=lambda:state
    env=SimpleNamespace(command_manager=SimpleNamespace(get_state=lambda _:c),
                        simulator=SimpleNamespace(scene=SimpleNamespace(env_origins=origins)))
    for kind in ('tilt','robot_velocity','feet_position','robot_position','box_velocity','box_position'):
        value=settling_cost(env,kind,72)
        if not torch.allclose(value,torch.zeros_like(value),atol=1e-6):
            raise RuntimeError('Reward regression: nonzero exact-tracking cost for '+kind)
    bounds=[[-.1,-.1,-.1],[.1,.1,.1]]
    if not torch.allclose(supported_hold(env,92,bounds),torch.ones(2,device=device),atol=1e-6):
        raise RuntimeError('Reward regression: stationary hold or environment translation')
    state[0,2]=.1
    if supported_hold(env,92,bounds)[0].item()>1e-6:
        raise RuntimeError('Reward regression: grounded box receives airborne bonus')
    state[0,2]=1.;state[1,7]=1.
    if supported_hold(env,92,bounds)[1].item()>=.2:
        raise RuntimeError('Reward regression: fast box movement is not discouraged')
    state[1,7]=0.
    # 35-degree torso pitch while pelvis remains at its reference orientation.
    import math
    c.robot_ref_quat_w[0]=tensor([0.,math.sin(math.radians(17.5)),0.,math.cos(math.radians(17.5))])
    if settling_cost(env,'tilt',72)[0].item()<=.9:
        raise RuntimeError('Reward regression: torso-only lean is not discouraged')
    c.time_steps[:]=20
    if torch.count_nonzero(settling_cost(env,'tilt',72)).item()!=0:
        raise RuntimeError('Reward regression: settling penalty active during pickup')
    return {'passed':True,'checks':['exact tracking in translated environments','ground clearance',
            'moving box','torso-only lean','pickup phase gating'],'physical_success_verified':False}


def replay(args):
    from holosoma.utils.eval_utils import CheckpointConfig,load_saved_experiment_config
    from holosoma.eval_agent import run_eval_with_tyro
    checkpoint_cfg=CheckpointConfig(checkpoint=str(args.checkpoint))
    saved,wandb=load_saved_experiment_config(checkpoint_cfg)
    cfg=saved.get_eval_config()
    cfg=replace(cfg,training=replace(cfg.training,num_envs=1,headless=False,
                max_eval_steps=1000,export_onnx=False))
    run_eval_with_tyro(cfg,checkpoint_cfg,saved,wandb)

def write_json(path,value):
    path.write_text(json.dumps(value,indent=2))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare(data,out):
    import numpy as np
    manifest=json.loads((data/'manifest.json').read_text())
    found=[c for c in manifest['clips'] if c['id']==CLIP and c['task']=='lift']
    if len(found)!=1:raise ValueError('Retained pilot clip missing or duplicated in export manifest')
    clip=found[0]
    if clip.get('t2_review')!='keep':raise ValueError('Pilot clip is not marked keep')
    motion=(data/clip['npz']).resolve(); source_urdf=(data/clip['object_urdf']).resolve()
    with np.load(motion,allow_pickle=False) as d:
        assert d['fps'].item()==50 and d['quaternion_order'].item()=='wxyz'
        assert d['joint_pos'].shape[1]==36 and d['joint_vel'].shape[1]==35
        for k in ('joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w',
                  'object_pos_w','object_quat_w','object_lin_vel_w','object_ang_vel_w'):
            if not np.isfinite(d[k]).all():raise ValueError('Nonfinite required state: '+k)
        initial_pos=d['object_pos_w'][0].tolist();initial_quat=d['object_quat_w'][0].tolist()
    tree=ET.parse(source_urdf);root=tree.getroot();links=root.findall('link')
    if len(links)!=1:raise ValueError('Pilot expects a single-link object')
    collision=links[0].find('collision');mesh=collision.find('geometry/mesh')
    origin=collision.find('origin')
    if origin is not None and (any(float(x)!=0 for x in origin.get('rpy','0 0 0').split()) or any(float(x)!=0 for x in origin.get('xyz','0 0 0').split())):
        raise ValueError('Nonzero mesh origin requires explicit mass-frame handling')
    asset=(source_urdf.parent/mesh.attrib['filename']).resolve()
    vertices=np.array([[float(x) for x in line.split()[1:4]] for line in asset.read_text().splitlines() if line.startswith('v ')])
    vertices*=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
    lo=vertices.min(0);hi=vertices.max(0);size=hi-lo;center=(hi+lo)/2
    if np.any(size<=0):raise ValueError('Degenerate object bounds')
    # Explicit approximation: uniform solid cuboid mass distribution over mesh bounds.
    # Collision and visual meshes are retained unchanged, including their authored scale.
    mass=1.0;x,y,z=size;inertia=mass/12*np.array([y*y+z*z,x*x+z*z,x*x+y*y])
    for element in root.findall('.//mesh'):
        element.set('filename',str((source_urdf.parent/element.attrib['filename']).resolve()))
    old=links[0].find('inertial')
    if old is not None:links[0].remove(old)
    inertial=ET.SubElement(links[0],'inertial')
    ET.SubElement(inertial,'origin',xyz=' '.join(map(str,center)),rpy='0 0 0')
    ET.SubElement(inertial,'mass',value=str(mass))
    ET.SubElement(inertial,'inertia',ixx=str(inertia[0]),iyy=str(inertia[1]),izz=str(inertia[2]),ixy='0',ixz='0',iyz='0')
    generated=out/'largebox_1kg.urdf';tree.write(generated,encoding='utf-8',xml_declaration=True)
    result=dict(clip_id=CLIP,motion=str(motion),object_urdf=str(generated),initial_pos=initial_pos,
                initial_quat=initial_quat,mass_kg=mass,inertia_diagonal_kg_m2=inertia.tolist(),
                center_of_mass_m=center.tolist(),mesh_bounds_size_m=size.tolist(),
                collision_bounds=[lo.tolist(),hi.tolist()],
                inertia_model='uniform solid cuboid approximation over scaled mesh bounds',
                static_friction=.9,dynamic_friction=.9,restitution=0.,
                source_hashes={'motion':digest(motion),'urdf':digest(source_urdf),'mesh':digest(asset)},
                limitations=['not a lift-success evaluation','no terminal hold extension',
                'stock clip-end reset behavior retained for pilot training only','geometry and hand alignment require review'])
    write_json(out/'pilot_inputs.json',result)
    return result

def append_terminal_hold(inputs,out):
    """Create a separate training reference; preserve every original export."""
    import numpy as np
    source=Path(inputs['motion'])
    with np.load(source,allow_pickle=False) as archive:
        arrays={key:archive[key].copy() for key in archive.files}
    n=len(arrays['joint_pos']);fps=float(arrays['fps'].item())
    hold_frames=int(round(3.0*fps))
    velocity_keys={'joint_vel','body_lin_vel_w','body_ang_vel_w',
                   'object_lin_vel_w','object_ang_vel_w'}
    time_keys=('joint_pos','joint_vel','body_pos_w','body_quat_w',
               'body_lin_vel_w','body_ang_vel_w','object_pos_w','object_quat_w',
               'object_lin_vel_w','object_ang_vel_w','qpos',
               'object_contact_sequence','object_ground_contact_sequence',
               'robot_foot_ground_contact_flags','per_link_contact_flags',
               'fixed_contact_points_per_frame_in_object_frame')
    for key in time_keys:
        if key not in arrays:continue
        value=arrays[key]
        if value.shape[0]!=n:raise ValueError('Reference length mismatch: '+key)
        tail=np.repeat(value[-1:],hold_frames,axis=0)
        if key in velocity_keys:tail[:]=0
        arrays[key]=np.concatenate((value,tail),axis=0)
    target=out/'lift_with_terminal_hold.npz'
    np.savez_compressed(target,**arrays)
    updated=dict(inputs,motion=str(target),original_motion=str(source),
                 terminal_hold_seconds=3.0,original_frames=n,derived_frames=n+hold_frames,
                 derived_motion_sha256=digest(target))
    updated['limitations']=[x for x in inputs['limitations'] if x!='no terminal hold extension']
    updated['limitations'].append('terminal pose repetition is a training target; physical hold still requires evaluation')
    write_json(out/'pilot_inputs.json',updated)
    write_json(out/'reference_derivation.json',dict(source_motion=str(source),source_sha256=digest(source),
        derived_motion=str(target),derived_sha256=digest(target),original_frames=n,
        appended_frames=hold_frames,fps=fps,method='repeat terminal pose and contacts; zero appended velocities'))
    return updated

def build_config(inputs,out,args):
    from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_w_object as base
    from holosoma.config_types.logger import DisabledLoggerConfig
    from holosoma.config_types.reward import RewardTermCfg
    from holosoma.config_types.scene import RigidObjectConfig,PhysicsConfig,IsaacSimPhysicsConfig
    motion_term=base.command.setup_terms['motion_command'];motion=motion_term.params['motion_config']
    required={'left_ankle_roll_link','right_ankle_roll_link','left_wrist_yaw_link','right_wrist_yaw_link'}
    if not required.issubset(motion.body_names_to_track) or motion.body_name_ref!=['torso_link']:
        raise RuntimeError('Unexpected tracking links: hold rewards require feet, wrists and torso reference')
    motion=replace(motion,motion_file=inputs['motion'],motion_dir='',use_adaptive_timesteps_sampler=True,
                   start_at_timestep_zero_prob=.5,freeze_at_timestep_zero_prob=0.,
                   enable_default_pose_prepend=False,enable_default_pose_append=False,
                   noise_to_initial_pose=replace(motion.noise_to_initial_pose,overall_noise_scale=0.))
    command=replace(base.command,setup_terms={**base.command.setup_terms,
        'motion_command':replace(motion_term,params={**motion_term.params,'motion_config':motion})})
    setup=dict(base.randomization.setup_terms)
    # Keep state/buffer terms, disable noise and pushes; remove object property randomizers.
    for name in list(setup):
        if name.startswith('randomize_object_') or name=='randomize_robot_rigid_body_material_startup':
            del setup[name]
    for name in ('push_randomizer_state','setup_dof_pos_bias','randomize_base_com_startup'):
        if name in setup:setup[name]=replace(setup[name],params={**setup[name].params,'enabled':False})
    scene=replace(base.scene,env_spacing=3.,rigid_objects={'object':RigidObjectConfig(
        urdf_file=inputs['object_urdf'],fixed=False,position=inputs['initial_pos'],orientation=inputs['initial_quat'],
        physics=PhysicsConfig(mass=1.,isaacsim=IsaacSimPhysicsConfig(static_friction=.9,dynamic_friction=.9,restitution=0.)))})
    sim=replace(base.simulator,config=replace(base.simulator.config,
        sim=replace(base.simulator.config.sim,fps=200,control_decimation=4)))
    reward_terms=dict(base.reward.terms)
    object_reward=reward_terms['object_global_ref_position_error_exp']
    reward_terms['object_global_ref_position_error_exp']=replace(object_reward,
        weight=3.0,params={**object_reward.params,'sigma':0.15})
    for key,weight,sigma in (
        ('object_global_ref_orientation_error_exp',3.0,0.35),
        ('motion_global_ref_orientation_error_exp',2.0,0.3),
        ('motion_relative_body_orientation_error_exp',2.0,0.4),
    ):
        term=reward_terms[key]
        reward_terms[key]=replace(term,weight=weight,params={**term.params,'sigma':sigma})
    for kind,weight in (('tilt',-1.0),('robot_velocity',-0.8),
                        ('feet_position',-1.0),('robot_position',-0.8),
                        ('box_velocity',-0.5),('box_position',-1.0)):
        reward_terms['settling_'+kind]=RewardTermCfg(
            func='run_lift_fresh:settling_cost',weight=weight,
            params={'kind':kind,'settling_start':max(0,inputs['original_frames']-20),'ramp_frames':20})
    reward_terms['supported_stationary_hold']=RewardTermCfg(
        func='run_lift_fresh:supported_hold',weight=3.0,
        params={'hold_start':inputs['original_frames'],'collision_bounds':inputs['collision_bounds']})
    termination_terms=dict(base.termination.terms)
    tracking_key='bad_tracking'
    tracking=termination_terms[tracking_key]
    termination_terms[tracking_key]=replace(tracking,params={**tracking.params,'bad_object_pos_threshold':0.45})
    cfg=replace(base,command=command,scene=scene,simulator=sim,
        reward=replace(base.reward,terms=reward_terms),
        termination=replace(base.termination,terms=termination_terms),
        randomization=replace(base.randomization,setup_terms=setup),
        logger=DisabledLoggerConfig(base_dir=str(out/'logs')),
        training=replace(base.training,project='LiftFresh',name=CLIP,checkpoint=None,num_envs=args.num_envs,headless=True,seed=2702,export_onnx=False),
        algo=replace(base.algo,config=replace(base.algo.config,num_learning_iterations=args.iterations,num_learning_epochs=5,
            num_mini_batches=4,save_interval=250,init_at_random_ep_len=False,load_optimizer=False,
            actor_learning_rate=3e-4,max_actor_learning_rate=3e-4)))
    if cfg.training.checkpoint is not None:
        raise RuntimeError('Fresh training must not load checkpoint weights')
    cfg.save_config(str(out/'pilot_config.yaml'))
    return cfg

def worker(args):
    import numpy as np
    inputs=json.loads((args.out/'pilot_inputs.json').read_text())
    cfg=build_config(inputs,args.out,args)
    # Launch IsaacSim before importing the environment or motion terms.
    from holosoma.train_agent import TrainingContext
    with TrainingContext(cfg) as context:
        from holosoma.managers.command.terms.wbt import MotionCommand,FAKE_BODY_NAME_ALIASES
        from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
        from holosoma.agents.ppo.ppo import PPO
        import torch
        write_json(args.out/'reward_math_check.json',check_reward_math('cuda' if torch.cuda.is_available() else 'cpu'))
        original_training_step=PPO._training_step
        updates_checked=0
        rollout_finite=None
        def finite_training_step(algo):
            nonlocal updates_checked
            # One host synchronization per rollout instead of many per simulation step.
            if rollout_finite is not None and not rollout_finite.item():
                raise RuntimeError('Nonfinite rollout action, observation or reward')
            if rollout_finite is not None:rollout_finite.fill_(True)
            losses=original_training_step(algo)
            for name,value in losses.items():
                if torch.is_tensor(value):
                    if not torch.isfinite(value).all():raise RuntimeError('Nonfinite PPO loss: '+name)
                elif isinstance(value,(int,float)) and not np.isfinite(value):
                    raise RuntimeError('Nonfinite PPO loss: '+name)
            updates_checked+=1
            return losses
        PPO._training_step=finite_training_step
        original_step=WholeBodyTrackingManager.step
        checked_steps=0
        def finite_step(env,actor_state):
            nonlocal checked_steps,rollout_finite
            if rollout_finite is None:
                rollout_finite=torch.ones((),dtype=torch.bool,device=actor_state['actions'].device)
            rollout_finite.logical_and_(torch.isfinite(actor_state['actions']).all())
            result=original_step(env,actor_state)
            def check(value):
                if isinstance(value,dict):
                    for item in value.values():check(item)
                elif torch.is_tensor(value):
                    rollout_finite.logical_and_(torch.isfinite(value).all().to(rollout_finite.device))
            check(result[0]);check(result[1]);checked_steps+=1
            return result
        WholeBodyTrackingManager.step=finite_step
        original=MotionCommand.setup
        def checked_setup(self):
            sim=self._env.simulator
            with np.load(inputs['motion'],allow_pickle=False) as d:
                bodies=d['body_names'].tolist();joints=d['joint_names'].tolist()
            runtime=dict(robot_body_names=list(sim._body_list),robot_joint_names=list(sim.dof_names))
            runtime['missing_bodies']=[n for n in sim._body_list if FAKE_BODY_NAME_ALIASES.get(n,n) not in bodies]
            runtime['missing_joints']=[n for n in sim.dof_names if n not in joints]
            obj=sim.scene.rigid_objects['object']
            view=obj.root_physx_view
            runtime['effective_mass_kg']=view.get_masses().cpu().tolist()
            runtime['effective_inertias']=view.get_inertias().cpu().tolist()
            runtime['effective_materials']=view.get_material_properties().cpu().tolist()
            write_json(args.out/'asset_runtime.json',runtime)
            print('LIFT_ASSET_CHECK:',json.dumps(runtime),flush=True)
            if runtime['missing_bodies'] or runtime['missing_joints']:
                raise RuntimeError('Reference/model name mismatch. See asset_runtime.json; no aliases fabricated.')
            masses=np.array(runtime['effective_mass_kg'])
            if not np.allclose(masses,1.,atol=.001):raise RuntimeError('Effective box mass is not 1 kg')
            inertias=np.array(runtime['effective_inertias'])
            if not np.isfinite(inertias).all():raise RuntimeError('Invalid effective box inertia')
            matrices=inertias.reshape(-1,3,3)
            if np.any(np.linalg.eigvalsh(matrices)<=0):raise RuntimeError('Nonpositive effective box inertia')
            expected=np.array(inputs['inertia_diagonal_kg_m2'])
            if not np.allclose(np.sort(np.linalg.eigvalsh(matrices),axis=-1),np.sort(expected),rtol=.05,atol=1e-5):
                raise RuntimeError('Imported box inertia differs from the authored pilot inertia')
            materials=np.array(runtime['effective_materials'])
            if not np.allclose(materials[...,:3],np.array([.9,.9,0.]),atol=.001):
                raise RuntimeError('Imported box material differs from the controlled pilot settings')
            original(self)
        MotionCommand.setup=checked_setup
        try:context.train()
        finally:
            MotionCommand.setup=original
            WholeBodyTrackingManager.step=original_step
            PPO._training_step=original_training_step
            write_json(args.out/'finite_rollout_check.json',dict(steps_checked=checked_steps,
                updates_checked=updates_checked,
                torch_peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path.home()/'holosoma')
    p.add_argument('--data',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--num-envs',type=int,default=128)
    p.add_argument('--iterations',type=int,default=7000)
    p.add_argument('--replay',action='store_true',help='Replay the checkpoint using this script so custom hold rewards resolve')
    p.add_argument('--checkpoint',type=Path,help='Replay checkpoint only; training always starts fresh')
    a=p.parse_args();a.repo=a.repo.resolve()
    if a.num_envs<4 or a.num_envs>256:raise SystemExit('Pilot supports 4 to 256 environments')
    if a.iterations<1:raise SystemExit('--iterations must be positive')
    if a.worker:
        try:worker(a)
        except BaseException:
            (a.out/'worker_exception.txt').write_text(traceback.format_exc());raise
        return
    if a.replay:
        if a.checkpoint is None:
            runs=sorted(a.repo.glob('t3/fresh_*/result.json'),key=lambda p:p.stat().st_mtime,reverse=True)
            for result in runs:
                data=json.loads(result.read_text())
                if data.get('training_completed') and data.get('checkpoints'):
                    a.checkpoint=Path(data['checkpoints'][-1]['path']);break
        if a.checkpoint is None or not a.checkpoint.expanduser().is_file():
            raise SystemExit('No completed fresh checkpoint found. Supply --checkpoint /path/to/model.pt')
        a.checkpoint=a.checkpoint.expanduser().resolve()
        replay(a);return
    if a.checkpoint is not None:
        raise SystemExit('--checkpoint is only for --replay. Training starts from random weights.')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo,text=True).strip()
    if commit!=COMMIT:raise SystemExit('Unexpected Holosoma revision: '+commit)
    if a.data is None:
        candidates=[a.repo/'t2/github_lift_carry',a.repo/'t2/holosoma_npz32']
        a.data=next((d for d in candidates if (d/'manifest.json').is_file()),None)
        if a.data is None:raise SystemExit('Export manifest not found. Supply --data /path/to/export-root')
    a.data=a.data.resolve()
    if a.out is None:a.out=a.repo/'t3'/('fresh_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'run_lift_fresh.py').write_text(Path(__file__).read_text())
    inputs=prepare(a.data,a.out)
    inputs=append_terminal_hold(inputs,a.out)
    write_json(a.out/'provenance.json',dict(commit=commit,python=sys.executable,script_sha256=digest(Path(__file__)),data=str(a.data),resume_checkpoint=None,initialization="random weights; fresh optimizer",
        changes={'object_position_reward_weight':3.0,'object_position_reward_sigma_m':0.15,
                 'object_position_reset_threshold_m':0.45,'load_optimizer':False,'object_orientation_weight':3.0,
                 'root_orientation_weight':2.0,'body_orientation_weight':2.0,
                 'terminal_hold_seconds':3.0,'actor_learning_rate_max':3e-4,'supported_stationary_hold_weight':3.0,
                 'foot_target_offset_fix':'body_pos_w already contains env origins',
                 'tilt_tracking':'mean pelvis and torso projected-gravity error',
                 'settling_cost_weights':{'tilt':-1.0,'robot_velocity':-0.8,'feet_position':-1.0,
                                         'robot_position':-0.8,'box_velocity':-0.5,'box_position':-1.0},
                 'settling_start_frame':inputs['original_frames']-20,
                 'settling_cost_gate':'20-frame phase ramp only; no object-error gate',
                 'settling_cost_shape':'sqrt(1+normalized squared error)-1'},
        diagnostic_evidence='Repeated forward drift after fine-tuning. Corrected double environment-origin addition in the previous stationary foot target; added supported hold reward; fresh initialization requested.'))
    if a.prepare_only:
        print('Prepared:',a.out);return
    argv=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--repo',str(a.repo),'--out',str(a.out),
          '--num-envs',str(a.num_envs),'--iterations',str(a.iterations)]
    env=os.environ.copy();env['PYTHONUNBUFFERED']='1'
    # Capture the full startup output as well as the trainer logs.
    with (a.out/'console.log').open('w') as log:
        process=subprocess.Popen(argv,cwd=a.repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            for line in process.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
            code=process.wait()
        except KeyboardInterrupt:
            process.terminate();code=process.wait()
    if (a.out/'worker_exception.txt').exists():code=1
    checkpoints=[dict(path=str(f),size_bytes=f.stat().st_size) for f in sorted(a.out.glob('logs/**/*.pt'))]
    finite_file=a.out/'finite_rollout_check.json'
    updates=json.loads(finite_file.read_text()).get('updates_checked',0) if finite_file.exists() else 0
    if not checkpoints or updates!=a.iterations:code=1
    write_json(a.out/'result.json',dict(returncode=code,training_completed=code==0,
        policy_success_rate=None,num_envs=a.num_envs,iterations_requested=a.iterations,updates_completed=updates,
        checkpoints=checkpoints,training_scope='one derived clip with 3s terminal hold; reference-phase resets; not task evaluation'))
    # Stock PPO exports ONNX on checkpoint save even when training.export_onnx is false.
    # Any automatically produced ONNX here is an unvalidated intermediate artifact.
    report=a.out/'lift_fresh_report.zip'
    with zipfile.ZipFile(report,'x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(a.out.rglob('*')):
            if f.is_file() and f.suffix in ('.json','.yaml','.log','.txt','.urdf','.py'):
                z.write(f,str(f.relative_to(a.out)))
    print('\nREPORT:',report,flush=True)
    if code:raise SystemExit(code)

if __name__=='__main__':main()
