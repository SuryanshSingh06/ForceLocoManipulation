"""Opt-in, process-local hooks for the uploaded Holosoma IsaacSim backend.

Call install_dynamic_contact_logging AFTER Isaac app startup and BEFORE env
construction. One environment only. No repo edits, mass/friction changes, welds
or pose writes. This adds filtered sensors and logs after the backend's existing
scene.update. It rejects kinematic objects. Run with a genuinely dynamic policy
controller, not reference-pose teleport playback.
"""
from pathlib import Path


def _rigid_prims(stage,root):
    from pxr import Usd,UsdPhysics
    prim=stage.GetPrimAtPath(root)
    if not prim.IsValid():raise RuntimeError(f'No Stage prim {root}')
    return [p for p in Usd.PrimRange(prim,Usd.TraverseInstanceProxies())
            if p.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get() is not False]


def install_dynamic_contact_logging(output,object_key=None):
    from holosoma.simulator.isaacsim.isaacsim import IsaacSim
    if getattr(IsaacSim,'_palm_qa_hooks_installed',False):raise RuntimeError('Contact hooks already installed.')
    output=Path(output).expanduser().resolve()
    if output.exists():raise ValueError('Contact-log output exists; choose a fresh directory.')
    output.mkdir(parents=True)
    original_setup=IsaacSim._setup_scene
    original_step=IsaacSim.simulate_at_each_physics_step

    def setup(self):
        if self.training_config.num_envs!=1:
            raise RuntimeError('This diagnostic hook requires one environment; configure the installed evaluator accordingly.')
        original_setup(self)
        import omni.usd
        from pxr import UsdPhysics
        from isaaclab.sensors import ContactSensor,ContactSensorCfg
        from isaac_palm_contact_tools import make_sensor_cfg,PalmForceLogger,dump_stage_colliders
        stage=omni.usd.get_context().get_stage()
        keys=list(self.scene.rigid_objects)
        key=object_key
        if key is None:
            if len(keys)!=1:raise RuntimeError(f'Specify one object key; available={keys}')
            key=keys[0]
        if key not in keys:raise RuntimeError(f'Missing object {key}; available={keys}')
        # Uploaded build_rigid_object uses /World/envs/env_.*/{name}.
        objects=_rigid_prims(stage,f'/World/envs/env_0/{key}')
        if len(objects)!=1:raise RuntimeError(f'Expected one rigid object body; found {[str(p.GetPath()) for p in objects]}')
        obj=objects[0]
        if UsdPhysics.RigidBodyAPI(obj).GetKinematicEnabledAttr().Get():
            raise RuntimeError('Object is kinematic. Use the dynamic task configuration, not the geometry viewer.')
        robot=_rigid_prims(stage,'/World/envs/env_0/Robot')
        palms=[]
        for side in ('left','right'):
            selected=None
            for name in (side+'_rubber_hand_link',side+'_wrist_yaw_link'):
                candidates=[p for p in robot if p.GetName()==name]
                if len(candidates)==1:selected=candidates[0];break
                if len(candidates)>1:raise RuntimeError('Ambiguous palm body path.')
            if selected is None:raise RuntimeError(f'Cannot identify {side} palm body; available={[p.GetName() for p in robot]}')
            palms.append(str(selected.GetPath()))
        self._palm_qa_sensors=[]
        point_tracking=hasattr(ContactSensorCfg,'track_contact_points')
        for side,path in zip(('left','right'),palms):
            sensor=ContactSensor(make_sensor_cfg(path,str(obj.GetPath()),points=point_tracking))
            self.scene.sensors['qa_'+side+'_palm']=sensor
            self._palm_qa_sensors.append(sensor)
        self._palm_qa_logger=PalmForceLogger(output/'filtered_forces.csv')
        (output/'resolved_paths.txt').write_text('\n'.join(['mode=dynamic_controller_run',
            'left='+palms[0],'right='+palms[1],'object='+str(obj.GetPath()),'point_tracking='+str(point_tracking),
            'The wrist fallback measures the rigid body carrying palm geometry; verify its colliders.',
            'Force values do not by themselves establish successful lifting.'])+'\n')
        dump_stage_colliders(output/'isaac_stage_colliders.json')

    def step(self,*args,**kwargs):
        result=original_step(self,*args,**kwargs)
        # Uploaded original_step already calls scene.update after sim.step.
        logger=getattr(self,'_palm_qa_logger',None)
        if logger is not None:
            logger.log(self._sim_step_counter/self.simulator_config.sim.fps,*self._palm_qa_sensors)
        return result

    IsaacSim._setup_scene=setup
    IsaacSim.simulate_at_each_physics_step=step
    IsaacSim._palm_qa_hooks_installed=True
