"""Integration helpers, not a standalone replay or a Holosoma scene patch.

Import only AFTER Isaac Sim AppLauncher/SimulationApp starts. Call make_sensor_cfg
when constructing the installed scene config; sensor construction must precede
scene initialization. Actual rigid-body paths must be measured in the Stage.
Use PhysX dynamic controller replay for force interpretation, never teleport
both reference bodies into position every frame. No simulation is launched here.
"""
import csv
import json
from pathlib import Path


def make_sensor_cfg(palm_body_path, box_body_path, points=True):
    from isaaclab.sensors import ContactSensorCfg
    if not palm_body_path or not box_body_path:
        raise ValueError('Supply exact, verified body paths.')
    kwargs=dict(prim_path=palm_body_path,filter_prim_paths_expr=[box_body_path],
                update_period=0.,history_length=1,debug_vis=True,track_pose=True)
    if points:
        if not hasattr(ContactSensorCfg,'track_contact_points'):
            raise RuntimeError('Installed sensor API lacks contact-point tracking; inspect version before adapting.')
        kwargs.update(track_contact_points=True,max_contact_data_count_per_prim=16)
    return ContactSensorCfg(**kwargs)


def dump_stage_colliders(output):
    """Call in Isaac Sim's Script Editor. Reads the live Stage without editing it."""
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics
    stage=omni.usd.get_context().get_stage()
    if stage is None: raise RuntimeError('No open Stage.')
    cache=UsdGeom.XformCache(Usd.TimeCode.Default())
    rows=[]
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        rigid=prim.HasAPI(UsdPhysics.RigidBodyAPI)
        collision=prim.HasAPI(UsdPhysics.CollisionAPI)
        if not (rigid or collision): continue
        path=str(prim.GetPath())
        # Inventory env_0 only, plus non-environment scene bodies.
        if '/envs/env_' in path and '/envs/env_0/' not in path and not path.endswith('/envs/env_0'): continue
        mesh=UsdPhysics.MeshCollisionAPI(prim) if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else None
        row=dict(path=path,type=prim.GetTypeName(),rigid_body=rigid,collider=collision,
            collision_enabled=UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() if collision else None,
            approximation=mesh.GetApproximationAttr().Get() if mesh else None,
            world_transform=[list(map(float,r)) for r in cache.GetLocalToWorldTransform(prim)])
        extent=prim.GetAttribute('extent')
        if extent and extent.HasValue(): row['local_extent']=[list(map(float,v)) for v in extent.Get()]
        size=prim.GetAttribute('size')
        if size and size.HasValue(): row['cube_size']=float(size.Get())
        relevant=('/Robot/' not in path or any(word in path.lower() for word in ('wrist','hand','palm')))
        if collision and relevant and prim.IsA(UsdGeom.Mesh):
            mesh_geom=UsdGeom.Mesh(prim)
            points=mesh_geom.GetPointsAttr().Get()
            if points is not None:
                row['authored_mesh_points_local']=[list(map(float,v)) for v in points]
                row['face_vertex_counts']=list(map(int,mesh_geom.GetFaceVertexCountsAttr().Get() or []))
                row['face_vertex_indices']=list(map(int,mesh_geom.GetFaceVertexIndicesAttr().Get() or []))
                row['mesh_note']='Authored collision source mesh; PhysX convex approximation may differ. Inspect collider debug outlines.'
        rows.append(row)
    output=Path(output).expanduser()
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(meters_per_stage_unit=UsdGeom.GetStageMetersPerUnit(stage),
        note='Extents alone are not calibrated palm sites. Convex hull extents do not describe crate wall geometry.',
        prims=rows),indent=2)+'\n')
    print('Stage collider inventory:',output)


class PalmForceLogger:
    """Per-environment filtered forces, reduced without mixing environments.

    Force matrices can contain normal forces only on older PhysX/IsaacLab
    versions. This logger labels them as filtered reported forces, not full
    frictional support or a grasp-success metric. A ~1 N threshold is diagnostic.
    """
    def __init__(self,path,threshold_N=1.):
        if threshold_N<=0: raise ValueError('Threshold must be positive.')
        path=Path(path).expanduser()
        if path.exists(): raise ValueError('Force log already exists; use a fresh filename.')
        path.parent.mkdir(parents=True,exist_ok=True)
        self.file=path.open('w',newline='');self.writer=csv.writer(self.file)
        self.writer.writerow(['time_s','env_id','left_filtered_force_N','right_filtered_force_N',
            'left_contact','right_contact','left_point_x','left_point_y','left_point_z',
            'right_point_x','right_point_y','right_point_z'])
        self.threshold=float(threshold_N)

    @staticmethod
    def sample(sensor):
        import torch
        matrix=sensor.data.force_matrix_w
        if matrix is None or matrix.ndim!=4 or tuple(matrix.shape[1:3])!=(1,1):
            raise RuntimeError('Expected filtered matrix (envs,1 palm body,1 box filter,3). Check prim paths/reporting.')
        if not torch.isfinite(matrix).all(): raise RuntimeError('Non-finite force readings.')
        forces=matrix[:,0,0,:].norm(dim=-1).detach().cpu().tolist()
        points=getattr(sensor.data,'contact_pos_w',None)
        if points is None:
            xyz=[[float('nan')]*3 for _ in forces]
        else:
            p=points.detach().cpu().reshape(len(forces),-1,3)
            xyz=torch.nanmean(p,dim=1).tolist()
        return forces,xyz

    def log(self,time_s,left_sensor,right_sensor):
        left,lp=self.sample(left_sensor);right,rp=self.sample(right_sensor)
        if len(left)!=len(right): raise RuntimeError('Sensor environment counts differ.')
        for i,(l,r) in enumerate(zip(left,right)):
            self.writer.writerow([time_s,i,l,r,int(l>self.threshold),int(r>self.threshold),*lp[i],*rp[i]])
        self.file.flush()

    def close(self): self.file.close()
