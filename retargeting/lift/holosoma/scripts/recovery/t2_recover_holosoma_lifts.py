#!/usr/bin/env python3
"""Recover all 104 Holosoma lift attempts without editing the installed solver.

Existing successful outputs are copied, not recomputed. Failed clips are rerun
through a process-local bounded solver retry. Completion is not grasp approval.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import runpy
import shutil
import signal
import subprocess
import sys

import t2_holosoma_lift_batch as base


def constraint_check(solver, q):
    import numpy as np
    _, phis=solver._update_jacobians_and_phis_from_q(q)
    _, self_phis=solver._compute_self_collision_constraints(getattr(solver,'_recovery_frame',0))
    worst=max([0.0,*[-float(p)-solver.penetration_tolerance for p in phis.values()],
               *[solver._self_collision_tolerance-float(p) for p in self_phis.values()]])
    actuated=np.asarray(q)[solver.q_a_indices]
    limits=max(0.,float(np.max(solver.q_a_lb-actuated)),float(np.max(actuated-solver.q_a_ub)))
    return dict(collision_excess_m=worst,joint_limit_excess=limits)


def checked_iterate(original, solver, positional, keyword, events):
    """On failed frame solves, follow the fixed object path in smaller steps.

    Foot anchoring remains at the previous OUTPUT frame for every internal step.
    Internal poses are discarded; the original final object pose is preserved.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp
    bound=inspect.signature(original).bind(solver,*positional,**keyword)
    bound.apply_defaults()
    args=dict(bound.arguments);args.pop('self')
    solver._recovery_frame=args.get('frame_idx',0)
    try:
        result=original(solver,**args)
        check=constraint_check(solver,result[0])
        if check['collision_excess_m']>.0002 or check['joint_limit_excess']>1e-5:
            raise RuntimeError('Nonlinear endpoint constraint check failed.')
        return result
    except RuntimeError as initial:
        if not any(s in str(initial) for s in ('Bounded recovery exhausted',
                                              'CVXPY solve failed: infeasible',
                                              'Nonlinear endpoint constraint check failed')):
            raise
        if args.get('init_t',False) or not solver.has_dynamic_object:
            raise
        start=np.asarray(args['q_t_last']).copy()
        target=np.asarray(args['q_locked']).copy()
        if np.allclose(start[-7:],target[-7:],atol=1e-10):
            raise
        rot=Slerp([0.,1.],Rotation.from_quat(np.stack([start[-4:],target[-4:]])[:,[1,2,3,0]]))
        for subdivisions in (4,8,16,32):
            current=start.copy()
            try:
                for substep in range(1,subdivisions+1):
                    alpha=substep/subdivisions
                    locked=target.copy()
                    locked[-7:-4]=(1-alpha)*start[-7:-4]+alpha*target[-7:-4]
                    locked[-4:]=rot(alpha).as_quat()[[3,0,1,2]]
                    if substep==subdivisions:
                        locked[-7:]=target[-7:]  # Exact requested endpoint, including quaternion sign.
                    trial=dict(args,q_locked=locked,q_n=current.copy(),n_iter=max(20,args['n_iter']))
                    # q_t_last deliberately stays at the previous output frame.
                    for polish in range(4):
                        candidate,cost=original(solver,**trial)
                        check=constraint_check(solver,candidate)
                        if check['collision_excess_m']<=.0002 and check['joint_limit_excess']<=1e-5:
                            current=candidate;break
                        trial['q_n']=candidate
                    else:
                        raise RuntimeError('Nonlinear endpoint constraint check failed.')
                if not np.allclose(current[-7:],target[-7:],atol=1e-10,rtol=0):
                    raise RuntimeError('Object endpoint changed unexpectedly.')
                event=dict(frame=args['frame_idx'],method='object_path_continuation',
                           subdivisions=subdivisions,endpoint_checks=check,
                           foot_anchor='previous_output_frame')
                events.append(event)
                print('[CONTINUATION] '+json.dumps(event),flush=True)
                return current,cost
            except RuntimeError as exc:
                if not any(s in str(exc) for s in ('Bounded recovery exhausted',
                                                  'CVXPY solve failed: infeasible',
                                                  'Nonlinear endpoint constraint check failed')):
                    raise
                print(f'[CONTINUATION] frame={args["frame_idx"]} steps={subdivisions}: {exc}',flush=True)
        raise RuntimeError(f'Object-path continuation exhausted at frame {args["frame_idx"]}; '
                           'object trajectory endpoints, collision and joint limits retained.') from initial


def recovery_solve(original, solver, positional, keyword, events):
    """Retry only infeasible subproblems; never disable collision or joint limits."""
    radius, foot = solver.step_size, solver.foot_sticking_tolerance
    frame = keyword.get('frame_idx', '?')
    profiles = [(radius, foot), (max(radius, .4), foot),
                (max(radius, .8), foot),
                (max(radius, .8), max(foot, .003)),
                (max(radius, .8), max(foot, .005)),
                (max(radius, .8), max(foot, .01))]
    profiles = list(dict.fromkeys(profiles))
    try:
        for attempt, (step, tolerance) in enumerate(profiles):
            solver.step_size, solver.foot_sticking_tolerance = step, tolerance
            try:
                result = original(solver, *positional, **keyword)
                if attempt:
                    event = dict(frame=frame, step_size=step,
                                 foot_sticking_tolerance_m=tolerance,
                                 collision_constraints=True, joint_limits=True)
                    events.append(event)
                    print('[RECOVERY] ' + json.dumps(event), flush=True)
                return result
            except RuntimeError as exc:
                if 'CVXPY solve failed: infeasible' not in str(exc):
                    raise
                print(f'[RECOVERY] frame={frame} step={step} foot={tolerance}: {exc}', flush=True)
                if attempt == len(profiles)-1:
                    raise RuntimeError(f'Bounded recovery exhausted at frame {frame}; '
                                       'collision/joint constraints retained.') from exc
    finally:
        solver.step_size, solver.foot_sticking_tolerance = radius, foot


def worker(entry, events_file, command):
    # Install the override on the package class before the entry point imports it.
    package = entry.parent.parent
    sys.path.insert(0, str(package.parent))
    module = importlib.import_module('holosoma_retargeting.src.interaction_mesh_retargeter')
    cls = module.InteractionMeshRetargeter
    original = cls.solve_single_iteration
    original_iterate=cls.iterate
    events = []
    instances=[]
    def wrapped(self, *args, **kwargs):
        if not instances:instances.append(self)
        if not self.activate_joint_limits or not self.activate_obj_non_penetration:
            raise RuntimeError('Recovery requires object nonpenetration and joint limits enabled.')
        return recovery_solve(original, self, args, kwargs, events)
    cls.solve_single_iteration = wrapped
    def iterate_wrapped(self,*args,**kwargs):
        return checked_iterate(original_iterate,self,args,kwargs,events)
    cls.iterate=iterate_wrapped
    # Some installed revisions import the same file by its short module name.
    sys.modules['interaction_mesh_retargeter'] = module
    sys.argv = [str(entry), *command]
    try:
        runpy.run_path(str(entry), run_name='__main__')
        # Validate every output frame of newly solved clips against nonlinear
        # geometry queries, not just the linearized QP constraints.
        import numpy as np
        result_dir=Path(command[command.index('--save-dir')+1])
        task=command[command.index('--task-name')+1]
        with np.load(result_dir/(task+'_original.npz'),allow_pickle=False) as data:
            qpos=np.asarray(data['qpos'])
        checks=[]
        for frame,q in enumerate(qpos):
            instances[0]._recovery_frame=frame
            checks.append(dict(frame=frame,**constraint_check(instances[0],q)))
        bad=[c for c in checks if c['collision_excess_m']>.0002 or c['joint_limit_excess']>1e-5]
        base.json_save(events_file.with_name('constraint_audit.json'),dict(frames=checks,failed_frames=len(bad),
                       note='Collision mask/pairs are those used by this retargeter. No grasp-force certification.'))
        if bad:
            raise RuntimeError(f'Nonlinear constraint audit failed in {len(bad)} frames; see constraint_audit.json.')
    finally:
        base.json_save(events_file, dict(recovery_events=events,
                       note='Foot tolerance is a per-frame XY bound, not total slip. '
                            'Completion does not establish physical support or hand contact.'))


def run(command, cwd, log):
    with log.open('w') as stream:
        proc = subprocess.Popen(command, cwd=cwd, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return proc.wait()
        except KeyboardInterrupt:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise


def diagnosis(log):
    lines = log.read_text(errors='replace').splitlines() if log.exists() else []
    return '\n'.join(lines[-35:])


def main():
    home = Path.home()/'holosoma'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, default=home/'t2/holosoma_lift_all03')
    parser.add_argument('--out', type=Path, default=home/'t2/holosoma_lift_all04')
    parser.add_argument('--selected-out', type=Path, default=home/'t2/holosoma_lift_selected24_04')
    parser.add_argument('--package', type=Path, default=home/'src/holosoma_retargeting/holosoma_retargeting')
    parser.add_argument('--selected-manifest', type=Path, default=home/'t2/hoi_all_lift_candidates01/passed_lift_manifest.json')
    parser.add_argument('--expected', type=int, default=104)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--entry', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--events', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.worker:
        worker(args.entry, args.events, args.command[1:] if args.command[:1]==['--'] else args.command)
        return
    for name in ('previous','out','selected_out','package','selected_manifest'):
        setattr(args,name,getattr(args,name).expanduser().resolve())
    for a,b in ((args.previous,args.out),(args.previous,args.selected_out),(args.out,args.selected_out)):
        if a==b or a.is_relative_to(b) or b.is_relative_to(a):
            parser.error('Previous, all-output and selected-output directories must be separate.')
    old_manifest = base.read(args.previous/'manifest.json')
    rows = old_manifest['clips']
    ids = [base.safe(c['id']) for c in rows]
    if len(rows)!=args.expected or len(set(ids))!=args.expected:
        parser.error(f'Expected {args.expected} distinct attempts; found {len(rows)}.')
    if any(c.get('task')!='lift' for c in rows):
        parser.error('Previous manifest contains non-lift rows.')
    pick_manifest=base.read(args.selected_manifest)
    picks=pick_manifest['clips'] if isinstance(pick_manifest,dict) else pick_manifest
    picked={c['id'] for c in picks}
    if len(picks)!=24 or len(picked)!=24 or not picked.issubset(ids):
        parser.error('Selected manifest must identify exactly 24 of the input clips.')
    entry=args.package/'examples/robot_retarget.py'
    solver=args.package/'src/interaction_mesh_retargeter.py'
    # Fail before processing the entire batch if the installed API differs.
    source=solver.read_text()
    required=('def solve_single_iteration(', 'CVXPY solve failed:',
              'self.step_size', 'self.foot_sticking_tolerance')
    if not all(s in source for s in required):
        parser.error('Installed solver differs from the supported API; no files changed.')
    for key in ('demo_data/height_dict.pkl', 'models/g1/g1_29dof.urdf'):
        if not (args.package/key).is_file():
            parser.error(f'Missing required package resource: {args.package/key}')
    import cvxpy as cp
    import mujoco
    import torch
    if 'CLARABEL' not in cp.installed_solvers():
        parser.error('CLARABEL is not installed in this environment.')
    settings=dict(version=2, previous=str(args.previous),
                  previous_manifest_sha256=base.digest(args.previous/'manifest.json'),
                  solver_sha256=base.digest(solver),entry_sha256=base.digest(entry),
                  script_sha256=base.digest(__file__),helper_sha256=base.digest(base.__file__),
                  selected_manifest_sha256=base.digest(args.selected_manifest),
                  out=str(args.out), selected_out=str(args.selected_out))
    for output in (args.out,args.selected_out):
        if output.exists() and any(output.iterdir()) and not (output/'recovery_settings.json').is_file():
            parser.error(f'Untracked nonempty output: {output}')
        output.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'.batch.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for output in (args.out,args.selected_out):
        path=output/'recovery_settings.json'
        if path.exists() and base.read(path)!=settings:
            parser.error('Recovery settings changed; choose fresh output paths.')
        base.json_save(path,settings)
    runtime=args.out/'runtime'
    if not (runtime/'ready.json').exists():
        if runtime.exists():
            parser.error(f'Incomplete runtime: {runtime}. Choose a fresh output folder.')
        shutil.copytree(args.previous/'runtime',runtime)
        for xml in (runtime/'models/g1').glob('*.xml'):
            xml.write_text(xml.read_text().replace(str(args.previous/'runtime'),str(runtime)))
    meta=runtime/'demo_data/height_dict.pkl'
    meta.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(args.package/'demo_data/height_dict.pkl',meta)
    # Use the same models that produced the original run.
    for obj in {c['object'] for c in rows}:
        model=mujoco.MjModel.from_xml_path(str(runtime/'models/g1'/f'g1_29dof_w_{obj}.xml'))
        if model.nq!=43:
            parser.error(f'{obj}: unexpected nq={model.nq}')
    records=[]
    def summary():
        counts=Counter(c['status'] for c in records)
        base.json_save(args.out/'manifest.json',dict(clips=records,counts=dict(counts),settings=settings))
        base.json_save(args.selected_out/'manifest.json',dict(expected=24,
                       completed=sum(c['status']=='completed' and c['id'] in picked for c in records),
                       clips=[c for c in records if c['id'] in picked]))
        return counts
    for index,row in enumerate(rows,1):
        cid=row['id']; old=args.previous/'lift'/cid; folder=args.out/'lift'/cid
        folder.mkdir(parents=True,exist_ok=True)
        state=folder/'status.json'
        record=dict(row,status='running',backend='holosoma',selected24=cid in picked)
        try:
            prior=base.read(state) if state.exists() else {}
            if prior.get('status')=='completed':
                for name,key in (('motion_original.npz','output_sha256'),('motion_fps50.npz','fps50_sha256')):
                    base.verify_motion(folder/name)
                    if base.digest(folder/name)!=prior[key]:
                        raise ValueError(f'Output hash changed: {folder/name}')
                record=prior
                print(f'[{index}/{len(rows)}] RESUME {cid}',flush=True)
            else:
                old_state=base.read(old/'status.json')
                base.json_save(folder/'selection.json',{k:row[k] for k in ('id','task','source','object','start','end')})
                if old_state.get('status')=='completed':
                    for name,key in (('motion_original.npz','output_sha256'),('motion_fps50.npz','fps50_sha256')):
                        base.verify_motion(old/name)
                        if base.digest(old/name)!=old_state.get(key):
                            raise ValueError(f'Original output hash mismatch: {old/name}')
                        shutil.copy2(old/name,folder/name)
                    shutil.copy2(old/'retarget.log',folder/'retarget.log')
                    for name in ('recovery_events.json','constraint_audit.json'):
                        if (old/name).exists():shutil.copy2(old/name,folder/name)
                    base.json_save(folder/'preserved_status.json',old_state)
                    record['recovery_method']='preserved_original_success'
                    print(f'[{index}/{len(rows)}] PRESERVE {cid}',flush=True)
                else:
                    source_path=Path(row['source'])
                    fingerprint=old_state.get('fingerprint',{})
                    if fingerprint.get('source_sha256')!=base.digest(source_path):
                        raise ValueError(f'Source hash mismatch: {source_path}')
                    try:
                        tensor=torch.load(source_path,map_location='cpu',weights_only=True)
                    except TypeError:
                        tensor=torch.load(source_path,map_location='cpu')
                    start,end=int(row['start']),int(row['end'])
                    if not isinstance(tensor,torch.Tensor) or tensor.ndim!=2 or tensor.shape[1]!=591 or not 0<=start<end<=len(tensor):
                        raise ValueError('Invalid source tensor or crop.')
                    input_dir=folder/'input';input_dir.mkdir(exist_ok=True)
                    torch.save(tensor[start:end].detach().cpu().clone(),input_dir/source_path.name)
                    result_dir=folder/'retargeted';result_dir.mkdir(exist_ok=True)
                    command=[sys.executable,str(Path(__file__).resolve()),'--worker',
                             '--entry',str(entry),'--events',str(folder/'recovery_events.json'),'--',
                             '--task-type','object_interaction','--robot','g1','--data-format','smplh',
                             '--task-name',source_path.stem,'--data-path',str(input_dir),
                             '--task-config.object-name',row['object'],'--save-dir',str(result_dir),
                             '--retargeter.activate-obj-non-penetration',
                             '--retargeter.activate-joint-limits','--retargeter.activate-foot-sticking']
                    record['command']=command
                    base.json_save(state,record)
                    print(f'[{index}/{len(rows)}] RECOVER {cid}',flush=True)
                    if run(command,runtime,folder/'retarget.log'):
                        raise RuntimeError(diagnosis(folder/'retarget.log'))
                    generated=result_dir/(source_path.stem+'_original.npz')
                    base.verify_motion(generated,end-start)
                    shutil.copy2(generated,folder/'motion_original.npz')
                    base.resample50(generated,folder/'motion_fps50.npz')
                    record['recovery_method']='bounded_recovery_and_object_path_continuation'
                record.update(status='completed',output_sha256=base.digest(folder/'motion_original.npz'),
                              fps50_sha256=base.digest(folder/'motion_fps50.npz'),
                              quality_review='pending', finished_at=base.stamp())
                record.pop('error',None)
                base.json_save(state,record)
            if cid in picked:
                destination=args.selected_out/'lift'/cid
                destination.mkdir(parents=True,exist_ok=True)
                for name in ('motion_original.npz','motion_fps50.npz','status.json','selection.json','retarget.log','recovery_events.json','constraint_audit.json'):
                    if (folder/name).exists():
                        shutil.copy2(folder/name,destination/name)
        except KeyboardInterrupt:
            record.update(status='interrupted');base.json_save(state,record)
            records.append(record);summary();raise
        except Exception as exc:
            record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            base.json_save(state,record)
            print(f'[{index}/{len(rows)}] FAILED {cid}: see {folder}/status.json',flush=True)
        records.append(record);summary()
    subset_runtime=args.selected_out/'runtime'
    if not subset_runtime.exists():
        shutil.copytree(runtime,subset_runtime)
        for xml in (subset_runtime/'models/g1').glob('*.xml'):
            xml.write_text(xml.read_text().replace(str(runtime),str(subset_runtime)))
    counts=summary()
    failed=[dict(id=r['id'],error=r.get('error','')) for r in records if r['status']=='failed']
    base.json_save(args.out/'failures.json',failed)
    print('Finished:',dict(counts),flush=True)
    print('All lifts:',args.out,flush=True)
    print('Selected 24:',sum(r['status']=='completed' and r['id'] in picked for r in records),'/24',flush=True)
    if failed:
        print('Remaining failures:',args.out/'failures.json',flush=True)
        raise SystemExit(1)


if __name__=='__main__':
    main()
