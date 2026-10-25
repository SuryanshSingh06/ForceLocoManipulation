#!/usr/bin/env python3
"""Retarget lift clips only, with source and corrected-motion graph screening.

Version 2. Accepted T1 lifts only by default, 10 most recently accepted attempts.
--limit 0 screens all accepted lifts; --include-pending opts into unreviewed lifts.
Existing T1 rejections and t2_rejections.json IDs are always excluded.
Use a NEW output folder for this version. Rerun the identical command to resume.

Requirements: existing hoi-retarget environment (torch, numpy, scipy, trimesh)
plus matplotlib. Input: T1 manifest retaining meshes path/checksum metadata and
original local InterMimic tensors. Source meshes must match original T1 scale.

Example:
  python t2_batch_retarget.py --out ~/holosoma/t2/hoi_lift_quality01

Per clip: source_quality.png/json, source_curves.csv, and, after retargeting,
retargeted_quality.png/json and retargeted_curves.csv. status.json distinguishes
quality_rejected from failed (input/solver/checker errors). Original reviews and
recordings are never changed. Rejected outputs are retained for inspection.
Only completed clips appear in passed_lift_manifest.json. Downstream conversion
must use this manifest rather than recursively collecting every output pickle.

Adjust thresholds with --quality-config /path/to/thresholds.json, e.g.
{"min_rise_m": 0.12, "hold_s": 0.4, "max_travel_m": 0.35}.
These defaults are screening heuristics, not calibrated labels or task-success
criteria. Root tilt is relative to world up; pure yaw is allowed. Foot sliding
uses ankle links near their per-clip lowest height as a stance proxy. It does
not prove contact and may flag legitimate stepping. Object bottom uses rotated
mesh geometry. Robot-object penetration, forces, and dynamic balance are not
certified. Continue visual review before training.

HOI still produces 30 Hz motion. Run the existing 50 Hz resampling/export steps
separately on the passing set. No training config or policy is modified here.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


QUALITY_DEFAULTS = {
    'min_rise_m': .12, 'ground_tolerance_m': .06,
    'hold_s': .4, 'hold_range_m': .05, 'max_travel_m': .35,
    'max_penetration_m': .04, 'max_object_speed_m_s': 3.,
    'max_object_angular_speed_rad_s': 10., 'max_root_tilt_deg': 65.,
    'bad_tilt_s': .15, 'foot_ground_band_m': .08,
    'max_foot_slide_m_s': .30, 'foot_slide_s': .2,
    'max_joint_speed_rad_s': 20.,
}


def longest_run(mask):
    best = run = 0
    for value in mask:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def mesh_points(path):
    import numpy as np
    import trimesh
    from scipy.spatial import ConvexHull, QhullError
    mesh = trimesh.load(str(path), force='mesh', process=False)
    points = np.asarray(mesh.vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or len(points) < 4:
        raise ValueError(f'Invalid object mesh: {path}')
    points = np.unique(points, axis=0)
    try:
        return points[ConvexHull(points).vertices]
    except QhullError:
        return points


def corrected_mesh(data):
    """Read the single-link HOI URDF, preserving its mesh/origin convention."""
    import numpy as np
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation
    path = Path(data['object_model_path']).expanduser().resolve()
    root = ET.parse(path).getroot()
    links = root.findall('link')
    if len(links) != 1:
        raise ValueError('Quality check supports single-link object URDFs only.')
    geometries = links[0].findall('collision') or links[0].findall('visual')
    points = []
    for geom in geometries:
        mesh = geom.find('geometry/mesh')
        if mesh is None:
            raise ValueError('Object quality check needs mesh geometry, not primitives.')
        name = mesh.attrib['filename']
        if '://' in name:
            raise ValueError('Resolve object mesh URI to a file before screening.')
        v = mesh_points(path.parent / name)
        v = v * np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
        origin = geom.find('origin')
        if origin is not None:
            v = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).apply(v)
            v += np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        points.append(v)
    if not points:
        raise ValueError('No object geometry found.')
    scale = float(data.get('object_scale', 1.)) if data.get('scale_object_mesh', True) else 1.
    if not np.isclose(float(data.get('object_aug_scale', 1.)), 1.):
        raise ValueError('Augmented object scaling requires a verified geometry mapping.')
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid object scale.')
    return np.concatenate(points) * scale


def evaluate_curve(pos, quat, vertices, fps, cfg, robot=None):
    """Numerical checks driving the saved graph. Quaternion input is XYZW."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    pos, quat = np.asarray(pos, float), np.asarray(quat, float)
    if pos.ndim != 2 or pos.shape[1] != 3 or len(pos) < 3 or quat.shape != (len(pos), 4):
        raise ValueError('Invalid object trajectory dimensions.')
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(pos).all() or not np.isfinite(quat).all():
        raise ValueError('Non-finite trajectory or invalid FPS.')
    if np.any(np.abs(np.linalg.norm(quat, axis=1) - 1) > .05):
        raise ValueError('Object quaternion norm differs from 1 by >0.05.')
    rot = Rotation.from_quat(quat)
    mats = rot.as_matrix()
    bottom = np.concatenate([
        (np.einsum('tij,vj->tvi', mats[s:s+64], vertices) + pos[s:s+64, None])[:, :, 2].min(axis=1)
        for s in range(0, len(pos), 64)])
    xy = pos[:, :2]
    # Pairwise diameter is the travel measure, including a path that returns to its start.
    travel = max(float(np.linalg.norm(xy - point, axis=1).max()) for point in xy)
    speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) * fps
    omega = (rot[:-1].inv() * rot[1:]).magnitude() * fps
    need = max(2, int(np.ceil(cfg['hold_s'] * fps)) + 1)
    # Entire initial hold interval must be close to the floor, so initially-held clips fail.
    initial = bottom[:min(len(bottom), max(2, int(np.ceil(.1 * fps))))]
    floor_ok = bool(np.max(np.abs(initial)) <= cfg['ground_tolerance_m'])
    hold = any(np.min(bottom[i:i+need]) >= cfg['min_rise_m']
               and np.ptp(bottom[i:i+need]) <= cfg['hold_range_m']
               for i in range(len(bottom) - need + 1))
    metrics = dict(travel_m=travel, bottom_min_m=float(bottom.min()),
                   bottom_max_m=float(bottom.max()), starts_grounded=floor_ok,
                   stable_raised_hold=bool(hold), max_object_speed_m_s=float(speed.max()),
                   max_object_angular_speed_rad_s=float(omega.max()))
    reasons = []
    if not floor_ok: reasons.append('not_grounded_at_start')
    if not hold: reasons.append('no_stable_raised_hold')
    if travel > cfg['max_travel_m']: reasons.append('excess_horizontal_travel')
    if bottom.min() < -cfg['max_penetration_m']: reasons.append('object_floor_penetration')
    if speed.max() > cfg['max_object_speed_m_s']: reasons.append('object_position_jump')
    if omega.max() > cfg['max_object_angular_speed_rad_s']: reasons.append('object_rotation_jump')
    curves = dict(time=np.arange(len(pos)) / fps, bottom=bottom,
                  horizontal=np.linalg.norm(xy - xy[0], axis=1), speed=speed, omega=omega)
    if robot is not None:
        root_q, joints, feet = robot
        root_q, joints, feet = map(lambda x: np.asarray(x, float), (root_q, joints, feet))
        if root_q.shape != (len(pos), 4) or joints.shape != (len(pos), 29) or feet.shape != (len(pos), 2, 3):
            raise ValueError('Robot trajectory has inconsistent frame counts or dimensions.')
        if not all(np.isfinite(x).all() for x in (root_q, joints, feet)):
            raise ValueError('Non-finite robot trajectories.')
        if np.any(np.abs(np.linalg.norm(root_q, axis=1) - 1) > .05):
            raise ValueError('Invalid robot root quaternion.')
        up_z = Rotation.from_quat(root_q).as_matrix()[:, 2, 2]
        tilt = np.degrees(np.arccos(np.clip(up_z, -1, 1)))
        joint_speed = np.max(np.abs(np.diff(joints, axis=0)), axis=1) * fps
        foot_speed = np.linalg.norm(np.diff(feet[:, :, :2], axis=0), axis=2) * fps
        stance = (feet[:-1, :, 2] <= feet[:, :, 2].min(axis=0) + cfg['foot_ground_band_m']) & (
                  feet[1:, :, 2] <= feet[:, :, 2].min(axis=0) + cfg['foot_ground_band_m'])
        sliding = stance & (foot_speed > cfg['max_foot_slide_m_s'])
        slide_s = max(longest_run(sliding[:, i]) for i in range(2)) / fps
        tilt_s = longest_run(tilt > cfg['max_root_tilt_deg']) / fps
        if tilt_s >= cfg['bad_tilt_s']: reasons.append('robot_excessive_tilt')
        if slide_s >= cfg['foot_slide_s']: reasons.append('stance_foot_sliding')
        if joint_speed.max() > cfg['max_joint_speed_rad_s']: reasons.append('joint_position_jump')
        metrics.update(max_root_tilt_deg=float(tilt.max()), excessive_tilt_s=tilt_s,
                       stance_slide_s=slide_s, max_joint_speed_rad_s=float(joint_speed.max()))
        curves.update(tilt=tilt, foot_speed=foot_speed, stance=stance, joint_speed=joint_speed)
    return dict(accepted=not reasons, reasons=reasons, metrics=metrics, thresholds=cfg), curves


def quality_report(folder, stage, report, curves):
    import csv
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    save_json(folder / f'{stage}_quality.json', report)
    time = curves['time']
    columns = dict(time_s=time, object_bottom_m=curves['bottom'], horizontal_from_start_m=curves['horizontal'],
                   object_speed_m_s=np.r_[np.nan, curves['speed']],
                   object_angular_speed_rad_s=np.r_[np.nan, curves['omega']])
    if 'tilt' in curves:
        columns['root_tilt_deg'] = curves['tilt']
        columns['max_joint_speed_rad_s'] = np.r_[np.nan, curves['joint_speed']]
        for i in range(2):
            columns[f'foot_{i}_speed_m_s'] = np.r_[np.nan, curves['foot_speed'][:, i]]
            columns[f'foot_{i}_stance'] = np.r_[False, curves['stance'][:, i]]
    with (folder / f'{stage}_curves.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    cfg = report['thresholds']
    axes[0].plot(time, curves['bottom'], label='object lowest point')
    axes[0].axhline(cfg['min_rise_m'], color='g', ls='--', label='raised threshold')
    axes[0].axhline(-cfg['max_penetration_m'], color='r', ls='--', label='penetration limit')
    axes[0].set_ylabel('height (m)')
    axes[1].plot(time, curves['horizontal'], label='horizontal distance from start')
    axes[1].axhline(cfg['max_travel_m'], color='r', ls='--', label='travel limit (pairwise)')
    axes[1].set_ylabel('distance (m)')
    axes[2].plot(time[1:], curves['speed'], label='object linear speed (m/s)')
    axes[2].axhline(cfg['max_object_speed_m_s'], color='r', ls='--', label='linear speed limit')
    axes[2].plot(time[1:], curves['omega'], label='object angular speed (rad/s)', alpha=.6)
    if 'tilt' in curves:
        axes[3].plot(time, curves['tilt'], label='root tilt (degrees)')
        axes[3].axhline(cfg['max_root_tilt_deg'], color='r', ls='--', label='tilt limit')
        ax = axes[3].twinx()
        for i in range(2):
            ax.plot(time[1:], np.where(curves['stance'][:, i], curves['foot_speed'][:, i], np.nan),
                    alpha=.5, label=f'stance foot {i} speed')
        ax.axhline(cfg['max_foot_slide_m_s'], color='orange', ls='--')
        ax.set_ylabel('stance speed (m/s)')
        ax.legend(loc='upper right')
    else:
        axes[3].text(.02, .5, 'Source screen: robot checks run after retargeting.', transform=axes[3].transAxes)
    for axis in axes:
        axis.grid(alpha=.2)
        handles, labels = axis.get_legend_handles_labels()
        if handles: axis.legend(loc='upper left')
    axes[-1].set_xlabel('time (s)')
    decision = 'PASS screening' if report['accepted'] else 'REJECT: ' + ', '.join(report['reasons'])
    fig.suptitle(f'{folder.name}\n{stage}: {decision}', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(folder / f'{stage}_quality.png', dpi=140)
    plt.close(fig)


def screen_corrected(path, cfg):
    import pickle
    import numpy as np
    # Load only local outputs produced by the trusted retargeter.
    with path.open('rb') as stream:
        data = pickle.load(stream)
    names = list(data['link_body_list'])
    feet = [names.index(x) for x in ('left_ankle_roll_link', 'right_ankle_roll_link')]
    report, curves = evaluate_curve(data['object_pos'], np.asarray(data['object_rot'])[:, [1, 2, 3, 0]],
        corrected_mesh(data), float(data['fps']), cfg,
        robot=(np.asarray(data['root_rot'])[:, [1, 2, 3, 0]], data['dof_pos'],
               np.asarray(data['world_body_pos'])[:, feet]))
    quality_report(path.parent, 'retargeted', report, curves)
    return report


def write_selection(out, manifest, results):
    passed = {r['id'] for r in results if r.get('status') == 'completed'}
    result = dict(manifest)
    result['clips'] = [c for c in manifest['clips'] if c['id'] in passed]
    result['quality_note'] = 'Lift-only automated screening; manual/physics verification still required.'
    save_json(out / 'passed_lift_manifest.json', result)
    save_json(out / 'quality_rejections.json', [r for r in results if r.get('status') == 'quality_rejected'])



class QualityRejected(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path.home() / 'holosoma/t1/run05/manifest.json')
    parser.add_argument('--repo', type=Path, default=Path.home() / 'hoi-retarget')
    parser.add_argument('--out', type=Path, default=Path.home() / 'holosoma/t2/hoi_lift_quality01')
    parser.add_argument('--include-pending', action='store_true', help='Also screen unreviewed T1 lift candidates.')
    parser.add_argument('--limit', type=int, default=10, help='Most recent accepted clips to attempt; 0 means all. No automatic replacements.')
    parser.add_argument('--rejections', type=Path, default=Path.home() / 'holosoma/t2/t2_rejections.json')
    parser.add_argument('--quality-config', type=Path, help='JSON overriding named QUALITY_DEFAULTS thresholds.')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative.')
    cfg = dict(QUALITY_DEFAULTS)
    if args.quality_config:
        custom = json.loads(args.quality_config.expanduser().read_text())
        if not isinstance(custom, dict) or set(custom) - set(cfg):
            parser.error('Quality config contains unknown thresholds.')
        cfg.update(custom)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v < float('inf') for v in cfg.values()):
        parser.error('Quality thresholds must be finite positive numbers.')
    import numpy as np
    import scipy
    import trimesh
    import matplotlib
    matplotlib.use('Agg')

    args.manifest = args.manifest.expanduser().resolve()
    args.repo = args.repo.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    exe = shutil.which('hoi-retarget')
    if exe is None:
        parser.error('Activate the hoi-retarget conda environment first.')
    if not args.repo.is_dir():
        parser.error(f'Repository not found: {args.repo}')
    manifest = json.loads(args.manifest.read_text())
    rejection_path = args.rejections.expanduser().resolve()
    excluded = json.loads(rejection_path.read_text()) if rejection_path.exists() else []
    if not isinstance(excluded, list) or not all(isinstance(x, str) for x in excluded):
        parser.error('Rejection file must be a JSON array of full clip ID strings.')
    allowed_reviews = ('accepted', 'pending') if args.include_pending else ('accepted',)
    clips = [c for c in manifest['clips'] if c.get('task') == 'lift'
             and c.get('review') in allowed_reviews and c['id'] not in excluded]
    clips.sort(key=lambda c: (c.get('review') == 'accepted', c.get('reviewed_at') or '', c['id']), reverse=True)
    if args.limit:
        clips = clips[:args.limit]
    ids = [c['id'] for c in clips]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r'[A-Za-z0-9_.-]+', x) or x in ('.', '..') for x in ids):
        parser.error('Clip IDs must be unique and safe folder names.')
    if any(c['review'] not in ('accepted', 'pending') for c in clips):
        parser.error('Unexpected review status in manifest.')
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / '.batch.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('Another batch is already using this output directory.')

    import torch
    print(f'{len(clips)} lift clips selected; other tasks and reviewed rejections excluded.', flush=True)
    print(f'Outputs: {args.out}', flush=True)
    print('Contact mode, G1, 30 FPS, no videos. Ctrl+C stops; rerun to resume.', flush=True)
    counts = {'completed': 0, 'resumed': 0, 'quality_rejected': 0, 'failed': 0}
    results = []
    digests = {}
    last_source, last_tensor = None, None
    for index, clip in enumerate(clips, 1):
        folder = args.out / clip['task'] / clip['id']
        folder.mkdir(parents=True, exist_ok=True)
        state_path = folder / 'status.json'
        record = {'id': clip['id'], 'task': clip['task'], 'review': clip['review'],
                  'source': clip['source'], 'start': clip['start'], 'end': clip['end'],
                  'output_dir': str(folder), 'updated_at': stamp()}
        child = None
        try:
            source = Path(clip['source']).expanduser().resolve()
            if source not in digests:
                with source.open('rb') as stream:
                    h = hashlib.sha256()
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        h.update(block)
                digests[source] = h.hexdigest()
            if clip.get('sha256') and digests[source] != clip['sha256']:
                raise ValueError('Source checksum differs from T1 manifest.')
            settings = {'source_sha256': digests[source], 'start': clip['start'],
                        'end': clip['end'], 'object': clip['object'], 'robot': 'unitree_g1',
                        'mode': 'contact', 'fps': 30, 'repo': str(args.repo), 'runner_version': 2, 'quality_thresholds': cfg}
            meshes = manifest.get('meshes', {})
            mesh_info = meshes.get(clip['object'])
            if not mesh_info or 'path' not in mesh_info:
                raise ValueError('Manifest must retain T1 meshes metadata for exact source bottom heights.')
            mesh_path = Path(mesh_info['path']).expanduser()
            if not mesh_path.is_absolute():
                mesh_path = args.manifest.parent / mesh_path
            mesh_bytes = mesh_path.read_bytes()
            mesh_hash = hashlib.sha256(mesh_bytes).hexdigest()
            if mesh_info.get('sha256') and mesh_info['sha256'] != mesh_hash:
                raise ValueError('T1 object mesh checksum mismatch.')
            settings['source_mesh_sha256'] = mesh_hash
            record['settings'] = settings
            old = json.loads(state_path.read_text()) if state_path.exists() else None
            outputs = [folder / 'kinematic_window.pkl', folder / 'contact_window.pkl']
            if old and old.get('settings') and old['settings'] != settings:
                raise ValueError('Settings changed. Use a fresh --out directory to protect old results.')
            if old and old.get('status') in ('completed', 'quality_rejected') and all(p.is_file() and p.stat().st_size > 0 for p in outputs):
                report = screen_corrected(outputs[1], cfg)
                record['quality'] = report
                if not report['accepted']:
                    raise QualityRejected(', '.join(report['reasons']))
                counts['resumed'] += 1
                record['status'] = 'completed'
                record['resumed'] = True
                results.append(record)
                save_json(args.out / 'summary.json', {'counts': counts, 'results': results})
                write_selection(args.out, manifest, results)
                print(f'[{index}/{len(clips)}] SKIP completed {clip["id"]}', flush=True)
                continue
            if not old and any(p.exists() for p in outputs):
                raise ValueError('Untracked output exists. Use a fresh --out directory.')
            if source != last_source:
                last_tensor = None
                last_tensor = torch.load(source, map_location='cpu', weights_only=False)
                last_source = source
            if not isinstance(last_tensor, torch.Tensor) or last_tensor.ndim != 2 or last_tensor.shape[1] != 591:
                raise ValueError('Expected an InterMimic tensor with shape (frames, 591).')
            start, end = int(clip['start']), int(clip['end'])
            if not 0 <= start < end <= len(last_tensor):
                raise ValueError('Invalid T1 frame range.')
            if end - start < 2:
                raise ValueError('Crop needs at least two frames.')
            array = last_tensor[start:end].detach().cpu().numpy()
            report, curves = evaluate_curve(array[:, 318:321], array[:, 321:325],
                                             mesh_points(mesh_path), 30., cfg)
            quality_report(folder, 'source', report, curves)
            record['quality'] = report
            if not report['accepted']:
                raise QualityRejected(', '.join(report['reasons']))
            staged = folder / 'input' / source.name
            staged.parent.mkdir(exist_ok=True)
            torch.save(last_tensor[start:end].detach().cpu().clone(), staged)
            save_json(folder / 'selection.json', clip)
            command = [exe, '--input_file', str(staged), '--out_dir', str(folder),
                       '--robot', 'unitree_g1', '--mode', 'contact', '--tgt_fps', '30', '--no-record_video']
            record.update(status='running', command=command)
            save_json(state_path, record)
            print(f'[{index}/{len(clips)}] RUN {clip["id"]} ({end - start} source frames)', flush=True)
            log_path = folder / 'retarget.log'
            env = dict(os.environ, PYTHONUNBUFFERED='1')
            with log_path.open('w') as log:
                child = subprocess.Popen(command, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT,
                                         env=env, start_new_session=True)
                code = child.wait()
            log_text = log_path.read_text(errors='replace')
            passes = re.findall(r'Unified-window pass complete:.*?(\d+) failed', log_text)
            if code != 0:
                raise RuntimeError(f'Command exited {code}. See {log_path}')
            if not passes or any(int(x) != 0 for x in passes):
                raise RuntimeError(f'Contact pass missing or has failed windows. See {log_path}')
            if not all(p.is_file() and p.stat().st_size > 0 for p in outputs):
                raise RuntimeError('Retargeter did not save both output pickles.')
            report = screen_corrected(outputs[1], cfg)
            record['quality'] = report
            if not report['accepted']:
                raise QualityRejected(', '.join(report['reasons']))
            record.update(status='completed', updated_at=stamp())
            save_json(state_path, record)
            counts['completed'] += 1
            print(f'[{index}/{len(clips)}] OK {clip["id"]}', flush=True)
        except KeyboardInterrupt:
            if child is not None and child.poll() is None:
                import signal
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            record.update(status='interrupted', updated_at=stamp())
            save_json(state_path, record)
            results.append(record)
            save_json(args.out / 'summary.json', {'counts': counts, 'results': results})
            write_selection(args.out, manifest, results)
            print('\nStopped. Run the same command to resume.', flush=True)
            return 130
        except QualityRejected as exc:
            record.update(status='quality_rejected', reasons=str(exc), updated_at=stamp())
            save_json(state_path, record)
            counts['quality_rejected'] += 1
            print(f'[{index}/{len(clips)}] REJECT {clip["id"]}: {exc}', flush=True)
        except Exception as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}', updated_at=stamp())
            # Preserve original settings so incompatible completed outputs cannot be overwritten on retry.
            if state_path.exists():
                prior = json.loads(state_path.read_text())
                if prior.get('settings') != record.get('settings') or prior.get('status') == 'completed':
                    record['preserved_status_file'] = True
                else:
                    save_json(state_path, record)
            else:
                save_json(state_path, record)
            counts['failed'] += 1
            print(f'[{index}/{len(clips)}] FAIL {clip["id"]}: {record["error"]}', flush=True)
        results.append(record)
        save_json(args.out / 'summary.json', {'counts': counts, 'results': results})
        write_selection(args.out, manifest, results)
    save_json(args.out / 'summary.json', {'counts': counts, 'results': results})
    write_selection(args.out, manifest, results)
    print(f'Finished: {counts}. Summary: {args.out / "summary.json"}', flush=True)
    return 1 if counts['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
