#!/usr/bin/env python3
"""Resample a corrected HOI-Retarget pickle, preserving its schema and mesh scale."""
import argparse
import copy
import os
import pickle
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


def resample(data, fps):
    source_fps = float(data['fps'])
    n = len(data['root_pos'])
    if n < 2 or source_fps <= 0 or not np.isfinite(source_fps):
        raise ValueError('Need at least two frames and a finite positive source FPS.')
    old_t = np.arange(n, dtype=float) / source_fps
    # Uniform sampling without inventing poses beyond the source duration.
    new_t = np.arange(int(np.floor(old_t[-1] * fps + 1e-8)) + 1, dtype=float) / fps
    if len(new_t) < 2:
        raise ValueError('Output needs at least two frames.')
    nearest = np.clip(np.floor(new_t * source_fps + 0.5).astype(int), 0, n - 1)
    result = copy.deepcopy(data)
    fk_fields = {'world_body_pos', 'world_body_orient', 'local_body_pos', 'local_body_orient'}
    for key, value in data.items():
        if not isinstance(value, np.ndarray) or value.ndim == 0 or len(value) != n:
            continue
        if key in fk_fields:
            continue
        if key in ('root_rot', 'object_rot'):
            q = np.asarray(value, dtype=float)
            if q.shape != (n, 4) or not np.all(np.isfinite(q)) or np.any(np.linalg.norm(q, axis=1) < 1e-8):
                raise ValueError(f'Invalid quaternion field {key}.')
            rotations = Rotation.from_quat(q[:, [1, 2, 3, 0]])
            result[key] = Slerp(old_t, rotations)(new_t).as_quat()[:, [3, 0, 1, 2]].astype(value.dtype)
        elif (value.dtype.kind in 'biu' or 'flag' in key or 'contact' in key
              or not np.all(np.isfinite(value))):
            # Contact targets include NaNs for inactive links: keep those aligned with labels.
            result[key] = value[nearest].copy()
        elif value.dtype.kind == 'f':
            result[key] = interp1d(old_t, value, axis=0)(new_t).astype(value.dtype)
        else:
            result[key] = value[nearest].copy()
    result['fps'] = float(fps)
    if 'object_pos' in result:
        result['object_vel'] = np.gradient(result['object_pos'].astype(float), 1 / fps, axis=0,
                                           edge_order=2 if len(new_t) > 2 else 1).astype(np.float32)
    result.setdefault('meta', {})['resampling'] = {
        'source_fps': source_fps, 'source_frames': n, 'output_fps': fps,
        'output_frames': len(new_t), 'source_duration_s': float(old_t[-1]),
        'output_duration_s': float(new_t[-1]), 'quaternion_order': 'wxyz',
        'contact_sampling': 'nearest', 'body_poses': 'recomputed_forward_kinematics',
    }
    return result


def recompute_body_poses(data):
    import torch
    from hoi_retarget.rendering import video

    robot = data.get('meta', {}).get('robot', 'unitree_g1')
    model = video.KinematicsModel(str(video.ROBOT_XML_DICT[robot]), device='cpu')
    root_pos = torch.from_numpy(np.asarray(data['root_pos'], dtype=np.float32))
    root_rot = torch.from_numpy(np.asarray(data['root_rot'], dtype=np.float32)[:, [1, 2, 3, 0]])
    dof_pos = torch.from_numpy(np.asarray(data['dof_pos'], dtype=np.float32))
    with torch.no_grad():
        wp, wq = model.forward_kinematics(root_pos=root_pos, root_rot=root_rot, dof_pos=dof_pos)
        identity = torch.zeros_like(root_rot)
        identity[:, 3] = 1
        lp, lq = model.forward_kinematics(root_pos=torch.zeros_like(root_pos),
                                         root_rot=identity, dof_pos=dof_pos)
    data['world_body_pos'] = wp.cpu().numpy()
    data['world_body_orient'] = wq[..., [3, 0, 1, 2]].cpu().numpy()
    data['local_body_pos'] = lp.cpu().numpy()
    data['local_body_orient'] = lq[..., [3, 0, 1, 2]].cpu().numpy()
    data['link_body_list'] = list(model.body_names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path.home() / 'holosoma/t2/hoi_test_lift/contact_window.pkl')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--fps', type=float, default=50)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error('--fps must be finite and positive.')
    source = args.input.expanduser().resolve()
    output = (args.output.expanduser().resolve() if args.output else
              source.with_name(f'{source.stem}_fps{args.fps:g}.pkl'))
    if output == source:
        parser.error('Output must differ from input.')
    if output.exists() and not args.overwrite:
        parser.error(f'{output} already exists. Use --overwrite to regenerate it.')
    with source.open('rb') as stream:
        data = pickle.load(stream)
    converted = resample(data, args.fps)
    recompute_body_poses(converted)
    count = len(converted['root_pos'])
    for key in ('root_pos', 'root_rot', 'dof_pos', 'world_body_pos', 'world_body_orient',
                'local_body_pos', 'local_body_orient', 'object_pos', 'object_rot'):
        value = converted[key]
        if len(value) != count or not np.all(np.isfinite(value)):
            raise ValueError(f'Invalid converted field: {key}')
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix('.pkl.tmp')
    with tmp.open('wb') as stream:
        pickle.dump(converted, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, output)
    print(f'Saved: {output}')
    print(f'{len(data["root_pos"])} frames at {data["fps"]} FPS -> {count} frames at {args.fps:g} FPS')
    info = converted['meta']['resampling']
    print(f'Duration: {info["source_duration_s"]:.6f}s -> {info["output_duration_s"]:.6f}s (uniform 50 Hz grid for --fps 50)')
    print('Body poses recomputed; object scale and contact labels preserved.')
    print('This is a HOI-format pickle. Holosoma NPZ export is a separate step.')


if __name__ == '__main__':
    main()
