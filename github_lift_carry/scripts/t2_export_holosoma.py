#!/usr/bin/env python3
"""Export reviewed 50 FPS HOI motions using Holosoma G1 models and scaled assets."""
import argparse
import copy
import hashlib
import json
import os
import pickle
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def numbers(text, default):
    return np.array([float(x) for x in (text or default).split()])


def vector_text(value):
    return ' '.join(f'{x:.12g}' for x in value)


def asset_copy(source, out):
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    name = digest + '_' + source.name
    target = out / 'assets' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)
    return name


def uri_path(uri, parent):
    if '://' in uri:
        raise ValueError(f'Unsupported asset URI: {uri}. Resolve it to a file first.')
    p = Path(uri)
    return p if p.is_absolute() else parent / p


def make_model(package, data, obj, out):
    original = package / f'models/g1/g1_29dof_w_{obj}.xml'
    fallback = not original.is_file()
    if fallback:
        original = package / 'models/g1/g1_29dof_w_largebox.xml'
    tree = ET.parse(original)
    root = tree.getroot()
    if fallback:
        placeholder = root.find('.//body[@name="largebox_link"]')
        if placeholder is None:
            raise ValueError('Largebox template has no object placeholder.')
        placeholder.set('name', obj + '_link')
        for geom in placeholder.findall('geom'):
            if geom.get('name') == 'largebox':
                geom.set('name', obj)
    if root.findall('.//include'):
        raise ValueError(f'Model includes require expansion before export: {original}')
    compiler = root.find('compiler')
    if compiler is None:
        compiler = ET.SubElement(root, 'compiler')
    meshdir = original.parent / compiler.get('meshdir', compiler.get('assetdir', ''))
    texturedir = original.parent / compiler.get('texturedir', compiler.get('assetdir', ''))
    compiler.attrib.pop('assetdir', None)
    compiler.set('meshdir', 'assets/')
    compiler.set('texturedir', 'assets/')
    compiler.set('strippath', 'false')

    urdf_path = Path(data['object_model_path']).expanduser().resolve()
    if urdf_path.suffix.lower() != '.urdf':
        raise ValueError('This exporter requires the recorded object URDF.')
    urdf = ET.parse(urdf_path)
    uroot = urdf.getroot()
    visuals = uroot.findall('.//visual')
    if len(uroot.findall('link')) != 1 or len(visuals) != 1:
        raise ValueError(f'Expected a single-link object with one visual: {urdf_path}')
    visual = visuals[0]
    umesh = visual.find('geometry/mesh')
    if umesh is None:
        raise ValueError('Object visual must be a mesh.')
    if not np.isclose(float(data.get('object_aug_scale', 1)), 1):
        raise ValueError('Augmented object scaling needs a separate verified mapping.')
    scale = float(data.get('object_scale', 1)) if data.get('scale_object_mesh', True) else 1.0
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid object scale.')
    body = root.find(f'.//body[@name="{obj}_link"]')
    if body is None or body.findall('body'):
        raise ValueError(f'Expected a simple free object body named {obj}_link.')
    geoms = body.findall('geom')
    if not geoms or any(g.get('mesh') is None for g in geoms):
        raise ValueError('Object body must use mesh geoms.')
    mesh_names = {g.get('mesh') for g in geoms}
    meshes = {m.get('name'): m for m in root.findall('.//asset/mesh')}
    origin = visual.find('origin')
    xyz = numbers(origin.get('xyz') if origin is not None else None, '0 0 0') * scale
    rpy = numbers(origin.get('rpy') if origin is not None else None, '0 0 0')
    quat = Rotation.from_euler('xyz', rpy).as_quat()[[3, 0, 1, 2]]
    object_mesh = uri_path(umesh.get('filename'), urdf_path.parent)
    object_asset = asset_copy(object_mesh, out)
    for name in mesh_names:
        mesh = meshes[name]
        mesh.set('file', object_asset)
        mesh.set('scale', vector_text(numbers(umesh.get('scale'), '1 1 1') * scale))
        # Do not apply a second transform on the mesh asset.
        mesh.attrib.pop('refpos', None)
        mesh.attrib.pop('refquat', None)
    for g in geoms:
        for field in ('euler', 'axisangle', 'xyaxes', 'zaxis'):
            g.attrib.pop(field, None)
        g.set('pos', vector_text(xyz))
        g.set('quat', vector_text(quat))
    inertial = body.find('inertial')
    if inertial is not None:
        inertial.set('pos', vector_text(numbers(inertial.get('pos'), '0 0 0') * scale))
        for key in ('diaginertia', 'fullinertia'):
            if key in inertial.attrib:
                inertial.set(key, vector_text(numbers(inertial.get(key), '') * scale ** 2))
    # Copy all robot meshes/textures too: generated MJCFs use relative assets.
    for asset in root.findall('.//asset/*'):
        if 'file' not in asset.attrib or (asset.tag == 'mesh' and asset.get('name') in mesh_names):
            continue
        base = meshdir if asset.tag in ('mesh', 'skin') else texturedir
        asset.set('file', asset_copy(uri_path(asset.get('file'), base), out))
    for mesh in uroot.findall('.//mesh'):
        mesh.set('filename', 'assets/' + asset_copy(uri_path(mesh.get('filename'), urdf_path.parent), out))
        mesh.set('scale', vector_text(numbers(mesh.get('scale'), '1 1 1') * scale))
    for node in uroot.findall('.//origin'):
        if 'xyz' in node.attrib:
            node.set('xyz', vector_text(numbers(node.get('xyz'), '0 0 0') * scale))
    for inertia in uroot.findall('.//inertia'):
        for key, value in list(inertia.attrib.items()):
            inertia.set(key, str(float(value) * scale ** 2))
    slug = f'{obj}_scale{scale:.8g}'
    model_path = out / f'{slug}.xml'
    object_urdf = out / f'{slug}.urdf'
    tree.write(model_path, encoding='utf-8', xml_declaration=True)
    urdf.write(object_urdf, encoding='utf-8', xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    return model, model_path, object_urdf, scale


def angular_velocity(q, fps):
    n, bodies, _ = q.shape
    output = np.zeros((n, bodies, 3))
    for i in range(n):
        lo, hi = max(0, i - 1), min(n - 1, i + 1)
        before = Rotation.from_quat(q[lo, :, [1, 2, 3, 0]].T)
        after = Rotation.from_quat(q[hi, :, [1, 2, 3, 0]].T)
        output[i] = (after * before.inv()).as_rotvec() / ((hi - lo) / fps)
    return output


def export_motion(data, model, obj):
    fps = float(data['fps'])
    n = len(data['root_pos'])
    if fps != 50 or n < 2:
        raise ValueError('Expected at least two frames at 50 FPS.')
    frees = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if len(frees) != 2 or model.nq != 43 or model.nv != 41:
        raise ValueError('Expected G1 29 DOF plus robot and object free joints (nq=43,nv=41).')
    robot, object_joint = frees
    object_body = int(model.jnt_bodyid[object_joint])
    if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, object_body) != obj + '_link':
        raise ValueError('Object free joint is not on the expected body.')
    if model.jnt_qposadr[robot] != 0 or model.jnt_qposadr[object_joint] != 36 or model.jnt_dofadr[object_joint] != 35:
        raise ValueError('Unexpected free-joint ordering.')
    source_names = data['dof_names']
    mapping = source_names if isinstance(source_names, dict) else {name: i for i, name in enumerate(source_names)}
    joint_ids = [j for j in range(model.njnt) if j not in frees]
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joint_ids]
    if set(names) != set(mapping) or len(names) != 29:
        raise ValueError('HOI and Holosoma joint names do not match.')
    qpos = np.tile(model.qpos0, (n, 1))
    qpos[:, :3], qpos[:, 3:7] = data['root_pos'], data['root_rot']
    qpos[:, -7:-4], qpos[:, -4:] = data['object_pos'], data['object_rot']
    for j, name in zip(joint_ids, names):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError('Expected scalar hinge joints.')
        qpos[:, model.jnt_qposadr[j]] = data['dof_pos'][:, int(mapping[name])]
    if not np.isfinite(qpos).all():
        raise ValueError('Nonfinite pose values.')
    for pose in qpos:
        mujoco.mj_normalizeQuat(model, pose)
    qvel = np.zeros((n, model.nv))
    for i in range(n):
        lo, hi = max(0, i - 1), min(n - 1, i + 1)
        mujoco.mj_differentiatePos(model, qvel[i], (hi - lo) / fps, qpos[lo], qpos[hi])
    md = mujoco.MjData(model)
    positions = np.empty((n, model.nbody, 3))
    quats = np.empty((n, model.nbody, 4))
    for i in range(n):
        md.qpos[:] = qpos[i]
        mujoco.mj_kinematics(model, md)
        positions[i], quats[i] = md.xpos.copy(), md.xquat.copy()
    linear = np.gradient(positions, 1 / fps, axis=0, edge_order=2 if n > 2 else 1)
    angular = angular_velocity(quats, fps)
    result = dict(fps=np.array([50], dtype=np.int64), joint_pos=qpos[:, :-7], joint_vel=qvel[:, :-6],
                  body_pos_w=positions, body_quat_w=quats, body_lin_vel_w=linear, body_ang_vel_w=angular,
                  object_pos_w=qpos[:, -7:-4], object_quat_w=qpos[:, -4:],
                  object_lin_vel_w=linear[:, object_body], object_ang_vel_w=angular[:, object_body],
                  joint_names=np.array(names), body_names=np.array([
                      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f'body_{b}' for b in range(model.nbody)]),
                  qpos=qpos, object_scale=np.array(float(data.get('object_scale', 1))),
                  quaternion_order=np.array('wxyz'))
    for key in ('object_contact_sequence', 'object_ground_contact_sequence', 'robot_foot_ground_contact_flags',
                'per_link_contact_flags', 'contact_link_names', 'fixed_contact_points_per_frame_in_object_frame'):
        if key in data:
            result[key] = np.asarray(data[key])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.home() / 'holosoma/t2')
    parser.add_argument('--package', type=Path, default=Path.home() / 'holosoma/src/holosoma_retargeting/holosoma_retargeting')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    out = args.out.expanduser().resolve() if args.out else root / 'holosoma_npz32'
    if out.exists():
        parser.error(f'{out} exists. Use a fresh --out to protect the previous export.')
    manifest = json.loads((root / 'selected40_manifest.json').read_text())
    rejected = set(json.loads((root / 't2_rejections.json').read_text()))
    if not rejected.issubset({c['id'] for c in manifest['clips']}):
        parser.error('Rejected IDs do not match the selection.')
    clips = [c for c in manifest['clips'] if c['id'] not in rejected]
    if not clips:
        parser.error('No retained clips.')
    motions = []
    for clip in clips:
        source = root / 'hoi_selected40' / clip['task'] / clip['id'] / 'contact_window_fps50.pkl'
        with source.open('rb') as f:
            data = pickle.load(f)
        template = args.package / f'models/g1/g1_29dof_w_{clip["object"]}.xml'
        if not template.is_file():
            template = args.package / 'models/g1/g1_29dof_w_largebox.xml'
        if not template.is_file():
            parser.error(f'Missing Holosoma model: {template}')
        if float(data['fps']) != 50:
            parser.error(f'Not 50 FPS: {source}')
        motions.append((clip, source, data))
    out.mkdir(parents=True)
    models_dir = out / 'models'
    models_dir.mkdir()
    records = []
    failures = []
    for i, (clip, source, data) in enumerate(motions, 1):
        try:
            model, xml, urdf, scale = make_model(args.package, data, clip['object'], models_dir)
            arrays = export_motion(data, model, clip['object'])
            target = out / 'motions' / clip['task'] / (clip['id'] + '.npz')
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **arrays)
            with np.load(target, allow_pickle=False) as saved:
                assert saved['joint_pos'].shape == (len(data['root_pos']), 36)
                assert saved['joint_vel'].shape == (len(data['root_pos']), 35)
                assert saved['fps'].item() == 50
            record = dict(clip, t2_review='keep', fps=50, frames=len(data['root_pos']),
                          npz=str(target.relative_to(out)), model_xml=str(xml.relative_to(out)),
                          object_urdf=str(urdf.relative_to(out)), object_geometry_scale=scale,
                          source_pickle=str(source), source_pickle_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            records.append(record)
            print(f'[{i}/{len(clips)}] OK {clip["id"]}', flush=True)
        except Exception as exc:
            failures.append({'id': clip['id'], 'error': f'{type(exc).__name__}: {exc}'})
            print(f'[{i}/{len(clips)}] FAIL {clip["id"]}: {exc}', flush=True)
        (out / 'manifest.json').write_text(json.dumps({'format': 'Holosoma converter field layout plus contact metadata',
            'quaternion_order': 'wxyz', 'fps': 50, 'units': {'position': 'm', 'angle': 'rad'},
            'counts': dict(Counter(c['task'] for c in records)), 'clips': records, 'failures': failures}, indent=2) + '\n')
    shutil.copy2(__file__, out / 't2_export_holosoma.py')
    (out / 'README.md').write_text('50 FPS reference motions exported with Holosoma G1 joint/body layout.\n'
        'Use each clip\'s model_xml/object_urdf from manifest.json for matching scaled object geometry.\n'
        'Robot and object poses were preserved, joints mapped by name, body poses recomputed in MuJoCo.\n'
        'Velocities use finite differences; contact labels come from HOI-Retarget.\n'
        'Object masses are the Holosoma template values, not measured real-object masses.\n'
        'This export does not perform physics simulation or certify task success.\n'
        'Replay every exported clip and check the T3/T4 loader and asset configuration before training.\n')
    print(f'Exported {len(records)}/{len(clips)} clips. Folder: {out}', flush=True)
    if failures:
        raise SystemExit('Some exports failed. See manifest.json; do not package as complete.')


if __name__ == '__main__':
    main()
