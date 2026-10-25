#!/usr/bin/env python3
"""Collect only a passed lift selection; optionally publish from an isolated Git worktree.

Default: collect files locally. --push explicitly commits and pushes a new branch.
No source .pt recordings, SMPL body models, checkpoints, or other action classes.
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
import xml.etree.ElementTree as ET

REPOSITORY = 'SuryanshSingh06/ForceLocoManipulation'
LIFT = Path('retargeting/lift')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text())


def inside(root, path):
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Asset escapes its expected directory: {path}')
    return path


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError(f'Invalid clip ID: {value!r}')
    return value


def check_model(path, model_dir):
    tree = ET.parse(path)
    compiler = tree.find('compiler')
    meshdir = compiler.get('meshdir', '') if compiler is not None else ''
    for node in tree.findall('.//mesh'):
        # MuJoCo mesh references under geoms use a name, not a file.
        name = node.get('filename') if path.suffix == '.urdf' else node.get('file')
        if not name:
            continue
        if Path(name).is_absolute() or '://' in name:
            raise ValueError(f'Nonportable mesh path in {path}: {name}')
        target = inside(model_dir, path.parent / (meshdir if path.suffix == '.xml' else '') / name)
        if not target.is_file():
            raise FileNotFoundError(target)


def plan(args):
    batch = args.batch.resolve()
    selection = args.selection.resolve() if args.selection else batch / 'passed_lift_manifest.json'
    manifest = load(selection)
    clips = manifest['clips'] if isinstance(manifest, dict) else manifest
    if not clips or any(c.get('task') != 'lift' for c in clips):
        raise ValueError('Selection must be nonempty and contain only lift clips.')
    ids = [safe_id(c['id']) for c in clips]
    if len(ids) != len(set(ids)):
        raise ValueError('Selection contains duplicate clip IDs.')
    if len(ids) != args.expected_count:
        raise ValueError(f'Found {len(ids)} selected clips, expected {args.expected_count}. No files copied.')
    exported = load(args.npz_root / 'manifest.json')
    if exported.get('failures'):
        raise ValueError('NPZ export reports failures. Finish conversion before publishing.')
    by_id = {c['id']: c for c in exported['clips'] if c.get('task') == 'lift'}
    files = {}
    missing_optional = []
    def add(source, destination, required=True):
        source = Path(source)
        if not source.is_file():
            if required:
                raise FileNotFoundError(source)
            missing_optional.append(str(source))
            return
        if source.is_symlink():
            raise ValueError(f'Symlink not allowed in publication: {source}')
        if destination in files and files[destination] != source:
            raise ValueError(f'Duplicate destination: {destination}')
        files[destination] = source
    clip_files = ['contact_window.pkl', 'contact_window_fps50.pkl', 'kinematic_window.pkl',
                  'status.json', 'selection.json', 'retarget.log', 'source_quality.json',
                  'source_quality.png', 'source_curves.csv', 'retargeted_quality.json',
                  'retargeted_quality.png', 'retargeted_curves.csv']
    for clip in clips:
        cid = clip['id']
        folder = inside(batch, batch / 'lift' / cid)
        status = load(folder / 'status.json')
        if status.get('status') != 'completed':
            raise ValueError(f'{cid}: status is {status.get("status")!r}, not completed.')
        if clip.get('review') == 'rejected':
            raise ValueError(f'{cid}: selection contains a manually rejected clip.')
        for name in clip_files:
            add(folder / name, Path('clips') / cid / name,
                required=name in {'contact_window.pkl', 'status.json', 'selection.json', 'retarget.log'})
        for pattern in ('*.mp4',):
            for video in folder.glob(pattern):
                if '.partial.' not in video.name:
                    add(video, Path('clips') / cid / video.name)
        if args.video_dir:
            add(args.video_dir / (cid + '.mp4'), Path('videos') / (cid + '.mp4'), required=False)
        if cid not in by_id:
            raise ValueError(f'{cid}: missing from NPZ export manifest.')
        record = by_id[cid]
        if float(record.get('fps', 0)) != 50:
            raise ValueError(f'{cid}: exported metadata does not declare 50 FPS.')
        npz = inside(args.npz_root, args.npz_root / record['npz'])
        if record.get('npz_sha256') and sha256(npz) != record['npz_sha256']:
            raise ValueError(f'{cid}: exported NPZ checksum mismatch.')
        add(npz, Path('motions') / (cid + '.npz'))
        model = inside(args.npz_root, args.npz_root / record['model_xml'])
        urdf = inside(args.npz_root, args.npz_root / record['object_urdf'])
        model_dir = model.parent
        if not urdf.is_relative_to(model_dir):
            raise ValueError('Object URDF must be in its matching clip model directory.')
        check_model(model, model_dir)
        check_model(urdf, model_dir)
        for asset in model_dir.rglob('*'):
            if asset.is_symlink():
                raise ValueError(f'Symlink not allowed: {asset}')
            if asset.is_file() and asset.suffix.lower() in {'.xml', '.urdf', '.obj', '.stl', '.dae', '.ply', '.mtl', '.png', '.jpg', '.jpeg', '.json'}:
                add(asset, Path('models') / cid / asset.relative_to(model_dir))
    # Capture the actual local scripts, which may differ from the repository snapshots.
    for name in ['t2_batch_retarget.py', 't2_resample_hoi.py', 't2_export_holosoma.py', 't2_palm_geometry_qa.py']:
        add(args.holosoma_root / 't2' / name, Path('local_scripts') / name, required=False)
    add(args.holosoma_root / 't1/t1_select.py', Path('local_scripts/t1_select.py'), required=False)
    selection_lift = dict(manifest) if isinstance(manifest, dict) else {}
    selection_lift['clips'] = clips
    records_lift = {**exported, 'clips': [by_id[cid] for cid in ids], 'failures': []}
    # Original paths retained as provenance; portable payload paths are listed separately.
    return files, selection_lift, records_lift, missing_optional


def collect(args, output):
    files, selection, exported, missing = plan(args)
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}. Use a fresh --out or upload branch.')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.lift_collect_', dir=output.parent))
    try:
        inventory = []
        for destination, source in sorted(files.items(), key=lambda item: str(item[0])):
            target = temporary / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            value = sha256(target)
            if value != sha256(source):
                raise RuntimeError(f'Source changed during collection: {source}')
            inventory.append({'path': destination.as_posix(), 'bytes': target.stat().st_size, 'sha256': value})
        (temporary / 'selected_lift_manifest.json').write_text(json.dumps(selection, indent=2) + '\n')
        (temporary / 'source_export_manifest.json').write_text(json.dumps(exported, indent=2) + '\n')
        report = dict(created_at=datetime.now(timezone.utc).isoformat(), task='lift',
                      clip_count=len(selection['clips']), object_counts=dict(Counter(c.get('object', 'unknown') for c in selection['clips'])),
                      contact_validation='Contact outputs collected; geometry and physical grasp validation are not implied.',
                      files=inventory, missing_optional=missing,
                      omitted=['source .pt recordings', 'SMPL/SMPL-X body models', 'policy checkpoints', 'other action classes'])
        (temporary / 'upload_inventory.json').write_text(json.dumps(report, indent=2) + '\n')
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f'Collected {report["clip_count"]} lift clips, {len(files)} files into {output}', flush=True)
    print('Object counts:', report['object_counts'], flush=True)
    return report


def git(repo, *args, capture=False):
    result = subprocess.run(['git', '-C', str(repo), *args], check=True,
                            text=True, stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else None


def publish(args):
    repo = args.repo.resolve()
    remote = git(repo, 'remote', 'get-url', 'origin', capture=True)
    if remote.removesuffix('.git').rstrip('/') not in {
        f'https://github.com/{REPOSITORY}', f'git@github.com:{REPOSITORY}'}:
        raise ValueError(f'Unexpected origin: {remote}')
    git(repo, 'lfs', 'version')
    # Validate inputs before creating a branch or worktree.
    plan(args)
    branch = args.branch or 'lift-retargeting-assets-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    subprocess.run(['git', 'check-ref-format', '--branch', branch], check=True, stdout=subprocess.DEVNULL)
    git(repo, 'fetch', 'origin', 'main')
    base = git(repo, 'rev-parse', 'HEAD', capture=True)
    changed = git(repo, 'diff', '--name-only', 'origin/main', base, capture=True).splitlines()
    if any(name != 'README.md' and not name.startswith(str(LIFT) + '/') for name in changed):
        raise ValueError('The checkout has unpublished changes outside the lift handoff. Use a clean dedicated clone.')
    parent = Path(tempfile.mkdtemp(prefix='lift_upload_'))
    worktree = parent / 'checkout'
    git(repo, 'worktree', 'add', '-b', branch, str(worktree), base)
    print('Isolated upload worktree:', worktree, flush=True)
    # Keep the worktree on failure for inspection and retry; never force-push.
    target = worktree / LIFT / 'artifacts' / args.dataset_name
    collect(args, target)
    git(worktree, 'lfs', 'install', '--local')
    attributes = worktree / LIFT / '.gitattributes'
    if not attributes.is_file():
        raise FileNotFoundError('Update the clone first: lift .gitattributes is missing.')
    git(worktree, 'add', '-f', '--', str(target.relative_to(worktree)))
    git(worktree, 'commit', '-m', f'Add selected lift retargeting artifacts ({args.expected_count} clips)')
    git(worktree, 'push', '-u', 'origin', branch)
    print(f'Published: https://github.com/{REPOSITORY}/tree/{branch}/{LIFT}/artifacts/{args.dataset_name}')
    print('The upload is on its own branch. It does not replace main or an existing dataset.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--holosoma-root', type=Path, default=Path.home() / 'holosoma')
    parser.add_argument('--batch', type=Path)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--npz-root', type=Path)
    parser.add_argument('--video-dir', type=Path)
    parser.add_argument('--expected-count', type=int, default=24)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--out', type=Path)
    parser.add_argument('--dataset-name', default='selected24')
    parser.add_argument('--push', action='store_true', help='Collect, commit and push on a new isolated branch; requires Git LFS and git authentication.')
    parser.add_argument('--branch')
    args = parser.parse_args()
    args.holosoma_root = args.holosoma_root.expanduser().resolve()
    args.batch = (args.batch or args.holosoma_root / 't2/hoi_all_lift_candidates01').expanduser().resolve()
    args.npz_root = (args.npz_root or args.holosoma_root / 't2/lift24_npz_contactqa').expanduser().resolve()
    if args.selection: args.selection = args.selection.expanduser().resolve()
    if args.video_dir: args.video_dir = args.video_dir.expanduser().resolve()
    args.repo = args.repo.expanduser().resolve()
    safe_id(args.dataset_name)
    if args.expected_count < 1: parser.error('--expected-count must be positive.')
    if args.push and args.out: parser.error('--out is for local collection; --push uses an isolated worktree.')
    try:
        if args.push:
            publish(args)
        else:
            output = (args.out or args.repo / LIFT / 'artifacts' / args.dataset_name).expanduser().resolve()
            collect(args, output)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
