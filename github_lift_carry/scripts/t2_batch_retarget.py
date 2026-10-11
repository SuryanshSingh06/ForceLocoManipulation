#!/usr/bin/env python3
"""Resume HOI-Retarget contact projection of non-rejected T1 crops."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path.home() / 'holosoma/t1/run05/manifest.json')
    parser.add_argument('--repo', type=Path, default=Path.home() / 'hoi-retarget')
    parser.add_argument('--out', type=Path, default=Path.home() / 'holosoma/t2/hoi_batch01')
    args = parser.parse_args()
    args.manifest = args.manifest.expanduser().resolve()
    args.repo = args.repo.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    exe = shutil.which('hoi-retarget')
    if exe is None:
        parser.error('Activate the hoi-retarget conda environment first.')
    if not args.repo.is_dir():
        parser.error(f'Repository not found: {args.repo}')
    manifest = json.loads(args.manifest.read_text())
    clips = [c for c in manifest['clips'] if c['review'] != 'rejected']
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
    print(f'{len(clips)} clips selected; {len(manifest["clips"]) - len(clips)} rejected clips skipped.', flush=True)
    print(f'Outputs: {args.out}', flush=True)
    print('Contact mode, G1, 30 FPS, no videos. Ctrl+C stops; rerun to resume.', flush=True)
    counts = {'completed': 0, 'resumed': 0, 'failed': 0}
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
                        'mode': 'contact', 'fps': 30, 'repo': str(args.repo), 'runner_version': 1}
            record['settings'] = settings
            old = json.loads(state_path.read_text()) if state_path.exists() else None
            outputs = [folder / 'kinematic_window.pkl', folder / 'contact_window.pkl']
            if old and old.get('settings') and old['settings'] != settings:
                raise ValueError('Settings changed. Use a fresh --out directory to protect old results.')
            if old and old.get('status') == 'completed' and all(p.is_file() and p.stat().st_size > 0 for p in outputs):
                counts['resumed'] += 1
                record['status'] = 'completed'
                record['resumed'] = True
                results.append(record)
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
            print('\nStopped. Run the same command to resume.', flush=True)
            return 130
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
    save_json(args.out / 'summary.json', {'counts': counts, 'results': results})
    print(f'Finished: {counts}. Summary: {args.out / "summary.json"}', flush=True)
    return 1 if counts['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
