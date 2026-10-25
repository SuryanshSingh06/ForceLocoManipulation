#!/usr/bin/env python3
"""Audit trusted, locally generated HOI lift outputs. Pickle loading executes code.
Run in hoi-retarget and only point --batch at your own generated results.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import re


def audit(batch):
    manifest = json.loads((batch / 'passed_lift_manifest.json').read_text())
    clips = manifest['clips']
    if not clips or any(c.get('task') != 'lift' for c in clips):
        raise ValueError('Expected a nonempty lift-only passed manifest.')
    seen = set()
    rows = []
    for clip in clips:
        cid = clip['id']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', cid) or cid in seen:
            raise ValueError('Unsafe or duplicate clip ID.')
        seen.add(cid)
        folder = batch / 'lift' / cid
        row = dict(id=cid, object=clip.get('object'), verified=False, warnings=[])
        try:
            with (folder / 'contact_window.pkl').open('rb') as f:
                data = pickle.load(f)
            meta = data.get('meta', {})
            row.update(mode=meta.get('projection_mode'), stage=meta.get('stage'), fps=float(data['fps']))
            status = json.loads((folder / 'status.json').read_text())
            row['batch_status'] = status.get('status')
            command = status.get('command', [])
            mode = None
            if isinstance(command, list) and '--mode' in command:
                index = command.index('--mode')
                mode = command[index + 1] if index + 1 < len(command) else 'MISSING_VALUE'
            row['command_mode'] = mode
            log = folder / 'retarget.log'
            failures = re.findall(r'Unified-window pass complete:.*?(\d+) failed', log.read_text(errors='replace')) if log.is_file() else []
            row['logged_failed_windows'] = [int(n) for n in failures]
            if not failures:row['warnings'].append('No window-completion record found in log.')
            if mode is None:row['warnings'].append('No explicit command mode recorded in status.json.')
            row['verified'] = (row['mode'] == 'contact' and row['stage'] == 'contact_window'
                               and row['batch_status'] == 'completed'
                               and mode in (None, 'contact') and not any(int(n) for n in failures))
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
        rows.append(row)
    return dict(clip_count=len(rows), object_counts=dict(Counter(c.get('object', 'unknown') for c in clips)),
                verified_contact=sum(r['verified'] for r in rows), clips=rows,
                limitation='Contact-mode provenance is not proof of geometric contact or physical support.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, default=Path.home() / 'holosoma/t2/hoi_all_lift_candidates01')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    report = audit(args.batch.expanduser().resolve())
    for row in report['clips']:
        print(('VERIFIED' if row['verified'] else 'CHECK'), row['id'], row.get('error', ''))
    print(f"Contact-mode metadata verified: {report['verified_contact']}/{report['clip_count']}")
    print('Object counts:', report['object_counts'])
    if args.out:
        args.out.expanduser().write_text(json.dumps(report, indent=2) + '\n')
    raise SystemExit(0 if report['verified_contact'] == report['clip_count'] else 1)


if __name__ == '__main__':main()
