#!/usr/bin/env python3
"""Resolve portable carry source/mesh paths before invoking local retargeters."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest',type=Path,default=Path(__file__).resolve().parents[1]/'t1/manifest.json')
p.add_argument('--out',type=Path,default=Path.home()/'Downloads/t1_carry_manifest_local.json')
a=p.parse_args();src=a.manifest.expanduser().resolve();data=json.loads(src.read_text())
for c in data['clips']:
    if c['task']!='carry':raise ValueError('Expected carry only.')
    c['source']=str((src.parent/c['source']).resolve())
    if not Path(c['source']).is_file():raise FileNotFoundError(c['source'])
for m in data['meshes'].values():m['path']=str((src.parent/m['path']).resolve())
data['data']=str(src.parent/'source_recordings')
out=a.out.expanduser();out.parent.mkdir(parents=True,exist_ok=True)
if out.exists():raise FileExistsError(f'{out} exists; choose a fresh --out.')
out.write_text(json.dumps(data,indent=2)+'\n');print('Local retargeting manifest:',out)
