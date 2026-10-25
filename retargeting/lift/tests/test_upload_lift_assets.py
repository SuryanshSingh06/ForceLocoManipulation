import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'scripts/upload_lift_assets.py'
spec = importlib.util.spec_from_file_location('upload', MODULE)
upload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upload)


class UploadSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.batch = self.root / 'batch'
        self.npz = self.root / 'exports'
        self.cid = 'sub1_smallbox_001_t1_lift_000000_000030'
        self.clip = dict(id=self.cid, task='lift', object='smallbox', review='accepted')
        self.folder = self.batch / 'lift' / self.cid
        self.folder.mkdir(parents=True)
        for name in ('contact_window.pkl', 'retarget.log'):
            (self.folder / name).write_bytes(b'fixture')
        (self.folder / 'status.json').write_text(json.dumps({'status': 'completed'}))
        (self.folder / 'selection.json').write_text(json.dumps(self.clip))
        (self.batch / 'passed_lift_manifest.json').write_text(json.dumps({'clips': [self.clip]}))
        model = self.npz / 'models' / self.cid
        (model / 'assets').mkdir(parents=True)
        (model / 'assets/box.obj').write_text('v 0 0 0\n')
        (model / 'model.xml').write_text('<mujoco><compiler meshdir="assets"/><asset><mesh name="box" file="box.obj"/></asset></mujoco>')
        (model / 'object.urdf').write_text('<robot name="box"><link name="box"><collision><geometry><mesh filename="assets/box.obj"/></geometry></collision></link></robot>')
        (self.npz / 'motion.npz').write_bytes(b'fixture NPZ bytes; collector does not deserialize')
        self.record = dict(self.clip, fps=50, npz='motion.npz', npz_sha256=upload.sha256(self.npz / 'motion.npz'),
                           model_xml=f'models/{self.cid}/model.xml', object_urdf=f'models/{self.cid}/object.urdf')
        (self.npz / 'manifest.json').write_text(json.dumps({'clips': [self.record], 'failures': []}))
        self.args = argparse.Namespace(batch=self.batch, selection=None, expected_count=1, npz_root=self.npz,
                                       holosoma_root=self.root, video_dir=None)

    def test_only_manifest_selection_is_collected(self):
        # A rejected/attempted folder must not leak into the published selection.
        extra = self.batch / 'lift/unselected_clip'
        extra.mkdir(); (extra / 'contact_window.pkl').write_bytes(b'unselected')
        carry = self.batch / 'carry/other_action'
        carry.mkdir(parents=True); (carry / 'contact_window.pkl').write_bytes(b'carry')
        out = self.root / 'payload'
        report = upload.collect(self.args, out)
        self.assertEqual(report['clip_count'], 1)
        self.assertEqual([p.name for p in (out / 'clips').iterdir()], [self.cid])
        self.assertFalse(any('unselected' in x['path'] or 'carry' in x['path'] for x in report['files']))
        for row in report['files']:
            self.assertEqual(upload.sha256(out / row['path']), row['sha256'])

    def test_mixed_actions_rejected(self):
        (self.batch / 'passed_lift_manifest.json').write_text(json.dumps({'clips': [self.clip, dict(self.clip, id='carry_clip', task='carry')]}))
        with self.assertRaises(ValueError):upload.plan(self.args)

    def test_rejected_status_rejected(self):
        (self.folder / 'status.json').write_text(json.dumps({'status': 'quality_rejected'}))
        with self.assertRaises(ValueError):upload.plan(self.args)

    def test_wrong_count_rejected(self):
        self.args.expected_count = 24
        with self.assertRaises(ValueError):upload.plan(self.args)

    def test_no_overwrite(self):
        out = self.root / 'existing'; out.mkdir(); (out / 'keep').write_text('keep')
        with self.assertRaises(FileExistsError):upload.collect(self.args, out)
        self.assertEqual((out / 'keep').read_text(), 'keep')

    def test_npz_checksum_mismatch_rejected(self):
        (self.npz / 'motion.npz').write_bytes(b'changed')
        with self.assertRaises(ValueError):upload.plan(self.args)

    def test_nonportable_mesh_rejected(self):
        p = self.npz / self.record['object_urdf']
        p.write_text('<robot name="box"><link name="box"><collision><geometry><mesh filename="/etc/hosts"/></geometry></collision></link></robot>')
        with self.assertRaises(ValueError):upload.plan(self.args)

    def test_path_traversal_rejected(self):
        self.clip['id'] = '../escape'
        (self.batch / 'passed_lift_manifest.json').write_text(json.dumps({'clips': [self.clip]}))
        with self.assertRaises(ValueError):upload.plan(self.args)

    def test_missing_required_output_leaves_no_payload(self):
        (self.folder / 'contact_window.pkl').unlink()
        out = self.root / 'payload'
        with self.assertRaises(FileNotFoundError):upload.collect(self.args, out)
        self.assertFalse(out.exists())


if __name__ == '__main__':unittest.main()
