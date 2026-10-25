import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

spec=importlib.util.spec_from_file_location('batch',Path(__file__).with_name('t2_holosoma_lift_batch.py'))
batch=importlib.util.module_from_spec(spec);spec.loader.exec_module(batch)

class Tests(unittest.TestCase):
    def test_pending_included_and_rejections_excluded(self):
        clips=[dict(id=f'clip{i}',task='lift',object='smallbox',source='a.pt',start=0,end=5,review='pending') for i in range(24)]
        clips += [dict(clips[0],id='rejected',review='rejected'),dict(clips[0],id='carry',task='carry')]
        selected={'clips':clips[:24]}
        all_lifts,picked=batch.selections({'clips':clips},selected)
        self.assertEqual(len(all_lifts),24);self.assertEqual(len(picked),24)
        self.assertEqual(len(batch.selections({'clips':clips},selected,True)[0]),25)

    def test_same_selected_ids_and_ranges_required(self):
        clip=dict(id='clip',task='lift',object='smallbox',source='a.pt',start=0,end=5,review='accepted')
        with self.assertRaises(ValueError):batch.selections({'clips':[clip]},{'clips':[dict(clip,end=6)]},expected=1)
        with self.assertRaises(ValueError):batch.selections({'clips':[dict(clip,id='../bad')]},{'clips':[clip]},expected=1)

    def test_resampling_both_quaternions_and_same_duration_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'raw.npz';target=Path(tmp)/'50.npz'
            q=np.zeros((41,43));q[:,0]=np.linspace(0,1,41);q[:,36]=np.linspace(1,2,41)
            angles=np.linspace(0,1,41);rot=np.column_stack([np.cos(angles/2),np.zeros(41),np.zeros(41),np.sin(angles/2)])
            q[:,3:7]=rot;q[:,39:43]=rot
            np.savez(source,qpos=q,fps=30,human_joints=np.zeros((41,52,3)),cost=1.)
            batch.resample50(source,target)
            with np.load(target,allow_pickle=False) as d:
                self.assertEqual(d['qpos'].shape,(67,43));self.assertEqual(d['human_joints'].shape,(67,52,3))
                self.assertAlmostEqual(d['qpos'][-1,0],.99)
                self.assertTrue(np.allclose(np.linalg.norm(d['qpos'][:,3:7],axis=1),1))
                self.assertTrue(np.allclose(d['qpos'][:,3:7],d['qpos'][:,39:43]))
            self.assertEqual(batch.verify_motion(source,41)['fps'],30)
            with self.assertRaises(ValueError):batch.verify_motion(source,40)

    def test_missing_model_adapts_only_private_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            models=Path(tmp)/'models';(models/'g1').mkdir(parents=True);(models/'smallbox').mkdir()
            template=models/'g1/g1_29dof_w_largebox.xml'
            text='<mujoco><compiler meshdir="assets"/><asset><mesh name="robot" file="robot.obj"/><mesh name="largebox_mesh" file="largebox.obj"/></asset><worldbody><body name="pelvis"><freejoint/><geom mesh="robot"/></body><body name="largebox_link"><freejoint/><inertial mass="0.1" pos="0 0 0" diaginertia=".002 .002 .002"/><geom name="largebox" mesh="largebox_mesh"/></body></worldbody></mujoco>'
            template.write_text(text)
            (models/'smallbox/smallbox.obj').write_text('v 0 0 0\n')
            (models/'smallbox/smallbox.urdf').write_text('<robot name="smallbox"><link name="box"><visual><geometry><mesh filename="smallbox.obj"/></geometry></visual></link></robot>')
            target=batch.make_object_model(models,'smallbox')
            self.assertEqual(template.read_text(),text)
            self.assertIn('smallbox_link',target.read_text());self.assertNotIn('name="largebox_link"',target.read_text())
            self.assertIn('name="robot"',target.read_text())
            self.assertEqual(batch.make_object_model(models,'smallbox'),target)

    def test_nested_urdf_references_differently_named_obj(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);models=root/'models';models.mkdir()
            fallback=root/'objects';folder=fallback/'smallbox/urdf';folder.mkdir(parents=True)
            mesh=folder.parent/'geometry/captured_box.obj';mesh.parent.mkdir();mesh.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
            original=mesh.read_bytes()
            (folder/'model.urdf').write_text('<robot name="smallbox"><link name="box"><visual><geometry><mesh filename="../geometry/captured_box.obj"/></geometry></visual><collision><geometry><mesh filename="../geometry/captured_box.obj"/></geometry></collision></link></robot>')
            batch.object_assets(models,'smallbox',fallback)
            self.assertEqual((models/'smallbox/smallbox.obj').read_bytes(),original)
            self.assertEqual(mesh.read_bytes(),original)
            tree=batch.ET.parse(models/'smallbox/smallbox.urdf')
            for node in tree.findall('.//mesh'):self.assertTrue((models/'smallbox'/node.get('filename')).is_file())

    def test_missing_mesh_does_not_create_fake_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);models=root/'models';models.mkdir();fallback=root/'objects';fallback.mkdir()
            (fallback/'smallbox.urdf').write_text('<robot name="box"><link name="box"><visual><geometry><mesh filename="missing.obj"/></geometry></visual></link></robot>')
            with self.assertRaises(FileNotFoundError):batch.object_assets(models,'smallbox',fallback)
            self.assertFalse((models/'smallbox').exists())

    def test_subset_copy_is_independent_and_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'all';dest=Path(tmp)/'24';source.mkdir()
            (source/'status.json').write_text(json.dumps({'fingerprint':{'source':'abc'}}))
            for n in ('selection.json','retarget.log','motion_original.npz','motion_fps50.npz'):(source/n).write_bytes(b'fixture')
            batch.sync_subset(source,dest)
            self.assertFalse((dest/'motion_original.npz').is_symlink())
            self.assertEqual(batch.digest(source/'motion_original.npz'),batch.digest(dest/'motion_original.npz'))
            (source/'motion_original.npz').write_bytes(b'changed')
            with self.assertRaises(ValueError):batch.sync_subset(source,dest)

if __name__=='__main__':unittest.main()
