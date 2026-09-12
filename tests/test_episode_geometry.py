"""CPU regression checks for portable episode geometry; no service or GPU needed."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skynet_app/adapters'))
from episode_geometry import HandKinematics, camera_layout, recorded_urdf
from episode_trace import EpisodeTrace

XML = '''<robot name="hand"><link name="base"/><link name="wrist"/><link name="finger"/><link name="tip"/>
<joint name="move" type="prismatic"><parent link="base"/><child link="wrist"/><axis xyz="1 0 0"/></joint>
<joint name="bend" type="revolute"><parent link="wrist"/><child link="finger"/><axis xyz="0 0 1"/></joint>
<joint name="end" type="fixed"><parent link="finger"/><child link="tip"/><origin xyz="1 0 0"/></joint></robot>'''
ROOT = [10, 0, 0, 1, 0, 0, 0]

class GeometryTest(unittest.TestCase):
    def test_named_order_root_and_rotated_joint(self):
        model = HandKinematics(XML, ['bend', 'move'], ['move'])
        points = model.points([np.pi / 2, 2], ROOT)
        np.testing.assert_allclose(points[model.links.index('tip')], [12, 1, 0], atol=1e-6)
        with self.assertRaisesRegex(ValueError, 'joint names'):
            HandKinematics(XML, ['move'])

    def test_geometry_checksum_and_mimic(self):
        capture = {'kinematics_urdf': XML, 'kinematics_sha256': hashlib.sha256(XML.encode()).hexdigest()}
        self.assertEqual(recorded_urdf(capture, '/missing'), XML)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            recorded_urdf({**capture, 'kinematics_urdf': XML+' '}, '/missing')
        xml = XML.replace('type="fixed"><parent link="finger"', 'type="revolute"><mimic joint="bend" multiplier="-1"/><axis xyz="0 0 1"/><parent link="finger"')
        model = HandKinematics(xml, ['move', 'bend'])
        self.assertTrue(np.isfinite(model.points([2, .5], ROOT)).all())

    def test_different_camera_sizes_match_centered_compositor(self):
        views = camera_layout({'a':{'width':256,'height':256}, 'b':{'width':128,'height':64}}, labels=True)
        self.assertEqual(views[1]['rect'], [320, 128, 128, 64])

    def test_targets_actual_and_demo_remain_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = {'robot':'test', 'hand':'right', 'action_joint_names':['bend','move'],
                       'robot_joint_names':['move','bend'], 'action_scale':[2,3], 'action_offset':[.1,.2],
                       'groups':[{'wrist_indices':[1]}], 'kinematics_urdf':XML,
                       'kinematics_sha256':hashlib.sha256(XML.encode()).hexdigest()}
            robot = SimpleNamespace(joint_names=['move','bend'], data=SimpleNamespace(
                root_pos_w=np.array([ROOT[:3]]),root_quat_w=np.array([ROOT[3:]]),joint_pos=np.array([[1.,0.]])))
            env = SimpleNamespace(scene={'robot':robot},step_dt=.1)
            trace = EpisodeTrace({'evaluator_runtime':{'source_dir':'/missing'}},capture,env,{},Path(directory)/'episode.mp4',policy_order=[1,0])
            trace.demonstration=[{'articulation':{'robot':{'joint_position':np.array([[4.,0.]]),'root_pose':ROOT}}}]
            trace.append(0,[2.,0.])
            trace.append(1,[2.,0.])
            model = trace.kinematics
            np.testing.assert_allclose(trace.frames[0]['actual'], model.points([0,1],ROOT))
            np.testing.assert_allclose(trace.frames[0]['prediction'], model.points([.1,6.2],ROOT))
            np.testing.assert_allclose(trace.frames[0]['demonstration'], model.points([0,4],ROOT))
            self.assertNotIn('demonstration',trace.frames[1])
            trace.finish(2)
            saved = json.loads(trace.path.read_text())
            for layer in ('actual', 'prediction', 'demonstration'):
                pose = saved['frames'][0]['hand_poses'][layer]
                np.testing.assert_allclose(saved['frames'][0][layer], model.points(pose['joints'], pose['root']))
            self.assertNotIn('demonstration', saved['frames'][1]['hand_poses'])
            self.assertTrue(saved['kinematics_urdf'])
            self.assertEqual(saved['joint_names'], capture['action_joint_names'])
            self.assertEqual(saved['duration'],.2)
            self.assertFalse(trace.path.with_suffix('.tmp').exists())

if __name__ == '__main__':
    unittest.main()


def test_camera_preview_keeps_final_saved_state(tmp_path, monkeypatch):
    import h5py
    import pickle
    from contextlib import contextmanager
    written = []
    @contextmanager
    def writer(path, **kwargs):
        yield SimpleNamespace(append_data=lambda image: written.append(image.copy()))
        Path(path).write_bytes(b'test camera stream')
    imageio = SimpleNamespace(v2=SimpleNamespace(get_writer=writer))
    monkeypatch.setitem(sys.modules, 'imageio', imageio)
    monkeypatch.setitem(sys.modules, 'imageio.v2', imageio.v2)
    from skynet_app import episode_preview_worker as worker
    from skynet_app.adapters import episode_geometry
    from skynet_app.live_xr_review import ArrayUnpickler
    for name in ('HandKinematics', 'camera_layout', 'recorded_urdf', 'replay_urdf'):
        monkeypatch.setattr(worker, name, getattr(episode_geometry, name), raising=False)
    monkeypatch.setattr(worker, 'ArrayUnpickler', ArrayUnpickler, raising=False)
    states = [{'articulation': {'robot': {'root_pose': np.array([ROOT]), 'joint_position': np.array([[i, .1*i]])}}} for i in range(3)]
    recording = tmp_path / 'episode.pkl'
    recording.write_bytes(pickle.dumps({'episodes':[{'states':states}]}))
    metadata = {'robot':'test', 'hand':'right', 'robot_joint_names':['move','bend'], 'action_joint_names':['bend','move'],
                'groups':[{'wrist_indices':[1]}], 'step_dt':.1, 'kinematics_urdf':XML,
                'kinematics_sha256':hashlib.sha256(XML.encode()).hexdigest(), 'cameras':{'front':{'width':16,'height':16}}}
    images = tmp_path / 'images.h5'
    with h5py.File(images, 'w') as output:
        output.attrs['schema'] = 'skynet.rgb-trajectory/v1'
        output.attrs['complete'] = True
        output.attrs['metadata'] = json.dumps(metadata)
        output['state'] = np.array([[0,0],[.1,1]])
        output['timestamps'] = np.array([0,.1])
        output['images/front'] = np.zeros((2,16,16,3), dtype='u1')
    result = worker.prepare_preview({'output':str(tmp_path/'preview'), 'images':{'path':str(images), 'size_bytes':images.stat().st_size, 'sha256':hashlib.sha256(images.read_bytes()).hexdigest()},
                                    'recording':str(recording), 'source_sha256':hashlib.sha256(recording.read_bytes()).hexdigest(), 'episode':0, 'repository':'/missing'})
    assert result == {'state':'READY'}
    data = json.loads((tmp_path/'preview/viewer.json').read_text())
    assert len(written) == 2
    assert len(data['frames']) == 3
    assert data['frames'][-1]['index'] == 2 and data['frames'][-1]['time'] == .2
    np.testing.assert_allclose(data['frames'][-1]['hand_poses']['actual']['joints'], [.2,2])
    np.testing.assert_allclose(data['frames'][-1]['actual'], HandKinematics(XML,['bend','move'],['move']).points([.2,2], ROOT))
