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
            self.assertEqual(saved['duration'],.2)
            self.assertFalse(trace.path.with_suffix('.tmp').exists())

if __name__ == '__main__':
    unittest.main()
