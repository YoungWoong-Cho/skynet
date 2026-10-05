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
    for name in ('HandKinematics', 'camera_layout', 'recorded_urdf', 'replay_urdf', 'pose_matrix', 'MAX_VIEWER_BYTES'):
        monkeypatch.setattr(worker, name, getattr(episode_geometry, name), raising=False)
    monkeypatch.setattr(worker, 'ArrayUnpickler', ArrayUnpickler, raising=False)
    states = [{'articulation': {'robot': {'root_pose': np.array([ROOT]), 'joint_position': np.array([[i, .1*i]])}}} for i in range(3)]
    recording = tmp_path / 'episode.pkl'
    recording.write_bytes(pickle.dumps({'format':'dexverse_trajectory', 'schema_version':3, 'robot_type':'test', 'episodes':[{'states':states, 'actions':np.zeros((2,2)), 'num_steps':2, 'success':True}]}))
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
    assert len(written) == 0
    assert not (tmp_path / "preview/views.mp4").exists()
    assert data["views"] == []
    assert len(data['frames']) == 3
    assert data['frames'][-1]['index'] == 2 and data['frames'][-1]['time'] == .2
    np.testing.assert_allclose(data['frames'][-1]['hand_poses']['actual']['joints'], [.2,2])
    np.testing.assert_allclose(data['frames'][-1]['actual'], HandKinematics(XML,['bend','move'],['move']).points([.2,2], ROOT))


def test_raw_recording_preview_has_no_image_dependency(tmp_path, monkeypatch):
    import pickle
    from skynet_app import episode_preview_worker as worker
    from skynet_app.adapters import episode_geometry
    from skynet_app.live_xr_review import ArrayUnpickler
    for name in ("HandKinematics", "recorded_urdf", "replay_urdf", "pose_matrix", "MAX_VIEWER_BYTES"):
        monkeypatch.setattr(worker, name, getattr(episode_geometry, name), raising=False)
    monkeypatch.setattr(worker, "ArrayUnpickler", ArrayUnpickler, raising=False)
    metadata = dict(robot="test", hand="right", robot_joint_names=["move", "bend"], action_joint_names=["bend", "move"],
                    groups=[dict(wrist_indices=[1])], step_dt=.02, kinematics_urdf=XML,
                    kinematics_sha256=hashlib.sha256(XML.encode()).hexdigest(),
                    scene_geometry=dict(schema="skynet.scene-geometry/v1", objects=[dict(name="cube", type="box")]))
    states = [dict(articulation=dict(robot=dict(root_pose=np.array([ROOT]), joint_position=np.array([[i, .1*i]]))),
                   rigid_object=dict(cube=dict(root_pose=np.array([[i,0,0,1,0,0,0]])))) for i in range(3)]
    path = tmp_path / "source.pkl"
    raw = pickle.dumps(dict(format="dexverse_trajectory", schema_version=3, robot_type="test", skynet_state_metadata=metadata,
                            episodes=[dict(actions=np.zeros((2,2)), states=states, success=True, num_steps=2)]))
    path.write_bytes(raw)
    # A recorded metadata path must never fall back to HDF5 or image encoders.
    monkeypatch.setattr(worker, "_legacy_metadata", lambda *a: (_ for _ in ()).throw(AssertionError("read images")))
    request = dict(output=str(tmp_path/"preview"), recording=str(path), source_sha256=hashlib.sha256(raw).hexdigest(),
                   robot="test", episode=0, repository="/missing", hand_visual_url="/frozen/hand", preview_version="test-v1")
    assert worker.prepare_preview(request) == {"state":"READY"}
    result = json.loads((tmp_path/"preview/viewer.json").read_text())
    assert result["views"] == [] and "video_file" not in result
    assert result["hand_visual_available"] is True
    assert [frame["time"] for frame in result["frames"]] == [0, .02, .04]
    assert result["frames"][-1]["objects"]["cube"] == [2,0,0,1,0,0,0]
    np.testing.assert_allclose(result["frames"][1]["actual"], HandKinematics(XML,["bend","move"],["move"]).points([.1,1],ROOT))
    assert path.read_bytes() == raw
    assert not list((tmp_path/"preview").glob("*.mp4"))
    path.write_bytes(raw + b"changed")
    with __import__("pytest").raises(ValueError, match="checksum changed"):
        worker.prepare_preview(request)


def test_frozen_hand_assets_are_verified_and_do_not_use_current_catalog(tmp_path):
    from skynet_app.episode_previews import verified_hand_file
    import pytest
    folder = tmp_path / "bundle"; (folder/"assets").mkdir(parents=True)
    asset = folder/"assets/hand.stl"; asset.write_bytes(b"original mesh")
    model = folder/"simulation.urdf"
    model.write_text('<robot><link name="palm"><visual><geometry><mesh filename="assets/hand.stl"/></geometry></visual></link></robot>')
    manifest = dict(digest="a"*64, robot="saved", files={str(p.relative_to(folder)):dict(size_bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (asset,model)})
    (folder/"manifest.json").write_text(json.dumps(manifest))
    request = dict(root=str(folder), digest="a"*64, robot="saved", url_prefix="/recording/hand/", name="simulation.urdf")
    result = verified_hand_file(request)
    assert 'filename="/recording/hand/assets/hand.stl"' in result["text"]
    assert "/api/hands/" not in result["text"]
    with pytest.raises(ValueError, match="Unknown"):
        verified_hand_file(dict(request, name="assets/../../outside"))
    asset.write_bytes(b"modified mesh")
    with pytest.raises(ValueError, match="checksum changed"):
        verified_hand_file(dict(request, name="assets/hand.stl"))


def test_cluster_preview_uses_task_version_pinned_repository():
    from pathlib import Path
    from types import SimpleNamespace
    from skynet_app.cluster_runtime import DEFAULT_GATEWAY, WORK_ROOT
    from skynet_app.dexverse_versions import V1_REVISION, V1_REPOSITORY
    from skynet_app.episode_previews import EpisodePreviews
    root = Path(__file__).resolve().parents[1]
    base = json.loads((root / 'config/live_video.json').read_text())
    for task, revision, repository in [
        ('Dexverse-PickCube-v0', base['source_revision'], base['repository']),
        ('Dexverse-PushT-v1', V1_REVISION, WORK_ROOT + '/' + V1_REPOSITORY),
    ]:
        session = dict(id='recording', recordings=['recordings/live/episode.pkl'],
                       recording_checksums={'recordings/live/episode.pkl': 'a' * 64},
                       profile=dict(task=task, source_revision=revision, robot='floating_shadow_hand'))
        cluster = object()
        live = SimpleNamespace(root=root, archive=SimpleNamespace(
            cluster=cluster, resolve=lambda job, path: (cluster, DEFAULT_GATEWAY, '/archive/' + path)))
        reviews = SimpleNamespace(live=live, source=lambda *args: (session, '/unused'),
                                  remote_location=lambda *args: SimpleNamespace(path='/archive/reviews/review.json'))
        previews = EpisodePreviews(reviews)
        try:
            transport, gateway, _, request = previews.location('recording', 0, 0)
            assert transport is cluster and gateway == DEFAULT_GATEWAY and request['repository'] == repository
        finally:
            previews.executor.shutdown(wait=True)


def test_recorded_hand_and_scene_are_read_through_storage_routing(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from skynet_app import simulation_hands
    from skynet_app.cluster_runtime import DEFAULT_GATEWAY
    from skynet_app.episode_previews import EpisodePreviews
    monkeypatch.setattr(simulation_hands, 'cache_root', lambda: tmp_path / 'cache')
    calls = []

    class Storage:
        def __init__(self, name, reply):
            self.name, self.reply = name, reply

        def gateway_for(self, recorded):
            return recorded

        def run_with_fallback(self, command, gateway, *, stdin=None, **budget):
            calls.append((self.name, gateway, command))
            if self.reply is None:
                raise RuntimeError('Recorded hand manifest is unavailable')
            return 'answering-host', 'login banner\n' + json.dumps(self.reply) + '\n'

    cluster = Storage('cluster', None)
    workstation = Storage('workstation', dict(path='/workstation/hand/simulation.urdf', text='<robot/>'))
    digest = 'a' * 64
    session = dict(id='recording', gateway='bonjour', recordings=['recordings/live/episode.pkl'],
                   profile=dict(robot='saved', hand_bundle=dict(digest=digest, root='/workstation/hands/saved/' + digest)),
                   archive=dict(state='READY', gateway='sky2'))
    live = SimpleNamespace(root=tmp_path, transport=lambda job: workstation, archive=SimpleNamespace(
        cluster=cluster, resolve=lambda job, path: (cluster, DEFAULT_GATEWAY, '/archive/' + path)))
    previews = EpisodePreviews(SimpleNamespace(live=live, source=lambda *args: (session, '/unused')))
    try:
        transport, gateway, result = previews.hand_file('recording', 0, 'simulation.urdf')
        assert transport is workstation and gateway == 'bonjour' and result['text'] == '<robot/>'
        # Archive storage by its default route first, then the recording's own collection host.
        assert [call[:2] for call in calls] == [('cluster', DEFAULT_GATEWAY), ('workstation', 'bonjour')]
        cluster.reply = {'state': 'READY'}
        key = ('recording', 0, 0)
        previews.states[key] = dict(state='PREPARING', source_sha256='b' * 64)
        previews.prepare(key, (cluster, DEFAULT_GATEWAY, '/runtime/bin/python', dict(source_sha256='b' * 64)))
        assert previews.states[key] == dict(state='READY', source_sha256='b' * 64)
        assert calls[-1] == ('cluster', DEFAULT_GATEWAY, '/runtime/bin/python -')
    finally:
        previews.executor.shutdown(wait=True)
