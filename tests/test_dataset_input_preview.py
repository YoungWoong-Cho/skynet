"""Saved modality previews preserve data identity, coordinates, timing and bounded reuse."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skynet_app/adapters'))
sys.path.insert(0, str(ROOT / 'ops/datasets'))
from recording_dataset import canonical, digest, stream_reference
from skynet_app.dataset_preview_worker import read_frames, input_streams
from skynet_app.dataset_previews import DatasetPreviews, describe, preview_chunk_frames


@pytest.fixture
def saved(tmp_path):
    path = tmp_path / 'values.hdf5'
    pose = np.eye(4); pose[:3, 3] = [1, 2, 3]
    arrays = dict(state=np.arange(6, dtype='f4').reshape(3, 2), timestamps=np.arange(3) / 60,
                  rgb=np.full((3, 2, 2, 3), 123, dtype='u1'),
                  depth=np.ones((3, 2, 2), dtype='f4'),
                  cloud=np.tile(np.array([[0, 0, 1, .1, .2, .3], [99, 99, 99, 1, 1, 1]], dtype='f4'), (3, 1, 1)),
                  cloud_valid_mask=np.tile([True, False], (3, 1)), cloud_frame_ids=np.arange(3),
                  scene_front_intrinsics=np.tile(np.eye(3), (3, 1, 1)),
                  scene_front_world_from_camera=np.tile(pose, (3, 1, 1)))
    with h5py.File(path, 'w') as f:
        for name, values in arrays.items(): f[name] = values
    sha = digest(path)
    streams = {name: stream_reference(path, name, sha, source_sha256='a'*64) for name in arrays}
    requirements = [dict(name=name, modality=modality, camera_ids=['scene_front'],
                        coordinate_frame='camera:scene_front', color_range='0_1')
                    for name, modality in [('rgb','rgb'), ('depth','depth'), ('cloud','point_cloud')]]
    episode = dict(index=0, id='episode-0', source={'sha256':'a'*64}, steps=3, streams=streams,
                   session_id='session-0', capture={'step_dt':1/60, 'action_joint_names':['x','y']})
    manifest = dict(format='skynet.recording-dataset/v1', contract='test-inputs/v1', episodes=[episode], steps=3,
                    split={'train':[0], 'validation':[]}, observations=['state','rgb','depth','point_cloud'],
                    observation_requirements={'streams':requirements})
    folder = tmp_path / 'dataset'; folder.mkdir()
    (folder / 'manifest.json').write_bytes(canonical(manifest))
    request = dict(path=str(folder), sha256=digest(folder / 'manifest.json'), allowed_roots=[str(tmp_path)],
                   episode_index=0, start=0, count=2)
    return request, manifest, path


def test_reads_only_declared_modalities_with_preaction_timing_and_world_coordinates(saved):
    request, manifest, _ = saved
    result = read_frames(request)
    first, second = result['frames']
    assert first['index'] == 0 and first['time'] == 0
    assert second['time'] == pytest.approx(1/60)
    assert first['state'][0]['values'] == [0, 1]
    assert first['state'][0]['labels'] == ['x','y']
    image = first['images'][0]
    assert image['data_url'].startswith('data:image/png;base64,iVBOR')
    assert image['width'] == image['height'] == 2
    np.testing.assert_allclose(image['world_from_camera'], np.array([[1,0,0,1],[0,1,0,2],[0,0,1,3],[0,0,0,1]]))
    assert first['point_cloud'][0]['positions'] == [[1,2,4]]
    np.testing.assert_allclose(first['point_cloud'][0]['colors'], [[.1,.2,.3]])
    assert first['depth'][0]['positions'] == [[1,2,4],[2,2,4],[1,3,4],[2,3,4]]
    manifest['observation_requirements']['streams'] = [manifest['observation_requirements']['streams'][2]]
    (Path(request['path'])/'manifest.json').write_bytes(canonical(manifest))
    request['sha256'] = digest(Path(request['path'])/'manifest.json')
    only_cloud = read_frames(request)['frames'][0]
    assert not only_cloud['images'] and not only_cloud['depth']
    assert only_cloud['point_cloud']


def test_reuses_verified_files_but_detects_changed_hdf5(saved, monkeypatch):
    request, _, path = saved
    result = read_frames(request)
    pinned = {key:result[key] for key in ('spec','manifest_stat','file_stamps')}
    import recording_dataset
    original = recording_dataset.digest
    monkeypatch.setattr(recording_dataset, 'digest', lambda _: pytest.fail('Unchanged immutable file was hashed again'))
    assert read_frames(dict(request, start=2, count=1, **pinned))['frames'][0]['index'] == 2
    monkeypatch.setattr(recording_dataset, 'digest', original)
    with h5py.File(path, 'r+') as file: file['state'][0] = [9,9]
    with pytest.raises(ValueError, match='checksum differs'):
        read_frames(dict(request, **pinned))


def test_rejects_manifest_tampering_and_path_escape(saved, tmp_path):
    request, manifest, path = saved
    (Path(request['path'])/'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='registered checksum'): read_frames(request)
    manifest['episodes'][0]['streams']['state']['path'] = '/etc/passwd'
    (Path(request['path'])/'manifest.json').write_bytes(canonical(manifest))
    request['sha256'] = digest(Path(request['path'])/'manifest.json')
    with pytest.raises(ValueError, match='registered workspace'): read_frames(request)


def test_rejects_symlinks_and_cross_recording_streams(saved):
    request, manifest, path = saved
    symlink = path.with_name('link.hdf5'); symlink.symlink_to(path)
    manifest['episodes'][0]['streams']['state']['path'] = str(symlink)
    manifest_file = Path(request['path'])/'manifest.json'
    manifest_file.write_bytes(canonical(manifest)); request['sha256'] = digest(manifest_file)
    with pytest.raises(ValueError, match='symbolic links'): read_frames(request)
    manifest['episodes'][0]['streams']['state']['path'] = str(path)
    manifest['episodes'][0]['streams']['state']['source_sha256'] = 'b'*64
    manifest_file.write_bytes(canonical(manifest)); request['sha256'] = digest(manifest_file)
    with pytest.raises(ValueError, match='different recording'): read_frames(request)


def test_rejects_camera_frame_misalignment(saved):
    request, manifest, path = saved
    with h5py.File(path, 'r+') as f: f['cloud_frame_ids'][1] = 2
    for reference in manifest['episodes'][0]['streams'].values(): reference['sha256'] = digest(path)
    (Path(request['path'])/'manifest.json').write_bytes(canonical(manifest))
    request['sha256'] = digest(Path(request['path'])/'manifest.json')
    with pytest.raises(ValueError, match='not aligned'): read_frames(request)


def test_prefers_saved_faas_input_state_and_ignores_sibling_rgb(saved):
    _, manifest, _ = saved
    episode = manifest['episodes'][0]
    episode['streams']['faas_state_absolute'] = episode['streams']['state']
    manifest['observation_requirements']['streams'] = []
    assert [item['name'] for item in input_streams(manifest, episode)] == ['faas_state_absolute']


def test_chunk_recommendation_follows_saved_input_shapes(saved):
    _, manifest, _ = saved
    episode = manifest['episodes'][0]
    assert preview_chunk_frames(manifest, episode) == 120
    episode['streams']['rgb']['shape'] = [3, 2048, 2048, 3]
    assert preview_chunk_frames(manifest, episode) == 1


def dataset_for(request, manifest):
    return dict(id='dataset-0', format=manifest['format'], metadata=manifest, manifest_sha256=request['sha256'],
                locations=[dict(id='loc-0', kind='cluster', status='AVAILABLE', path=request['path'], manifest_sha256=request['sha256'])])


def test_description_resolves_reordered_original_by_checksum(saved):
    request, manifest, _ = saved
    job = dict(recordings=['new-first.pkl','original.pkl'], recording_checksums={'new-first.pkl':'b'*64,'original.pkl':'a'*64})
    class DB:
        @contextmanager
        def connection(self): yield self
        def execute(self, query, params): return self
        def fetchall(self): return [{'id':'session-0','payload_json':json.dumps(job)}]
    dataset = dataset_for(request, manifest)
    result = describe(DB(), dataset)
    assert result['episodes'][0]['source_recording_index'] == 1
    assert result['episodes'][0]['steps'] == 3
    job['recording_checksums']['original.pkl'] = 'c'*64
    result = describe(DB(), dataset)
    assert result['episodes'][0]['source_session_id'] is None
    assert result['state'] == 'READY'


def test_chunk_cache_is_content_and_workspace_scoped_bounded_and_rechecks_availability(saved):
    request, manifest, _ = saved
    result = read_frames(request)
    dataset = dataset_for(request, manifest)
    cache = DatasetPreviews()
    class Cluster:
        work_root = request['allowed_roots'][0]
        calls = 0
        def run_with_fallback(self, *args, **kwargs):
            self.calls += 1
            return 'test', json.dumps(result)
    cluster = Cluster(); db = SimpleNamespace(workspace_id='first')
    first = cache.frames(db, cluster, dataset, 0, 0, 2)
    assert cache.frames(db, cluster, dataset, 0, 0, 2) == first
    assert cluster.calls == 1
    cache.frames(SimpleNamespace(workspace_id='second'), cluster, dataset, 0, 0, 2)
    assert cluster.calls == 2
    dataset['locations'][0]['status'] = 'MISSING'
    with pytest.raises(ValueError, match='verified cluster location'): cache.frames(db, cluster, dataset, 0, 0, 2)
    dataset['locations'][0]['status'] = 'AVAILABLE'
    cache.MAX_CACHE_BYTES = 1
    cache.frames(db, cluster, dataset, 0, 1, 2)
    assert cache.cache_bytes == 0 and not cache.cache


@pytest.mark.parametrize('source_root', ['historical', 'shared'])
def test_registered_old_dataset_and_streams_remain_readable_after_work_root_change(saved, tmp_path, monkeypatch, source_root):
    import ast
    import skynet_app.dataset_previews as module
    request, manifest, _ = saved
    old_root = request['allowed_roots'][0]
    new_root = str(tmp_path / 'new-personal-root')
    roots = {new_root, old_root} if source_root == 'historical' else {new_root}
    config = SimpleNamespace(paths=SimpleNamespace(datasets=old_root if source_root == 'shared' else '/registered/shared/datasets'),
                             runtime_profiles=module.CLUSTER.runtime_profiles)
    monkeypatch.setattr(module, 'CLUSTER', config)
    class Cluster:
        work_root = new_root
        storage = SimpleNamespace(allowed_roots=lambda:roots)
        calls = 0
        def run_with_fallback(self, *args, **kwargs):
            self.calls += 1
            expression = ast.parse(kwargs['stdin']).body[-1].value
            dispatched = ast.literal_eval(expression.args[0].args[0].args[0])
            assert old_root in dispatched['allowed_roots']
            assert new_root in dispatched['allowed_roots']
            return 'test', json.dumps(read_frames(dispatched))
    cluster, cache = Cluster(), DatasetPreviews()
    dataset = dataset_for(request, manifest)
    output = json.loads(cache.frames(SimpleNamespace(workspace_id='owner'), cluster, dataset, 0, 0, 2))
    assert output['frames'][0]['state'][0]['values'] == [0, 1]
    assert output['frames'][0]['point_cloud'][0]['positions'] == [[1, 2, 4]]
    assert cluster.calls == 1
    foreign = deepcopy(dataset)
    foreign['locations'][0]['path'] = '/unregistered/other-user/dataset'
    with pytest.raises(ValueError, match='registered workspace root'):
        cache.frames(SimpleNamespace(workspace_id='owner'), cluster, foreign, 0, 0, 2)
    assert cluster.calls == 1
