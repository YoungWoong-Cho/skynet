"""Archived videos keep their pinned runtime after conversion worker retirement."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skynet_app.cluster_runtime import WORK_ROOT
from skynet_app.dexverse_versions import V1_REVISION, V1_REPOSITORY
from skynet_app.live_xr_video import LiveVideoService, VideoGeneration


@pytest.fixture
def video(tmp_path):
    configured = json.loads((Path(__file__).resolve().parents[1] / 'config/live_video.json').read_text())
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config/live_video.json').write_text(json.dumps(configured))
    cluster = SimpleNamespace(candidates=lambda host: (host,), _remote_path=lambda path: path)
    live = SimpleNamespace(root=tmp_path, archive=SimpleNamespace(cluster=cluster))
    service = LiveVideoService(SimpleNamespace(live=live))
    service.sources = {}
    service.version = 'a' * 16
    job = dict(id='session', root='/workstation/session', gateway='workstation',
               profile=dict(task='Dexverse-PushT-v1', robot='skynet_wuji_2_right',
                            source_revision=V1_REVISION, hand='right', hand_name='WUJI',
                            hand_bundle=dict(digest='b' * 64, root='/old/local/path')),
               archive=dict(state='READY', gateway='sky2'))
    yield service, job, configured
    service.executor.shutdown(wait=True)


def test_archived_video_profile_preserves_recording_identity_without_legacy_config(video):
    service, job, configured = video
    before = copy.deepcopy(job)
    profile = service.cluster_profile(job)
    assert profile['execution'] == 'slurm'
    assert profile['repository'] == f'{WORK_ROOT}/{V1_REPOSITORY}'
    assert profile['source_revision'] == V1_REVISION
    assert profile['runtime'] == configured['runtime']
    assert profile['robot'] == job['profile']['robot']
    assert profile['hand_bundle']['root'] == f"{WORK_ROOT}/hands/{profile['robot']}/{'b' * 64}"
    assert job == before
    assert not hasattr(service.live, 'conversions')
    assert not (service.live.root / 'config/capture_pipelines.json').exists()
    assert not (service.live.root / 'config/live_conversion.json').exists()


def test_archived_video_profile_rejects_revision_or_hand_identity_change(video):
    service, job, _ = video
    job['profile']['source_revision'] = 'wrong-revision'
    with pytest.raises(ValueError, match='different DexVerse revisions'):
        service.cluster_profile(job)
    job['profile']['source_revision'] = V1_REVISION
    job['profile']['hand_bundle']['digest'] = '../another-hand'
    with pytest.raises(ValueError, match='invalid hand bundle identity'):
        service.cluster_profile(job)


@pytest.mark.parametrize('legacy', [False, True])
def test_video_finds_pinned_hand_in_cache_or_historical_app_path(video, tmp_path, monkeypatch, legacy):
    from skynet_app import simulation_hands

    service, _, _ = video
    cache = tmp_path / 'cache'
    monkeypatch.setattr(simulation_hands, 'cache_root', lambda: cache)
    local = (service.live.root / 'data/simulation-hands' if legacy else cache) / 'robot' / 'digest'
    local.mkdir(parents=True)
    (local / 'manifest.json').write_text('{}')
    calls = []
    monkeypatch.setattr('skynet_app.live_xr_video.upload_hand',
                        lambda path, root, transport, gateway: calls.append((path, root, gateway)) or '/cluster/hand/digest')
    service.ensure_hand(dict(robot='robot', hand_bundle=dict(digest='digest', root='/cluster/hand/digest')), object(), 'sky2')
    assert calls == [(local, WORK_ROOT, 'sky2')]
    with pytest.raises(ValueError, match='differs from the saved request'):
        service.ensure_hand(dict(robot='robot', hand_bundle=dict(digest='digest', root='/wrong/path')), object(), 'sky2')


def test_archived_render_uses_standalone_video_profile(video, monkeypatch):
    service, job, _ = video
    calls = []
    monkeypatch.setattr(service, 'ensure_hand', lambda *args: calls.append(('hand', args)))
    def control(transport, render_job, generation, root, operation, **value):
        calls.append(('render', transport, render_job, root, operation))
        return dict(state='READY', path=root + '/video.mp4')
    monkeypatch.setattr(service, '_control', control)
    generation = VideoGeneration()
    result = service.render(None, job, WORK_ROOT + '/raw/demo.pkl', dict(sha256='c' * 64), 0, generation)
    assert result['state'] == 'READY'
    assert calls[0][0] == 'hand'
    _, transport, rendered, root, operation = calls[1]
    assert transport is service.live.archive.cluster
    assert rendered['profile']['source_revision'] == V1_REVISION
    assert rendered['gateway'] == 'sky2' and operation == 'start'
    assert root == f'{WORK_ROOT}/jobs/runs/{generation.token}'
    assert generation.remote_stopped


def test_interrupted_archived_video_cancellation_recovers_without_conversion_service(video, tmp_path, monkeypatch):
    service, job, _ = video
    directory = tmp_path / 'video'
    directory.mkdir()
    token = 'd' * 32
    root = f'{WORK_ROOT}/jobs/runs/{token}'
    service.publish(directory, state='PREPARING', generation=token, remote_root=root,
                    cluster_submission_started=True, cluster_job_id='123')
    monkeypatch.setattr(service, 'source', lambda *args: (job, '/archive/recording.pkl', dict(sha256='c' * 64), directory))
    cancelled = []
    def control(transport, restored, generation, remote, operation, **kwargs):
        cancelled.append((transport, restored, generation.cluster_job_id, remote, operation))
        return dict(state='CANCELLED')
    monkeypatch.setattr(service, '_control', control)
    assert service.cancel('session', 0, 0, token)['state'] == 'CANCELLED'
    assert len(cancelled) == 1
    transport, restored, job_id, remote, operation = cancelled[0]
    assert transport is service.live.archive.cluster
    assert restored['profile']['source_revision'] == V1_REVISION
    assert (job_id, remote, operation) == ('123', root, 'cancel')
    assert not service.active
