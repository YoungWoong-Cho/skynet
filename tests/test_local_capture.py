import copy
import json
from uuid import uuid4
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from skynet_app.database import Database
from skynet_app.local_capture import LocalCaptureService, VisionProTracking


def records():
    matrix = [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]
    return [
        {'type':'header','schema':'skynet.visionpro-tracking/v1','session_id':str(uuid4()),
         'task':'Synthetic import regression fixture','units':'meters','matrix_order':'column-major',
         'clock':'arkit-monotonic-seconds','robot_actions':False,'images_recorded':False},
        {'type':'frame','index':0,'timestamp':100.0,'source_timestamp':99.9,'hand':'left','tracked':True,
         'origin_from_hand':matrix,'joints':[{'name':'wrist','tracked':True,'anchor_from_joint':matrix}],
         'head_tracked':True,'origin_from_head':matrix},
        {'type':'footer','frames':1,'completed':True},
    ]


def save(tmp_path, data):
    path = tmp_path / 'capture.jsonl'
    path.write_text('\n'.join(json.dumps(record) for record in data) + '\n')
    return path


def test_import_preserves_native_data_and_is_idempotent(tmp_path):
    service = LocalCaptureService(Database(tmp_path / 'test.db'))
    path = save(tmp_path, records())
    result = service.import_file('visionpro-local', path)
    assert result['imported']
    capture = result['capture']
    assert service.file(capture['sha256']).read_bytes() == path.read_bytes()
    assert service.import_file('visionpro-local', path)['imported'] is False
    assert len(service.list()) == 1
    version = service.database.get_data_resource_version(capture['version_id'])
    assert version['status'] == 'LOCAL'
    assert version['format'] == 'visionpro_tracking_jsonl_v1'
    assert version['metadata']['native_raw_preserved'] is True
    assert capture['summary']['warnings'] == ['No right hand frames were tracked.']


@pytest.mark.parametrize('mutation,message', [
    (lambda d:d.pop(), 'incomplete'),
    (lambda d:d[-1].update(completed=False), 'interrupted'),
    (lambda d:d[-1].update(frames=2), 'count'),
    (lambda d:d[1].update(index=1), 'out-of-order'),
    (lambda d:d[1].update(timestamp=float('nan')), 'finite'),
    (lambda d:d[1].update(origin_from_head=None), 'tracked head'),
    (lambda d:d[1].update(origin_from_hand=[0]*15), '16 column-major'),
    (lambda d:d[1].update(joints=[]), 'joint data'),
    (lambda d:d[1].update(tracked=False), 'No tracked hand'),
    (lambda d:d[0].update(schema='unknown/v1'), 'Unsupported'),
    (lambda d:d.append(d[1]), 'after'),
])
def test_rejects_invalid_capture(tmp_path, mutation, message):
    data = records(); mutation(data)
    with pytest.raises(ValueError, match=message):
        VisionProTracking().inspect(save(tmp_path, data))


def test_receive_clock_orders_frames_without_rewriting_source_time(tmp_path):
    data = records()
    second = copy.deepcopy(data[1]); second.update(index=1,timestamp=99,source_timestamp=98)
    data.insert(2, second); data[-1]['frames'] = 2
    with pytest.raises(ValueError, match='backwards'):
        VisionProTracking().inspect(save(tmp_path, data))
    second['timestamp'] = 101
    assert VisionProTracking().inspect(save(tmp_path, data))['duration_seconds'] == 1


def test_changed_session_cannot_overwrite_original(tmp_path):
    service = LocalCaptureService(Database(tmp_path / 'test.db'))
    data = records()
    first = service.import_file('visionpro-local', save(tmp_path, data))
    data[0]['task'] = 'changed'
    with pytest.raises(ValueError, match='immutable'):
        service.import_file('visionpro-local', save(tmp_path, data))
    assert service.list()[0]['sha256'] == first['capture']['sha256']


def test_api_import_and_download(tmp_path, monkeypatch):
    from skynet_app import local_capture_api
    service = LocalCaptureService(Database(tmp_path / 'test.db'))
    monkeypatch.setattr(local_capture_api, 'service', service)
    app = FastAPI(); app.include_router(local_capture_api.router)
    with TestClient(app) as client:
        data = save(tmp_path, records()).read_bytes()
        response = client.post('/api/collection/local/captures/visionpro-local', content=data)
        assert response.status_code == 201, response.text
        digest = response.json()['capture']['sha256']
        assert client.get(f'/api/collection/local/captures/{digest}/download').content == data
        assert client.post('/api/collection/local/captures/unknown', content=data).status_code == 422
        assert client.post('/api/collection/local/captures/visionpro-local', content=b'bad').status_code == 422
        assert list(service.root.glob('*.upload')) == []
        monkeypatch.setattr(local_capture_api, 'MAX_CAPTURE_BYTES', 1)
        assert client.post('/api/collection/local/captures/visionpro-local', content=data).status_code == 413
        assert list(service.root.glob('*.upload')) == []


def test_pinned_dashboard_gateway_does_not_fallback():
    from skynet_app.main import _gateway_candidates
    assert _gateway_candidates('sky1') == ('sky1',)
    assert _gateway_candidates('sky2') == ('sky2',)


def test_distinct_hand_and_head_clocks_are_preserved(tmp_path):
    data = records()
    data[1].update(timestamp=1324.0, source_timestamp=8120.0, head_timestamp=1324.0)
    summary = VisionProTracking().inspect(save(tmp_path, data))
    assert summary['head_tracked_frames'] == 1
    data[1]['head_timestamp'] = float('nan')
    with pytest.raises(ValueError, match='Head timestamp must be a finite number'):
        VisionProTracking().inspect(save(tmp_path, data))


def test_identical_recording_reuses_validation_and_restores_missing_file(tmp_path, monkeypatch):
    service = LocalCaptureService(Database(tmp_path / 'test.db'))
    path = save(tmp_path, records())
    first = service.import_file('visionpro-local', path)
    stored = service.file(first['capture']['sha256'])
    stored.unlink()
    def unexpected(*args):
        pytest.fail('unchanged recording was reparsed')
    monkeypatch.setattr(VisionProTracking, 'inspect', unexpected)
    repeated = service.import_file('visionpro-local', path)
    assert repeated['imported'] is False
    assert stored.read_bytes() == path.read_bytes()
    assert repeated['capture']['summary'] == first['capture']['summary']


def test_import_disk_error_is_actionable_and_cleans_upload(tmp_path, monkeypatch):
    import errno
    from skynet_app import local_capture_api
    service = LocalCaptureService(Database(tmp_path / 'test.db'))
    monkeypatch.setattr(local_capture_api, 'service', service)
    def full(*args):
        raise OSError(errno.ENOSPC, 'No space left')
    monkeypatch.setattr(service, 'import_file', full)
    app = FastAPI(); app.include_router(local_capture_api.router)
    with TestClient(app) as client:
        response = client.post('/api/collection/local/captures/visionpro-local', content=b'upload')
    assert response.status_code == 507
    assert 'free disk space' in response.json()['detail']
    assert not list(service.root.glob('*.upload'))
