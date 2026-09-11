import hashlib
import json
from pathlib import Path
import subprocess
import threading
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from capture_storage_fake import MemoryStorage
from test_local_capture import records, save
from skynet_app.cluster_runtime import ClusterError
from skynet_app.database import Database, canonical_json, utc_now
from skynet_app.local_capture import LocalCaptureService, VisionProTracking
from skynet_app.local_capture_storage import CaptureStorage


def legacy_capture(tmp_path, storage=None):
    storage = storage or MemoryStorage()
    db = Database(tmp_path / 'test.db')
    service = LocalCaptureService(db, storage=storage)
    data = save(tmp_path, records()).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    service.root.mkdir()
    original = service.root / f'{digest}.jsonl'
    original.write_bytes(data)
    summary = VisionProTracking().inspect(original)
    resource = db.create_data_resource(provider='collection', namespace='visionpro-local',
                                      name=summary['header']['session_id'], kind='raw_capture')
    version = db.create_data_resource_version(resource['id'], revision=digest,
        path=str(original), format='visionpro_tracking_jsonl_v1', manifest_sha256=digest,
        size_bytes=len(data), status='LOCAL', metadata={'original': True})
    with db.transaction() as connection:
        connection.execute('INSERT INTO local_captures VALUES (?,?,?,?,?,?,?,?)',
            (digest, 'visionpro-local', summary['header']['session_id'], original.name, len(data),
             canonical_json(summary), version['id'], utc_now()))
    return service, digest, original, version


def test_legacy_migration_verifies_location_preserves_history_and_never_deletes(tmp_path):
    service, digest, original, version = legacy_capture(tmp_path)
    assert service.list()[0]['location'] == 'Awaiting cluster migration'
    with pytest.raises(ValueError, match='awaiting verified migration'):
        service.file(digest)
    result = service.migrate_capture(digest)
    assert result['location'] == 'sky2'
    assert original.is_file()
    after = service.database.get_data_resource_version(version['id'])
    assert {key: after[key] for key in ('path','revision','status','metadata')} == {
        key: version[key] for key in ('path','revision','status','metadata')}
    assert after['locations'][0]['path'] == service.storage.path(digest)
    expected = original.read_bytes()
    original.unlink()  # Simulate separately authorized, verified cleanup.
    assert service.read(digest).data == expected
    assert service.migrate_capture(digest)['location'] == 'sky2'
    assert not original.exists()
    service.import_bytes('visionpro-local', expected)
    assert not original.exists()


def test_migration_refuses_changed_local_bytes_before_upload(tmp_path):
    service, digest, original, _ = legacy_capture(tmp_path)
    data = original.read_bytes(); original.write_bytes(b'x' + data[1:])
    with pytest.raises(ValueError, match='checksum'):
        service.migrate_capture(digest)
    assert not service.storage.files
    assert original.exists()


def test_failed_remote_verification_does_not_publish_location_or_delete(tmp_path):
    storage = MemoryStorage()
    service, digest, original, version = legacy_capture(tmp_path, storage)
    storage.verify = lambda *args: (_ for _ in ()).throw(ClusterError('checksum failure'))
    with pytest.raises(ClusterError):
        service.migrate_capture(digest)
    assert service.database.get_data_resource_version(version['id'])['locations'] == []
    assert original.exists()


def test_remote_read_detects_same_size_corruption(tmp_path):
    service = LocalCaptureService(Database(tmp_path / 'test.db'), storage=MemoryStorage())
    result = service.import_file('visionpro-local', save(tmp_path, records()))
    digest = result['capture']['sha256']; path = service.storage.path(digest)
    service.storage.files[path] = b'x' + service.storage.files[path][1:]
    with pytest.raises(ValueError, match='checksum changed'):
        service.read(digest)
    assert not service.root.exists()


def test_api_streams_ranges_without_a_local_file_and_limits_parallel_memory(tmp_path, monkeypatch):
    from skynet_app import local_capture_api as api
    service = LocalCaptureService(Database(tmp_path / 'test.db'), storage=MemoryStorage())
    monkeypatch.setattr(api, 'service', service)
    app = FastAPI(); app.include_router(api.router)
    data = save(tmp_path, records()).read_bytes()
    with TestClient(app) as client:
        result = client.post('/api/collection/local/captures/visionpro-local', content=data)
        assert result.status_code == 201
        digest = result.json()['capture']['sha256']
        response = client.get(f'/api/collection/local/captures/{digest}/download', headers={'Range':'bytes=2-9'})
        assert response.status_code == 206
        assert response.content == data[2:10]
        assert response.headers['cache-control'] == 'no-store'
        assert not service.root.exists()
        assert api._import_slot.acquire(blocking=False)
        try:
            response = client.post('/api/collection/local/captures/visionpro-local', content=data)
            assert response.status_code == 429
        finally:
            api._import_slot.release()
        assert client.post('/api/collection/local/captures/visionpro-local', content=data).status_code == 201


def test_independent_services_publish_one_capture_and_version(tmp_path):
    storage = MemoryStorage()
    services = [LocalCaptureService(Database(tmp_path / 'test.db'), storage=storage) for _ in range(2)]
    data = save(tmp_path, records()).read_bytes()
    barrier = threading.Barrier(2)
    original = storage.publish
    def publish(*args):
        barrier.wait(timeout=5)
        return original(*args)
    storage.publish = publish
    outcomes = []; errors = []
    def run(service):
        try:
            outcomes.append(service.import_bytes('visionpro-local', data))
        except Exception as error:
            errors.append(error)
    threads = [threading.Thread(target=run, args=(service,)) for service in services]
    for thread in threads: thread.start()
    for thread in threads: thread.join(timeout=10)
    assert not errors
    assert sorted(item['imported'] for item in outcomes) == [False, True]
    assert len(services[0].list()) == 1
    resource = services[0].database.list_data_resources()[0]
    assert len(services[0].database.get_data_resource(resource['id'])['versions']) == 1


def test_storage_upload_uses_binary_stdin_and_verifies_before_success(monkeypatch):
    class Cluster:
        def candidates(self, gateway): assert gateway == 'sky2'
        def ssh(self, gateway, command, **kwargs):
            assert 'sha256sum' in command and 'wc -c' in command
            checks.append(command)
            return ''
    checks = []; sent = []
    storage = CaptureStorage(Cluster()); data = b'{"x":"\xc3\xa9"}\r\n'
    digest = hashlib.sha256(data).hexdigest()
    def run(argv, **kwargs):
        assert kwargs['input'] == data
        assert 'text' not in kwargs and 'stdin' not in kwargs
        assert argv[-2] == 'sky2' and digest in argv[-1]
        sent.append(argv)
        return subprocess.CompletedProcess(argv, 0, b'', b'')
    monkeypatch.setattr(subprocess, 'run', run)
    assert storage.publish(data, digest) == storage.path(digest)
    assert len(sent) == len(checks) == 1


def test_storage_stage_is_cluster_native_and_rejects_other_hosts():
    class Cluster:
        _run_id = staticmethod(lambda value: str(__import__('uuid').UUID(value)))
        def ssh(self, gateway, command, **kwargs):
            calls.append((gateway, command)); return ''
    calls = []; storage = CaptureStorage(Cluster()); digest = 'a'*64
    identifier = str(uuid4())
    target = storage.stage(digest, 10, identifier, 'sky2')
    assert target.endswith(f'{identifier}/original.jsonl')
    assert len(calls) == 2
    assert 'cp -- ' in calls[-1][1] and 'sha256sum' in calls[-1][1]
    assert storage.path(digest) in calls[-1][1]
    with pytest.raises(ValueError, match='must run on sky2'):
        storage.stage(digest, 10, identifier, 'rl2-bonjour')
    assert len(calls) == 2


def test_processing_archive_transport_preserves_frozen_execution_and_hides_manifest(tmp_path, monkeypatch):
    from skynet_app.capture_processing.service import ProcessingService
    from skynet_app.cluster_config import CLUSTER
    captures = LocalCaptureService(Database(tmp_path / 'test.db'), storage=MemoryStorage())
    capture = captures.import_file('visionpro-local', save(tmp_path, records()))['capture']
    cluster = object(); service = ProcessingService(captures, cluster=cluster)
    monkeypatch.setattr(service, 'dispatch', lambda identifier: None)
    job = service.create(capture['sha256'])
    private = service.get(job['id'], private=True)
    private['config']['pipeline']['execution'] = 'workstation'
    private['config']['pipeline']['gateway'] = 'rl2-bonjour'
    manifest = {'schema':'skynet.live-archive/v1','session_id':job['id'], 'files':[], 'directories':['output']}
    checksum = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
    root = f"{CLUSTER.paths.datasets}/raw/dexverse-live/{job['id']}/{checksum}"
    archive = {'state':'VERIFIED','gateway':'sky2','root':root+'/output',
               'manifest_sha256':checksum,'manifest':manifest}
    service.update(job['id'], state='SUCCEEDED', config=private['config'], archive=archive, root=root, gateway='sky2')
    public = service.get(job['id'])
    assert 'manifest' not in public['archive']
    assert public['config']['pipeline']['execution'] == 'workstation'
    assert service.transport(public) is cluster
    assert service.refresh(job['id'], force=True)['state'] == 'SUCCEEDED'
    assert service.get(job['id'], private=True)['archive']['manifest'] == manifest
    with pytest.raises(ValueError, match='archived'):
        service.retry(job['id'])
    from skynet_app.capture_processing import api
    monkeypatch.setattr(api, 'service', service)
    app = FastAPI(); app.include_router(api.router)
    with TestClient(app) as client:
        assert client.post(f"/api/collection/processing/jobs/{job['id']}/cancel").status_code == 409
        assert client.post(f"/api/collection/processing/jobs/{job['id']}/recover").status_code == 409
    service.update(job['id'], state='FAILED', job_id=None)
    assert service.refresh(job['id'], force=True)['state'] == 'FAILED'
    with pytest.raises(ValueError, match='archived'):
        service.retry(job['id'])
    service.update(job['id'], state='SUCCEEDED')
    broken = dict(archive, root='/unexpected/output')
    service.update(job['id'], archive=broken)
    with pytest.raises(ValueError, match='location'):
        service.transport(service.get(job['id']))
    service.update(job['id'], archive=archive, state='RUNNING')
    with pytest.raises(ValueError, match='not been verified'):
        service.transport(service.get(job['id']))
    service.update(job['id'], state='SUCCEEDED', archive=dict(archive, manifest=dict(manifest, files=[{'path':'unexpected'}])))
    with pytest.raises(ValueError, match='not been verified'):
        service.transport(service.get(job['id']))
