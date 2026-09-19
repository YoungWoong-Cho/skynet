"""Real catalog and file ownership for deletion tests, with no model loader."""
import hashlib
import json
from types import SimpleNamespace

from test_recording_preparation_service import preparation, create
from skynet_app import dataset_cleanup, policy_exports


def published(preparation, tmp_path, monkeypatch, preset='state'):
    service, session, *_ = preparation
    root = tmp_path / 'cluster'
    monkeypatch.setattr(policy_exports, 'WORK_ROOT', str(root))
    job = create(preparation, preset)
    manifest = dict(format=job['format'], contract=job['contract'], adapter=job['adapter'],
        episodes=[dict(id=str(i), source=dict(sha256=s['sha256'])) for i,s in enumerate(job['sources'])],
        steps=4, split=job['split'], validation=dict(status='PASSED'))
    raw=json.dumps(manifest).encode()
    digest=hashlib.sha256(raw).hexdigest()
    destination=root/'datasets/prepared'/digest
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'manifest.json').write_bytes(raw)
    capsule=root/'jobs/runs'/job['id']
    capsule.mkdir(parents=True,exist_ok=True)
    (capsule/'log.txt').write_text('preparation log')
    raw_source=root/'datasets/raw/original.pkl'
    raw_source.parent.mkdir(parents=True,exist_ok=True)
    raw_source.write_bytes(b'original recording; must remain byte-for-byte intact')
    shared=root/'datasets/recordings'/'shared.hdf5'
    shared.parent.mkdir(parents=True,exist_ok=True)
    shared.write_bytes(b'shared stream retained for other adapters')
    version=service.register(job,manifest,digest,remote_path=str(destination))
    service.database.record_data_location(version['id'],kind='cluster',host='skynet',path=str(destination),manifest_sha256=digest)
    job=service.update(job['id'],version_id=version['id'],manifest_sha256=digest,
                       state='READY',training_ready=True)
    # Run the actual guarded filesystem cleanup with its serialized plan.
    import shlex
    def ssh(host,command,timeout):
        payload=json.loads(shlex.split(command)[-1])
        return json.dumps(dataset_cleanup.cleanup(**payload))
    service.cluster=SimpleNamespace(candidates=lambda _:['test'],resolve_gateway=lambda x:x,ssh=ssh)
    return service,job,destination,capsule,raw_source,shared
