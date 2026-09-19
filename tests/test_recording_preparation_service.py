"""Adapter selection, CPU preflight and publication use one recording pipeline."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from skynet_app.database import Database, canonical_json
from test_policy_exports import fixture_recording_manifest
from skynet_app.dataset_formats import resolve_adapter
from skynet_app.live_xr_review import LiveReviewService
from skynet_app.policy_exports import PolicyExportService, DATASET_FORMAT
from skynet_app.observation_preparation import ObservationsPending
from skynet_app.observation_store import ObservationStore

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    db = Database(tmp_path / 'recording.store')
    declared = fixture_recording_manifest().model_dump(mode='json')
    adapter = db.create_adapter(name='Synthetic shared-data test', manifest=declared)
    version = adapter['latest_version']
    session = dict(id=str(uuid4()), state='STOPPED', root='/recording-source', created_at='2026-09-19',
        gateway='sky2', profile=dict(display_name='Recording', task='test-task', robot='floating_shadow_right'),
        recordings=['recordings/live/episode-1.pkl','recordings/live/episode-2.pkl'],
        recording_checksums={'recordings/live/episode-1.pkl':'a'*64,'recordings/live/episode-2.pkl':'b'*64},
        recording_images={})
    with db.transaction() as c:
        c.execute('INSERT INTO live_xr_sessions VALUES (?,?)',(session['id'],canonical_json(session)))
    live = SimpleNamespace(root=ROOT,database=db,get=lambda _:session,list=lambda **_: [session])
    reviews = LiveReviewService(live, root=tmp_path/'reviews')
    monkeypatch.setattr(reviews,'source_for_session',lambda *a:None)
    cluster = SimpleNamespace(candidates=lambda _:['sky2'])
    service = PolicyExportService(reviews,root=tmp_path/'exports',cluster=cluster)
    dispatched=[]
    monkeypatch.setattr(service,'dispatch',dispatched.append)
    yield service,session,adapter,version,dispatched
    service.stop()


def create(preparation,preset='state',**kwargs):
    service,session,adapter,version,_=preparation
    return service.create(session['id'],adapter['id'],'Dataset',
        adapter_version_id=version['id'],adapter_data_preset=preset,**kwargs)


def test_selection_pins_adapter_and_only_declared_observations(preparation):
    service,session,adapter,version,dispatched=preparation
    state=create(preparation)
    rgb=create(preparation,'rgb')
    assert state['format']==rgb['format']==DATASET_FORMAT
    assert state['adapter_id']==adapter['id'] and state['adapter_version_id']==version['id']
    assert state['observation_contract']['streams']==[]
    assert len(rgb['observation_contract']['streams'])==3
    assert state['resource_id']==rgb['resource_id']
    assert create(preparation)['id']==state['id']
    assert dispatched==[state['id'],rgb['id']]
    files=list((service.root/state['id']/'worker').rglob('*'))
    assert not any(p.name in {'policy_export.py','formats.json','egoverse_export.py'} for p in files)
    assert (service.root/state['id']/'worker/recording_prepare.py').exists()
    assert (service.root/state['id']/'worker/recording_dataset.py').exists()
    assert not list(service.root.rglob('dataset.zip'))


def test_wrong_version_and_recipe_ids_are_rejected(preparation):
    service,session,adapter,version,dispatched=preparation
    with pytest.raises(ValueError):
        service.create(session['id'],adapter['id'],'Dataset',adapter_version_id=str(uuid4()))
    with pytest.raises((ValueError,KeyError)):
        service.create(session['id'],'unregistered-adapter','Dataset',adapter_version_id=version['id'])
    assert not dispatched and service.list()==[]


def test_ready_job_is_reused_without_zip(preparation):
    service,*_=preparation
    job=create(preparation)
    service.update(job['id'],state='READY',version_id='saved-version')
    assert create(preparation)['id']==job['id']
    assert service.retry(job['id'])['state']=='READY'


def test_pure_options_do_not_schedule_or_read_remote_payloads(preparation,monkeypatch):
    service,session,*_=preparation
    monkeypatch.setattr(service,'dispatch',lambda _:pytest.fail('Options scheduled conversion'))
    options=service.preparation_options(session['id'])
    assert 'policies' not in options and 'formats' not in options
    assert options['adapters'][0]['available']
    assert service.preparation_options(session['id'])['session']['eligible']


def test_legacy_job_is_not_resumed_and_cannot_be_retried(preparation):
    service,*_=preparation
    job=create(preparation)
    service.update(job['id'],format='egoverse',state='FAILED')
    assert service.job_overview()['exports']==[]
    with pytest.raises(ValueError,match='retired'):
        service.retry(job['id'])


def test_camera_preparation_skips_imported_rgb(preparation,monkeypatch):
    service,*_=preparation
    job=create(preparation,'rgb')
    refs={stream['name']:{'path':'/archive/capture.hdf5','dataset':'images/'+stream['name']}
          for stream in job['observation_contract']['streams']}
    source=dict(job['sources'][0],shared_image_streams=refs)
    monkeypatch.setattr(service.observations,'profile',lambda _:pytest.fail('Unneeded simulator profile'))
    monkeypatch.setattr(service.observations,'tick',lambda:None)
    assert service.observations.ensure(job,[source])[0]['shared_image_streams']==refs
    assert service.get(job['id'])['observation_progress']['reused']==3


def test_register_shared_guards_source_ownership_and_tracks_consumer(preparation,monkeypatch,tmp_path):
    service,*_=preparation
    import skynet_app.cluster_runtime as runtime
    monkeypatch.setattr(runtime,'WORK_ROOT',str(tmp_path))
    job=create(preparation)
    source=job['sources'][0]
    path=str(tmp_path/'datasets/recordings'/source['sha256']/'state'/('c'*64)/'values.hdf5')
    spec=dict(schema='skynet.recording-file/v1',source_sha256=source['sha256'],sha256='d'*64,path=path,owned=True)
    key=hashlib.sha256(canonical_json(spec).encode()).hexdigest()
    receipt=dict(artifact_key=key,spec=spec,path=path,manifest_sha256='d'*64,source_sha256=source['sha256'])
    store=service.observations.store
    store.register_shared(job,[receipt])
    store.register_shared(job,[receipt])
    assert list(store.for_job(job['id']))==[key]
    with service.database.connection() as c:
        assert c.execute('SELECT count(*) FROM observation_sources').fetchone()[0]==1
    invalid=deepcopy(receipt)
    invalid['spec']['path']='/outside/values.hdf5'
    invalid.update(path='/outside/values.hdf5',artifact_key=hashlib.sha256(canonical_json(invalid['spec']).encode()).hexdigest())
    with pytest.raises(ValueError,match='outside'):
        store.register_shared(job,[invalid])
