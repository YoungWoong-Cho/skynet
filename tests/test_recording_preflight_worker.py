"""CPU preflight and atomic manifest publication regressions."""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

from test_recording_preparation_service import preparation,create
from skynet_app.cluster_runtime import ClusterError,SubmissionOutcomeUnknown
from skynet_app.observation_preparation import ObservationsPending


def test_preflight_retains_submission_identity_after_uncertain_ack(preparation):
    service,*_=preparation;job=create(preparation)
    sent=[];uploads=[]
    def submit(script,identifier,gateway,submission_key):
        sent.append((script,submission_key))
        if len(sent)==1:raise SubmissionOutcomeUnknown('lost acknowledgement')
        return SimpleNamespace(job_id='123')
    service.cluster=SimpleNamespace(write_capsule_files=lambda *a:uploads.append(a),submit_script=submit,
        job_statuses=lambda *a:('sky2',{'123':{'State':'PENDING'}}))
    with pytest.raises(SubmissionOutcomeUnknown):service._preflight_sources(job,job['sources'])
    uncertain=service.update(job['id'],state='SUBMISSION_UNKNOWN')
    assert service.retry(job['id'])['preflight_token']==uncertain['preflight_token']
    with pytest.raises(ObservationsPending):service._preflight_sources(service.get(job['id']),job['sources'])
    assert len(uploads)==1 and len(sent)==2 and sent[0]==sent[1]
    assert '--gres' not in sent[0][0] and 'CUDA_VISIBLE_DEVICES=' in sent[0][0]
    assert service.get(job['id'])['preflight_job_id']=='123'


def test_preflight_bootstrap_failure_does_not_wait_forever(preparation):
    service,*_=preparation;job=create(preparation)
    job=service.update(job['id'],preflight_script='saved',preflight_token='token',preflight_root='/test',preflight_job_id='123')
    def read(path,*a,**kw):
        if path.endswith('result.json'):raise ClusterError('No such file')
        return 'sky2','interpreter missing package'
    service.cluster=SimpleNamespace(job_statuses=lambda *a:('sky2',{'123':{'State':'FAILED'}}),read_file=read)
    with pytest.raises(ValueError,match='FAILED.*missing package'):
        service._preflight_sources(job,job['sources'])


def test_preflight_verifies_source_membership_and_reuses_evidence(preparation):
    service,*_=preparation;job=create(preparation)
    job=service.update(job['id'],preflight_script='saved',preflight_token='token',preflight_root='/test',preflight_job_id='123')
    result=dict(schema='skynet.recording-preflight/v1',job_id=job['id'],attempt_id='token',verified=True,
        sources=[dict(source_sha256=s['sha256'],shared_image_streams={},capture={'robot':'test'},steps=2) for s in job['sources']])
    reads=[]
    service.cluster=SimpleNamespace(job_statuses=lambda *a:('sky2',{'123':{'State':'COMPLETED'}}),
        read_file=lambda *a,**kw:(reads.append(a) or 'sky2',json.dumps(result)))
    sources=service._preflight_sources(job,job['sources'])
    assert sources[0]['capture']['robot']=='test'
    again=service._preflight_sources(service.get(job['id']),job['sources'])
    assert sources==again and len(reads)==1
    path=service.root/job['id']/'source-preflight.json'
    path.write_text('{}')
    with pytest.raises(ValueError,match='changed'):
        service._preflight_sources(service.get(job['id']),job['sources'])


def test_cluster_worker_publishes_manifest_only_and_validates_reader(tmp_path,monkeypatch):
    from test_recording_dataset import source,request,digest
    from skynet_app import policy_export_worker as worker
    raw,_,_=source(tmp_path)
    req=request(tmp_path,[raw])
    req.update(job_id='test',attempt_id='one',prepared_root=str(tmp_path/'prepared'),receipt_path=str(tmp_path/'result.json'))
    script=tmp_path/'loader.py'
    script.write_text('''import sys,json\nfrom recording_dataset import RecordingDataset\ndataset=RecordingDataset(sys.argv[1],sys.argv[2],verify_files=True)\nassert dataset.read(0,"action").shape==(4,7)\nprint(json.dumps(dict(schema="test-reader/v1",manifest_sha256=sys.argv[2],observation_mode="state")))\n''')
    root=Path(__file__).resolve().parents[1]
    monkeypatch.setenv('PYTHONPATH',str(root/'skynet_app/adapters'))
    req['loader']=dict(argv=[sys.executable,str(script),'{dataset}','{manifest_sha256}'],schemas=['test-reader/v1'],mode='state')
    original=digest(raw['recording'])
    result=worker.run(req)
    destination=Path(result['path'])
    assert result['verified'] is True and result['loader_validation']['schema']=='test-reader/v1'
    assert [p.name for p in destination.iterdir()]==['manifest.json']
    assert not Path(req['output']).exists() and not list(tmp_path.rglob('*.zip'))
    assert digest(raw['recording'])==original
    state=list((tmp_path/'recordings').rglob('values.hdf5'))
    assert len(state)==1
    before=state[0].stat().st_mtime_ns
    second=worker.run(req)
    assert second['path']==result['path'] and state[0].stat().st_mtime_ns==before
    req['loader']['schemas']=['incorrect/v1']
    with pytest.raises(ValueError,match='different dataset'):worker.run(req)
