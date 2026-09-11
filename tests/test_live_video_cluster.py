import json
from types import SimpleNamespace

import pytest

from skynet_app.cluster_runtime import WORK_ROOT, ClusterError, SubmissionOutcomeUnknown
from skynet_app.live_xr_video import VideoGeneration
from skynet_app.live_xr_video_cluster import control_cluster


@pytest.fixture
def cluster_video():
    generation = VideoGeneration()
    root = WORK_ROOT + '/jobs/runs/' + generation.token
    state = {'scheduler': 'PENDING', 'metadata': {}}
    calls = []
    profile = {'account':'overcap','partition':'overcap','gpu_type':'any',
               'runtime':WORK_ROOT+'/runtime','repository':WORK_ROOT+'/repo'}
    transport = SimpleNamespace(
        write_capsule_file=lambda *a:calls.append(('write',*a)),
        submit_script=lambda *a,**kw:(calls.append(('submit',a,kw)) or SimpleNamespace(job_id='123')),
        recover_submission=lambda *a:SimpleNamespace(job_id='123'),
        job_statuses=lambda *a:('sky2',{'123':{'State':state['scheduler']}}),
        cancel=lambda *a:calls.append(('cancel',*a)),
        ssh=lambda *a,**kw:json.dumps(state['metadata']),
    )
    job={'gateway':'sky2'}
    def control(operation):
        return control_cluster(transport,job,generation,root,operation,
            profile=profile,request={'recording':WORK_ROOT+'/raw/demo.pkl'},sources={'render_recording.py':'pass'})
    return control,generation,state,calls,transport,root


def test_replay_submits_once_and_waits_for_confirmed_render_and_shutdown(cluster_video):
    control,generation,state,calls,_,root=cluster_video
    assert control('start')['state']=='STARTING'
    assert control('status')=={'state':'STARTING','phase':'queued','job_id':'123'}
    state.update(scheduler='RUNNING',metadata={'state':'RENDERING'})
    assert control('status')['phase']=='rendering'
    state['metadata']={'state':'READY','path':root+'/video.mp4'}
    assert control('status')['phase']=='finalizing'
    state['scheduler']='COMPLETED'
    assert control('status')==state['metadata']
    submissions=[v for v in calls if v[0]=='submit']
    assert len(submissions)==1
    script=submissions[0][1][0]
    assert '#SBATCH --time=00:10:00' in script and '#SBATCH --gres=gpu:1' in script
    assert submissions[0][2]['submission_key']==generation.token


def test_cancel_requires_scheduler_confirmation_and_exact_owned_job(cluster_video):
    control,_,state,calls,_,_=cluster_video
    control('start')
    with pytest.raises(ValueError,match='confirm video cancellation'):
        control('cancel')
    assert ('cancel','123','sky2') in calls
    state['scheduler']='CANCELLED'
    assert control('cancel')['state']=='CANCELLED'


def test_queued_cancel_prevents_submission(cluster_video):
    control,generation,_,calls,_,_=cluster_video
    generation.cancel.set()
    assert control('start')['state']=='CANCELLED'
    assert control('cancel')['state']=='CANCELLED'
    assert calls==[]


def test_lost_submission_cannot_be_declared_cancelled(cluster_video):
    control,generation,_,_,transport,_=cluster_video
    transport.submit_script=lambda *a,**k:(_ for _ in ()).throw(SubmissionOutcomeUnknown('lost'))
    transport.recover_submission=lambda *a:None
    with pytest.raises(SubmissionOutcomeUnknown): control('start')
    assert generation.cluster_submission_started
    with pytest.raises(ValueError,match='reconciled'): control('cancel')


def test_definitive_submission_rejection_allows_cleanup_without_a_job(cluster_video):
    control,generation,_,_,transport,_=cluster_video
    transport.submit_script=lambda *a,**k:(_ for _ in ()).throw(ClusterError('rejected'))
    with pytest.raises(ClusterError): control('start')
    assert not generation.cluster_submission_started
    assert control('cancel')['state']=='CANCELLED'


def test_completed_metadata_must_belong_to_generation(cluster_video):
    control,_,state,_,_,_=cluster_video
    control('start')
    state.update(scheduler='COMPLETED',metadata={'state':'READY','path':'/other/video.mp4'})
    with pytest.raises(ValueError,match='does not match'): control('status')


def test_repeated_start_reconciles_without_rewriting_or_resubmitting(cluster_video):
    control,_,_,calls,_,_=cluster_video
    control('start')
    existing=list(calls)
    assert control('start')['phase']=='queued'
    assert calls==existing


def test_allocated_job_startup_is_not_reported_as_queued(cluster_video):
    control,_,state,_,_,_=cluster_video
    control('start')
    state['scheduler']='RUNNING'
    assert control('status')['phase']=='starting'


def test_submission_boundary_is_persisted_before_job_can_be_accepted(cluster_video):
    control,generation,_,_,transport,_=cluster_video
    states=[]
    generation.persist_submission=lambda:states.append((generation.cluster_submission_started,generation.cluster_job_id))
    def submit(*args,**kwargs):
        assert states==[(True,None)]
        return SimpleNamespace(job_id='123')
    transport.submit_script=submit
    control('start')
    assert states==[(True,None),(True,'123')]


def test_failed_submission_boundary_persistence_never_submits(cluster_video):
    control,generation,_,calls,_,_=cluster_video
    generation.persist_submission=lambda:(_ for _ in ()).throw(OSError('disk unavailable'))
    with pytest.raises(OSError): control('start')
    assert not generation.cluster_submission_started
    assert not any(call[0]=='submit' for call in calls)
    assert control('cancel')['state']=='CANCELLED'


def test_recovered_job_id_must_match_exact_submission_receipt(cluster_video):
    control,generation,_,calls,transport,_=cluster_video
    generation.cluster_job_id='456'
    generation.cluster_submission_started=True
    with pytest.raises(ValueError,match='does not match its submission receipt'):
        control('cancel')
    assert not any(call[0]=='cancel' for call in calls)
