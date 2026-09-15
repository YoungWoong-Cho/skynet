import hashlib
import json
import pickle
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from skynet_app.live_xr_review import LiveReviewService
from skynet_app.remote_artifacts import RemoteArtifact
from test_live_review import payload


class Files:
    def file_size(self, path, gateway):
        return gateway, Path(path).stat().st_size

    def stream_file_range(self, path, gateway, *, start, end):
        yield Path(path).read_bytes()[start:end + 1]

    def ssh(self, gateway, command, *, stdin=None, timeout=30):
        return subprocess.run(shlex.split(command), input=stdin, text=True, capture_output=True, check=True).stdout


def test_remote_review_preserves_checksums_without_mac_payloads(tmp_path, payload):
    cluster = tmp_path / 'cluster'
    cluster.mkdir()
    raw = pickle.dumps(payload)
    (cluster / 'recording.pkl').write_bytes(raw)
    job = {'id': 'session', 'state':'CAPTURED', 'root':'/bonjour/session', 'gateway':'bonjour',
           'profile': {'task':payload['task'], 'robot':payload['robot_type']},
           'recordings':['recordings/demo.pkl'], 'recording_checksums':{'recordings/demo.pkl':hashlib.sha256(raw).hexdigest()},
           'archive':{'state':'READY','gateway':'sky2'}}
    transport = Files()
    archive = SimpleNamespace(cluster=transport,
        resolve=lambda j,p:(transport,'sky2',str(cluster / 'recording.pkl')),
        derived_root=lambda j:str(cluster / 'derived'))
    live = SimpleNamespace(root=tmp_path/'mac', get=lambda _:job, archive=archive)
    reviews = LiveReviewService(live)
    reviews.prepare('session',0)
    assert reviews.status('session',0)['state'] == 'READY'
    assert reviews.status('session',0)['recording_source'] == {'gateway':'sky2','path':str(cluster/'recording.pkl')}
    result=json.loads(reviews.artifact('session',0,'review.json').read_text())
    assert result['sha256']==hashlib.sha256(raw).hexdigest()
    assert len(result['episodes'][0]['frames'])==4
    assert reviews.artifact('session',0,'recording.pkl').read_bytes()==raw
    local=list((tmp_path/'mac').rglob('*'))
    assert [p.name for p in local if p.is_file()] == ['status.json']
    # A corrupt original cannot be published as a valid remote review.
    job['recording_checksums']['recordings/demo.pkl']='0'*64
    reviews.prepare('session',0)
    assert reviews.status('session',0)['state']=='FAILED'
    reviews.executor.shutdown()


def test_remote_artifact_streams_ranges_without_a_cache(tmp_path):
    path=tmp_path/'remote.mp4';path.write_bytes(b'0123456789')
    artifact=RemoteArtifact(Files(),'sky2',str(path),100)
    app=FastAPI()
    @app.get('/video')
    def get(request:Request):
        return artifact.response(request,'video/mp4')
    with TestClient(app) as client:
        response=client.get('/video',headers={'Range':'bytes=2-5'})
        assert response.status_code==206 and response.content==b'2345'
        assert response.headers['content-range']=='bytes 2-5/10'
        assert response.headers['cache-control']=='no-store'
        assert client.get('/video',headers={'Range':'bytes=-2'}).content==b'89'
        for value in ['bytes=20-30','bytes=-0','bytes=5-3','bytes=0-1,3-4']:
            assert client.get('/video',headers={'Range':value}).status_code==416
    assert list(tmp_path.iterdir())==[path]


@pytest.fixture
def remote_video(tmp_path):
    from skynet_app.live_xr_video import LiveVideoService
    cluster=tmp_path/'cluster'
    original=cluster/'original/output'
    derived=cluster/'derived'
    job={'id':'session','root':'/bonjour/sessions/session','gateway':'bonjour',
         'profile':{},'recordings':['recordings/demo.pkl'],
         'archive':{'state':'READY','gateway':'sky2','root':str(original),'source_removed':True}}
    transport=Files()
    def resolve(job,relative):
        assert relative in {'recordings/demo.pkl','recordings/demo.mp4'}
        return transport,'sky2',str(original/relative)
    archive=SimpleNamespace(cluster=transport,resolve=resolve,derived_root=lambda job:str(derived))
    live=SimpleNamespace(get=lambda key:job,archive=archive,
                         conversions=SimpleNamespace(profile=lambda job:{'execution':'slurm'}))
    summary={'sha256':'a'*64,'episodes':[{'steps':2}]}
    directory=tmp_path/'mac/session/0'
    reviews=SimpleNamespace(live=live,status=lambda *args:{'state':'READY','summary':summary},
                            source=lambda *args:(job,str(original/'recordings/demo.pkl')),
                            directory=lambda *args:directory)
    videos=LiveVideoService(reviews)
    videos.version='a'*16
    yield SimpleNamespace(videos=videos,job=job,transport=transport,original=original,derived=derived,
                          directory=directory/'video-0')
    videos.executor.shutdown(wait=True)


@pytest.mark.parametrize('legacy_derived',[False,True])
def test_archive_and_legacy_cached_video_play_without_loading_recording_states(remote_video,legacy_derived):
    c=remote_video
    video=(c.derived/'cached-videos/0/0'/'a.mp4') if legacy_derived else c.original/'recordings/demo.mp4'
    video.parent.mkdir(parents=True)
    video.write_bytes(b'cached MP4')
    # Legacy READY receipts may have no generation. Summary-only video lookup
    # must not need the full review states or a local video file.
    c.videos.publish(c.directory,state='READY',kind='capture',remote_artifact={'gateway':'sky2','path':str(video)})
    artifact=c.videos.artifact('session',0)
    assert isinstance(artifact,RemoteArtifact) and artifact.read_bytes()==b'cached MP4'
    assert [path.name for path in c.directory.iterdir()]==['status.json']


def test_recovered_pre_submission_stage_is_safely_cancellable_without_a_job(remote_video):
    from skynet_app.cluster_runtime import WORK_ROOT
    c=remote_video
    token='a'*32
    c.transport.recover_submission=lambda *args:pytest.fail('No submission boundary was crossed')
    c.videos.publish(c.directory,state='STARTING',generation=token,cluster_submission_started=False,
                     remote_root=f'{WORK_ROOT}/jobs/runs/{token}')
    assert c.videos.cancel('session',0)['state']=='CANCELLED'


def test_archived_workstation_receipt_cannot_recreate_deleted_source_for_cancel(remote_video):
    c=remote_video
    c.videos.publish(c.directory,state='PREPARING',generation='a'*32,
        remote_root=c.job['root']+'/output/review-videos/'+'a'*64+'/0/'+'a'*16+'/attempts/'+'a'*32)
    c.videos._control=lambda *args,**kwargs:pytest.fail('Archive already proved this workstation process stopped')
    assert c.videos.status('session',0)['state']=='NOT_PREPARED'
    assert c.videos.cancel('session',0)['state']=='NOT_PREPARED'


def test_review_and_video_creation_pause_while_source_is_being_archived(remote_video):
    c=remote_video
    c.job['archive']['state']='COPYING'
    reviews=LiveReviewService(c.videos.live,root=c.directory.parent)
    with pytest.raises(ValueError,match='moving to sky2'):
        reviews.create('session',0)
    with pytest.raises(ValueError,match='moving to sky2'):
        c.videos.create('session',0)
    reviews.executor.shutdown()


def test_review_storage_change_cannot_publish_split_location_as_ready(tmp_path,payload):
    raw=pickle.dumps(payload)
    source=tmp_path/'source.pkl';source.write_bytes(raw)
    original=tmp_path/'bonjour/session'
    derived=tmp_path/'sky2/derived'
    job={'id':'session','root':str(original),'gateway':'bonjour',
         'profile':{'task':payload['task'],'robot':payload['robot_type']},
         'recordings':['recordings/demo.pkl'],'recording_checksums':{'recordings/demo.pkl':hashlib.sha256(raw).hexdigest()}}
    transport=Files()
    archive=SimpleNamespace(cluster=transport,
        resolve=lambda job,p:(transport,'sky2',str(source)),derived_root=lambda job:str(derived))
    writes=[]
    ssh=transport.ssh
    def changed(gateway,command,**kwargs):
        writes.append((gateway,shlex.split(command)[-1]))
        result=ssh(gateway,command,**kwargs)
        if len(writes)==1:
            job['archive']={'state':'READY','gateway':'sky2'}
        return result
    transport.ssh=changed
    live=SimpleNamespace(root=tmp_path/'mac',get=lambda key:job,archive=archive,transport=lambda job:transport)
    reviews=LiveReviewService(live)
    reviews.prepare('session',0)
    assert reviews.status('session',0)['state']=='READY'
    assert [gateway for gateway,path in writes]==['bonjour','bonjour','sky2','sky2']
    assert (derived/'reviews/0/review.json').is_file() and (derived/'reviews/0/summary.json').is_file()
    assert reviews.status('session',0)['storage_path']==str(derived/'reviews/0/review.json')
    reviews.executor.shutdown()
