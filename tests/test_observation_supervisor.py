"""Real child-process checks: native stalls, owned-group cleanup and receipts."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from ops.datasets import observation_supervisor as supervisor
from skynet_app.observation_contracts import PREPARE_SCHEMA

ROOT = Path(__file__).resolve().parents[1]

CHILD = r'''
import argparse, ctypes, json, os, signal, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, ROOT)
from ops.datasets.observation_supervisor import ProgressReporter, write_json
p=argparse.ArgumentParser()
p.add_argument('--request');p.add_argument('--result');p.add_argument('--render-child',action='store_true');p.add_argument('--progress')
a=p.parse_args();request=json.loads(Path(a.request).read_text());case=request['case']
progress=ProgressReporter(request,a.progress,interval=0)
if case in ('native_init_hang','first_frame_hang','native_frame_hang','ready_then_hang'):
 signal.signal(signal.SIGTERM,signal.SIG_IGN)
 child=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'])
 write_json(request['pids'],[os.getpid(),child.pid])
if case=='staging_hang':
 signal.signal(signal.SIGTERM,signal.SIG_IGN)
 residue=Path(request['staging_root'])/'.artifact.preparing-123'/'values.hdf5'
 residue.parent.mkdir(parents=True,exist_ok=True);residue.write_bytes(b'partial native arrays')
 write_json(request['pids'],[os.getpid()])
progress('initializing')
if case=='native_init_hang':ctypes.CDLL(None).sleep(60)
if case=='exit_zero':os._exit(0)
if case=='crash':os.kill(os.getpid(),signal.SIGKILL)
if case=='repeat_initializing':
 for i in range(100):progress('initializing');time.sleep(.05)
progress('episode',episode_key='episode')
progress('capturing',episode_key='episode')
if case in ('first_frame_hang','staging_hang'):ctypes.CDLL(None).sleep(60)
for frame in range(1,5):
 progress('frame',episode_key='episode',frame=frame,frames=4)
 if case=='native_frame_hang':ctypes.CDLL(None).sleep(60)
 if case=='success':time.sleep(.12)
artifacts=[dict(artifact_key=j['artifact_key'],path=j['output_dir'],status='READY',manifest_sha256='b'*64) for j in request['requests']]
progress('publishing',artifact_key=request['requests'][0]['artifact_key'])
result=dict(schema=request['schema'],request_id=request['request_id'],attempt_token=request['attempt_token'],state='READY',artifacts=artifacts)
if case=='wrong_attempt':result['attempt_token']='other-attempt'
if case=='failed_partial':result.update(state='FAILED',error='Explicit child failure',artifacts=artifacts[:1])
write_json(a.result,result)
if request.get('ready_marker'):write_json(request['ready_marker'],result)
progress('closing')
if case=='ready_then_hang':ctypes.CDLL(None).sleep(60)
'''


@pytest.fixture
def child_request(tmp_path):
    worker = tmp_path / 'child.py'
    worker.write_text('ROOT=' + repr(str(ROOT)) + '\n' + CHILD)
    request = dict(schema=PREPARE_SCHEMA, mode='render', request_id='request', attempt_token='attempt',
        sources=[dict(episode_key='episode')],
        requests=[dict(artifact_key='a'*64,output_dir=str(tmp_path/'artifact-a')),
                  dict(artifact_key='c'*64,output_dir=str(tmp_path/'artifact-c'))], pids=str(tmp_path/'pids.json'))
    return worker, request, tmp_path/'request.json', tmp_path/'result.json'


def run(child_request, case, **limits):
    worker, request, path, output = child_request
    request['case'] = case
    path.write_text(json.dumps(request))
    result = supervisor.supervise(request, path, output, worker_path=worker,
        initialization_timeout=limits.pop('initialization_timeout', 2),
        frame_timeout=limits.pop('frame_timeout', .4), termination_grace=.1, poll_interval=.01, **limits)
    assert json.loads(output.read_text()) == result
    assert result['request_id'] == request['request_id'] and result['attempt_token'] == request['attempt_token']
    assert not list(output.parent.glob('.render-supervisor-*'))
    return result


def running(pid):
    result = subprocess.run(['ps','-o','stat=','-p',str(pid)], capture_output=True, text=True)
    return bool(result.stdout.strip()) and not result.stdout.strip().startswith('Z')


@pytest.mark.parametrize('case,phase', [('native_init_hang','initializing'),('first_frame_hang','capturing'),('native_frame_hang','frame')])
def test_native_stall_kills_only_its_own_process_group(child_request, case, phase):
    unrelated = subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'], start_new_session=True)
    started = time.monotonic()
    try:
        result = run(child_request, case, initialization_timeout=.4)
        assert time.monotonic()-started < 5
        assert result['state']=='FAILED' and 'stalled during '+phase in result['error']
        pids = json.loads(Path(child_request[1]['pids']).read_text())
        for _ in range(50):
            if not any(running(pid) for pid in pids):break
            time.sleep(.02)
        assert not any(running(pid) for pid in pids), 'Child and descendant must be stopped'
        assert unrelated.poll() is None, 'Supervisor must not signal unrelated processes'
    finally:
        unrelated.terminate();unrelated.wait(timeout=2)


@pytest.mark.parametrize('case', ['exit_zero','crash','wrong_attempt'])
def test_missing_or_mismatched_child_receipt_fails(child_request, case):
    # A prior valid-looking receipt cannot mask this invocation's missing result.
    child_request[3].write_text(json.dumps(dict(state='READY',artifacts=[])))
    result = run(child_request, case)
    assert result['state']=='FAILED'
    assert 'without a result receipt' in result['error'] if case!='wrong_attempt' else 'different attempt' in result['error']


def test_real_progress_extends_frame_deadline_and_success_is_preserved(child_request):
    started=time.monotonic()
    result=run(child_request,'success',frame_timeout=.25)
    assert time.monotonic()-started>.4
    assert result['state']=='READY' and len(result['artifacts'])==2


def test_ready_receipt_survives_stalled_native_shutdown(child_request):
    result=run(child_request,'ready_then_hang',frame_timeout=.2)
    assert result['state']=='READY' and len(result['artifacts'])==2
    assert 'stalled during closing' in result['cleanup_warning']
    assert not running(json.loads(Path(child_request[1]['pids']).read_text())[0])


def test_valid_failed_receipt_keeps_partial_artifacts(child_request):
    result=run(child_request,'failed_partial')
    assert result['state']=='FAILED' and result['error']=='Explicit child failure'
    assert len(result['artifacts'])==1


def test_ready_receipt_requires_confirmed_process_group_stop(child_request, monkeypatch):
    stop = supervisor._stop_group
    def unconfirmed(process, grace):
        stop(process, grace)
        return False
    monkeypatch.setattr(supervisor, '_stop_group', unconfirmed)
    result = run(child_request, 'ready')
    assert result['state'] == 'FAILED'
    assert 'confirm renderer process group stopped' in result['error']


def test_interruption_after_data_commit_does_not_become_success(child_request):
    worker, request, path, output = child_request
    marker = output.parent / 'committed.json'
    request.update(case='ready_then_hang', ready_marker=str(marker))
    path.write_text(json.dumps(request))
    script = ('import sys;sys.path.insert(0,'+repr(str(ROOT))+');'
              'from ops.datasets.observation_supervisor import supervise;'
              'import json;from pathlib import Path;'
              f'supervise(json.loads(Path({str(path)!r}).read_text()),{str(path)!r},{str(output)!r},'
              f'worker_path={str(worker)!r},termination_grace=.1,poll_interval=.01)')
    process = subprocess.Popen([sys.executable, '-c', script], start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert marker.exists()
        process.terminate()
        process.wait(timeout=3)
        result = json.loads(output.read_text())
        assert result['state'] == 'FAILED' and 'interrupted by signal' in result['error']
        assert len(result['artifacts']) == 2
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)


def test_repeated_phase_notifications_cannot_hide_a_stall(child_request):
    result=run(child_request,'repeat_initializing',initialization_timeout=.25)
    assert result['state']=='FAILED' and 'stalled during initializing' in result['error']


def test_parent_termination_stops_child_and_persists_failed_receipt(child_request):
    worker,request,path,output=child_request
    request['case']='native_init_hang';path.write_text(json.dumps(request))
    script=('import json,sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);'
            'from ops.datasets.observation_supervisor import supervise;'
            'supervise(json.loads(Path(sys.argv[2]).read_text()),sys.argv[2],sys.argv[3],'
            'worker_path=sys.argv[4],termination_grace=.1,poll_interval=.01)')
    process=subprocess.Popen([sys.executable,'-c',script,str(ROOT),str(path),str(output),str(worker)],start_new_session=True)
    try:
        until=time.monotonic()+5
        while not Path(request['pids']).exists() and time.monotonic()<until:time.sleep(.02)
        assert Path(request['pids']).exists()
        process.terminate();process.wait(timeout=3)
        result=json.loads(output.read_text())
        assert result['state']=='FAILED' and 'interrupted by signal' in result['error']
        assert result['attempt_token']==request['attempt_token']
        assert not running(json.loads(Path(request['pids']).read_text())[0])
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=2)


def test_worker_progress_follows_completed_frames_and_publication(tmp_path):
    from test_observation_prepare import source, requests, request, FakeRenderer
    from ops.datasets import observation_prepare as worker
    from skynet_app.observation_contracts import rgb_requirements
    src=source(tmp_path);jobs=requests(tmp_path,src,rgb_requirements(['scene_front'],width=3,height=2))
    events=[]
    result=worker.run_request(request([src],jobs),renderer_factory=FakeRenderer,
                              progress=lambda phase,**values:events.append((phase,values)))
    assert result['state']=='READY',result
    assert [phase for phase,_ in events]==['initializing','episode','capturing','frame','frame','publishing','closing']
    assert [values['frame'] for phase,values in events if phase=='frame']==[1,2]


def test_frozen_render_cli_supervises_reuse_without_importing_simulator(tmp_path):
    from types import SimpleNamespace
    from test_observation_prepare import source, requests, request, FakeRenderer
    from ops.datasets import observation_prepare as worker
    from skynet_app.observation_contracts import rgb_requirements
    from skynet_app.observation_preparation import ObservationPreparation
    src=source(tmp_path)
    jobs=requests(tmp_path,src,rgb_requirements(['scene_front'],width=3,height=2))
    original=request([src],jobs)
    assert worker.run_request(original,renderer_factory=FakeRenderer)['state']=='READY'
    capsule=tmp_path/'worker';capsule.mkdir()
    files=ObservationPreparation(SimpleNamespace(live=SimpleNamespace(root=ROOT),database=None)).frozen_files()
    assert 'observation_supervisor.py' in files
    for name,content in files.items():(capsule/name).write_text(content)
    source_path=tmp_path/'request.json';source_path.write_text(json.dumps(original))
    result_path=tmp_path/'result.json'
    completed=subprocess.run([sys.executable,str(capsule/'observation_prepare.py'),'--request',str(source_path),
                              '--result',str(result_path)],capture_output=True,text=True,timeout=15)
    assert completed.returncode==0,completed.stderr
    result=json.loads(result_path.read_text())
    assert result['state']=='READY' and result['attempt_token']==original['attempt_token']
    assert all(item['reused'] for item in result['artifacts'])
    assert not list(tmp_path.glob('.render-supervisor-*'))


def test_stalled_child_staging_is_cleaned_after_owned_group_stops(child_request):
    _, request, _, output = child_request
    staging = output.parent / 'staging'
    original = output.parent / 'recording.pkl'; original.write_bytes(b'original recording')
    published = Path(request['requests'][0]['output_dir']) / 'values.hdf5'
    published.parent.mkdir(); published.write_bytes(b'published arrays')
    request['sources'][0]['recording'] = str(original)
    request['staging_root'] = str(staging)
    result = run(child_request, 'staging_hang', frame_timeout=.2)
    assert result['state'] == 'FAILED' and 'stalled during capturing' in result['error']
    assert not running(json.loads(Path(request['pids']).read_text())[0])
    assert not staging.exists()
    assert original.read_bytes() == b'original recording'
    assert published.read_bytes() == b'published arrays'


@pytest.mark.parametrize('scope', ['outside', 'symlink', 'output', 'output_symlink', 'recording', 'dependency'])
def test_staging_cleanup_preserves_out_of_scope_or_protected_paths(child_request, scope):
    _, request, _, output = child_request
    staging = output.parent / 'staging'
    outside = output.parent / 'elsewhere'; outside.mkdir()
    if scope == 'outside':
        staging = outside
    elif scope == 'symlink':
        staging.symlink_to(outside, target_is_directory=True)
    else:
        staging.mkdir()
    protected = staging / 'protected'
    if scope == 'output_symlink':
        protected.symlink_to(outside, target_is_directory=True)
    else:
        protected.mkdir()
    marker = protected / 'values.bin'; marker.write_bytes(b'keep these bytes')
    if scope in {'output', 'output_symlink'}:
        request['requests'][0]['output_dir'] = str(protected)
    elif scope == 'recording':
        request['sources'][0]['recording'] = str(marker)
    elif scope == 'dependency':
        request['requests'][0]['dependencies'] = [dict(path=str(protected))]
    request['staging_root'] = str(staging)
    result = run(child_request, 'staging_hang', frame_timeout=.2)
    assert result['state'] == 'FAILED'
    assert staging.exists() and marker.read_bytes() == b'keep these bytes'
    assert (staging / '.artifact.preparing-123' / 'values.hdf5').exists()
    if scope == 'symlink':
        assert staging.is_symlink()


def test_unconfirmed_process_group_retains_staging(child_request, monkeypatch):
    _, request, _, output = child_request
    staging = output.parent / 'staging'
    request['staging_root'] = str(staging)
    stop = supervisor._stop_group
    def unconfirmed(process, grace):
        assert stop(process, grace)
        return False
    monkeypatch.setattr(supervisor, '_stop_group', unconfirmed)
    result = run(child_request, 'staging_hang', frame_timeout=.2)
    assert result['state'] == 'FAILED'
    assert (staging / '.artifact.preparing-123' / 'values.hdf5').exists()


def test_valid_ready_receipt_preserves_published_output_inside_staging(child_request):
    _, request, _, output = child_request
    staging = output.parent / 'staging'
    published = staging / 'published'; published.mkdir(parents=True)
    marker = published / 'values.hdf5'; marker.write_bytes(b'published arrays')
    request['staging_root'] = str(staging)
    request['requests'][0]['output_dir'] = str(published)
    result = run(child_request, 'ready')
    assert result['state'] == 'READY' and len(result['artifacts']) == 2
    assert marker.read_bytes() == b'published arrays'


@pytest.mark.parametrize('closing', ['clean', 'exception', 'exit', 'hang'])
def test_actual_worker_commits_files_before_native_teardown(tmp_path, closing):
    from test_observation_prepare import source, requests, request
    from ops.datasets.observation_prepare import verify_artifact
    from skynet_app.observation_contracts import rgb_requirements
    src = source(tmp_path)
    jobs = requests(tmp_path, src, rgb_requirements(['scene_front'], width=3, height=2))
    req = request([src], jobs)
    req['staging_root'] = str(tmp_path / 'staging')
    req['closing'] = closing
    path = tmp_path / 'request.json'; path.write_text(json.dumps(req))
    output = tmp_path / 'result.json'
    child = tmp_path / 'real_worker.py'
    child.write_text('ROOT=' + repr(str(ROOT)) + '\n' + r'''
import ctypes, json, os, signal, sys, types
from pathlib import Path
sys.path[:0] = [ROOT, ROOT + '/tests']
from test_observation_prepare import FakeRenderer
from ops.datasets import observation_prepare as worker
request = json.loads(Path(sys.argv[sys.argv.index('--request') + 1]).read_text())
result_path = Path(sys.argv[sys.argv.index('--result') + 1])
class ClosingRenderer(FakeRenderer):
 def __enter__(self):
  residue = Path(request['staging_root']) / 'observation-hand-test' / 'robot.usd'
  residue.parent.mkdir(parents=True); residue.write_text('private generated asset')
  return super().__enter__()
 def __exit__(self, *args):
  saved = json.loads(result_path.read_text())
  assert saved['state'] == 'READY' and len(saved['artifacts']) == 1
  self.progress('closing_app')
  if request['closing'] == 'exception': raise RuntimeError('native teardown failed')
  if request['closing'] == 'exit': os._exit(0)
  if request['closing'] == 'hang':
   signal.signal(signal.SIGTERM, signal.SIG_IGN)
   ctypes.CDLL(None).sleep(60)
module = types.ModuleType('observation_render'); module.DexVerseRenderer = ClosingRenderer
sys.modules['observation_render'] = module
raise SystemExit(worker.main())
''')
    result = supervisor.supervise(req, path, output, worker_path=child,
        initialization_timeout=5, frame_timeout=.3, termination_grace=.1, poll_interval=.01)
    assert result['state'] == 'READY', result
    assert len(result['artifacts']) == 1
    verify_artifact(jobs[0]['output_dir'], request=jobs[0])
    assert not Path(req['staging_root']).exists(), 'Supervisor removes private assets after child exit'
    if closing in {'exception', 'hang'}:
        assert result.get('cleanup_warning')
    else:
        assert not result.get('cleanup_warning')
