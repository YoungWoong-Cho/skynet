"""Exercise the actual HTTP boundary, not only the tracking metadata mock."""
import json
import pytest
from skynet_app.tracking import WandBBridge, WandBSettings


@pytest.mark.parametrize('status,exitcode', [('FINISHED', 0), ('FAILED', 1), ('KILLED', 1)])
def test_completion_uses_sdk_file_stream_protocol(monkeypatch, status, exitcode):
    requests = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def read(self): return b'{}'
    def capture(request, **_kwargs):
        requests.append(request)
        return Response()
    monkeypatch.setattr('urllib.request.urlopen', capture)
    bridge = object.__new__(WandBBridge)
    bridge.settings = WandBSettings(api_key='test-key', entity='team')
    bridge._upsert_run = lambda **_kwargs: None
    state = {'runs': {'local': {'name': 'remote', 'entity': 'team', 'project': 'test',
        'storage_id': 'storage', 'summary': {}, 'tags': {}, 'config': {}, 'history_offset': 19}}}
    bridge._deliver_event({'operation': 'finish_run', 'payload': {'local_run_id': 'local', 'status': status}}, state)
    assert len(requests) == 1
    assert requests[0].full_url == 'https://api.wandb.ai/files/team/test/remote/file_stream'
    assert json.loads(requests[0].data) == {'complete': True, 'exitcode': exitcode}
    assert state['runs']['local']['history_offset'] == 19
    assert state['runs']['local']['completion_sent'] == ('finished' if exitcode == 0 else 'failed')


def test_late_metrics_get_fresh_completion_but_unchanged_retry_is_noop(tmp_path, monkeypatch):
    from test_tracking import FakeWandBBridge
    bridge=FakeWandBBridge(tmp_path,WandBSettings(api_key='test',entity='team',auto_flush=False))
    bridge.ensure_run(entity='team',project='test',local_run_id='run',run_name='run',group='test')
    bridge.drain_spool()
    completions=[]
    monkeypatch.setattr(bridge,'_post_file_stream',lambda run,payload:completions.append(payload))
    bridge.finish_run('run',idempotency_key='final');bridge.drain_spool()
    bridge.finish_run('run',idempotency_key='final')
    assert bridge.drain_spool().attempted==0
    bridge.log_metrics('run',{'late/final_loss':.2},step=8000,idempotency_key='late-metric')
    bridge.finish_run('run',idempotency_key='final');bridge.drain_spool()
    assert completions==[{'complete':True,'exitcode':0}]*2
    assert bridge.binding('run')['summary']['late/final_loss']==.2
    bridge.finish_run('run',idempotency_key='final')
    assert bridge.drain_spool().attempted==0
