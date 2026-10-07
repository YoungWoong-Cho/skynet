"""Metric uploads retain every sample and back off across bridge instances."""
import io
import json
import urllib.error
from email.message import Message

import pytest

from test_tracking import FakeWandBBridge
from skynet_app.tracking import WandBBridge, WandBSettings


class WireBridge(FakeWandBBridge):
    _append_history_rows = WandBBridge._append_history_rows
    _post_file_stream = WandBBridge._post_file_stream


def bridge(tmp_path):
    return WireBridge(tmp_path, WandBSettings(api_key="fake", entity="team", auto_flush=False))


def create_run(client):
    client.ensure_run(entity="team", project="test", local_run_id="run", run_name="test", group="test")
    assert client.drain_spool().remaining == 0


class Response:
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self): return b'{}'


def test_many_metrics_share_requests_without_losing_steps(tmp_path, monkeypatch):
    calls = []
    def send(request, **kwargs):
        calls.append(json.loads(request.data)["files"]["wandb-history.jsonl"])
        return Response()
    monkeypatch.setattr("skynet_app.tracking.urllib.request.urlopen", send)
    client = bridge(tmp_path)
    create_run(client)
    for step in range(230):
        client.log_metrics("run", {"train/loss": step / 1000}, step=step, timestamp_ms=100000 + step)
    report = client.drain_spool()
    assert report.delivered == 230 and report.remaining == 0
    assert [len(c["content"]) for c in calls] == [100, 100, 30]
    assert [c["offset"] for c in calls] == [0, 100, 200]
    rows = [json.loads(line) for call in calls for line in call["content"]]
    assert [r["_step"] for r in rows] == list(range(230))
    assert [r["train/loss"] for r in rows] == [i / 1000 for i in range(230)]
    assert client.binding("run")["history_offset"] == 230
    assert client.drain_spool().attempted == 0


def test_rate_limit_waits_persists_and_retries_same_offset(tmp_path, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("skynet_app.tracking.time.time", lambda: now[0])
    calls = []
    def send(request, **kwargs):
        calls.append(json.loads(request.data)["files"]["wandb-history.jsonl"])
        if len(calls) == 1:
            headers = Message(); headers["Retry-After"] = "120"
            raise urllib.error.HTTPError(request.full_url, 429, "rate limited", headers, io.BytesIO(b'{}'))
        return Response()
    monkeypatch.setattr("skynet_app.tracking.urllib.request.urlopen", send)
    client = bridge(tmp_path)
    create_run(client)
    for step in range(3):
        client.log_metrics("run", {"loss": step}, step=step, timestamp_ms=100000 + step)
    first = client.drain_spool()
    assert first.remaining == 3 and first.delivered == 0
    assert first.errors == ("W&B is busy. Saved metrics will sync automatically.",)
    assert client.binding("run")["history_offset"] == 0
    # A new bridge (or app restart) observes the same persisted cooldown.
    remote = client.remote_runs
    client = bridge(tmp_path)
    client.remote_runs = remote
    now[0] = 1119
    assert client.drain_spool().attempted == 0
    assert len(calls) == 1
    now[0] = 1121
    assert client.drain_spool().remaining == 0
    assert calls[0] == calls[1]
    assert client.binding("run")["history_offset"] == 3


def test_summary_failure_replays_history_at_same_offset(tmp_path, monkeypatch):
    from skynet_app.tracking import TrackingRequestError
    now = [1000.0]
    monkeypatch.setattr("skynet_app.tracking.time.time", lambda: now[0])
    calls = []
    def send(request, **kwargs):
        calls.append(json.loads(request.data)["files"]["wandb-history.jsonl"])
        return Response()
    monkeypatch.setattr("skynet_app.tracking.urllib.request.urlopen", send)
    client = bridge(tmp_path)
    create_run(client)
    real_upsert = client._update_summary
    def fail_summary(*args, **kwargs):
        raise TrackingRequestError("busy", status_code=429)
    monkeypatch.setattr(client, "_update_summary", fail_summary)
    client.log_metrics("run", {"loss": 0.5}, step=1, timestamp_ms=100001)
    assert client.drain_spool().remaining == 1
    assert client.binding("run")["history_offset"] == 0
    monkeypatch.setattr(client, "_update_summary", real_upsert)
    now[0] += 61
    assert client.drain_spool().remaining == 0
    assert calls[0] == calls[1]


def test_reconciliation_retries_tracking_without_active_jobs(tmp_path, monkeypatch):
    from test_pipeline import FakeCluster
    from skynet_app.database import Database
    from skynet_app.pipeline_api import PipelineService
    service = PipelineService(Database(tmp_path / "test.db"), FakeCluster())
    calls = []
    monkeypatch.setattr(service, "_flush_tracking_provider", lambda provider, **kwargs: calls.append(provider))
    assert service.reconcile()["checked"] == 0
    # Optional delivery has its own worker; it must run even without Slurm jobs.
    service.reconcile_tracking()
    assert calls == ["wandb", "mlflow"]


def test_metrics_never_resend_configuration(tmp_path):
    client = FakeWandBBridge(tmp_path, WandBSettings(api_key="fake", entity="team", auto_flush=False))
    bundle = {"id": "selection:original", "manifest_sha256": "a" * 64,
              "metadata": {"large": "x" * 4000000}}
    client.ensure_run(entity="team", project="test", local_run_id="run", run_name="test", group="test",
                      config={"batch_value": 16, "data_bundle_id": bundle})
    client.drain_spool()
    config = client.remote_runs["run"]["config"]
    assert config["data_bundle_id"]["value"] == "selection:original"
    assert config["data_bundle_manifest_sha256"]["value"] == "a" * 64
    assert client.spool_path.stat().st_size < 5000
    client.graphql_calls.clear()
    client.log_metrics("run", {"loss": 0.25}, step=8000)
    assert client.drain_spool().remaining == 0
    assert all("config" not in variables for _, variables in client.graphql_calls)
    assert client.remote_runs["run"]["config"] == config
    assert client.remote_runs["run"]["summary"]["loss"] == 0.25


@pytest.mark.parametrize("trigger", ["enqueue", "drain"])
def test_legacy_large_spool_compacts_without_losing_ids_or_metric_rows(tmp_path, trigger):
    client = FakeWandBBridge(tmp_path, WandBSettings(api_key="fake", entity="team", auto_flush=False))
    create_run(client)
    client._enqueue("log_params", {"local_run_id": "run", "params": {
        "data_bundle_id": {"id": "bundle-original", "manifest_sha256": "b" * 64,
                           "metadata": "x" * 4000000}}})
    before = [(e["id"], e["sequence"]) for e in client._events_unlocked()]
    cursor = client._load_state_unlocked()["acked_through"]
    assert client.spool_path.stat().st_size > 4000000
    if trigger == "enqueue":
        client.log_metrics("run", {"loss": 0.5}, step=8000, idempotency_key="sample-final")
        assert client._load_state_unlocked()["acked_through"] == cursor
    else:
        assert client.drain_spool().remaining == 0
    assert client.spool_path.stat().st_size < 5000
    assert [(e["id"], e["sequence"]) for e in client._events_unlocked()][:len(before)] == before
    assert client.drain_spool().remaining == 0
    assert client.remote_runs["run"]["config"]["data_bundle_id"]["value"] == "bundle-original"
    if trigger == "enqueue":
        assert client.history_rows[0]["_step"] == 8000
    assert client.drain_spool().attempted == 0


def test_unchanged_spool_is_parsed_once_across_reads_and_bridges(tmp_path, monkeypatch):
    monkeypatch.setattr("skynet_app.tracking.urllib.request.urlopen",
                        lambda *args, **kwargs: pytest.fail("Spool reads must not contact the network"))
    settings = WandBSettings(api_key="fake", entity="team", auto_flush=False)
    client = FakeWandBBridge(tmp_path, settings)
    create_run(client)
    client.log_metrics("run", {"loss": 0.5}, step=1, idempotency_key="sample-1")
    parses = []
    parse = WandBBridge._parse_events
    monkeypatch.setattr(WandBBridge, "_parse_events",
                        lambda self, payload: parses.append(len(payload)) or parse(self, payload))
    # The append already memoised the parsed queue for every reader of this capsule.
    assert client.pending_count() == 1
    assert client.metric_idempotency_keys() == {"sample-1"}
    assert client.metric_names_by_idempotency_key() == {"sample-1": {"loss"}}
    other = FakeWandBBridge(tmp_path, settings)
    other.remote_runs = client.remote_runs
    assert other.pending_count() == 1
    assert other.drain_spool().remaining == 0
    assert parses == []
    # A write outside the bridge changes the content digest: parsed once, then shared again.
    events = other._events_unlocked()
    injected = json.dumps({**events[-1], "id": "injected", "sequence": 99}).encode() + b"\n"
    other._atomic_write(other.spool_path, other.spool_path.read_bytes() + injected)
    assert client.pending_count() == 1
    assert other.pending_count() == 1
    assert [event["id"] for event in client._events_unlocked()][-1] == "injected"
    assert len(parses) == 1


def test_artifact_links_batch_preserves_all_links_and_cursor_on_failure(tmp_path, monkeypatch):
    from skynet_app.tracking import TrackingRequestError
    client = FakeWandBBridge(tmp_path, WandBSettings(api_key='fake', entity='team', auto_flush=False))
    create_run(client)
    links = [{'name':f'checkpoint-{i}','uri':f'/run/checkpoint-{i}', 'idempotency_key':f'artifact-{i}'} for i in range(57)]
    client.log_artifact_links('run', links)
    original_update = client._update_summary
    before = client._load_state_unlocked()['acked_through']
    monkeypatch.setattr(client, '_update_summary', lambda *args: (_ for _ in ()).throw(TrackingRequestError('temporary failure')))
    report=client.drain_spool()
    assert report.remaining==57 and report.delivered==0
    assert client._load_state_unlocked()['acked_through']==before
    assert 'skynet/artifact/checkpoint-0' not in client.binding('run')['summary']
    monkeypatch.setattr(client, '_update_summary', original_update)
    client.graphql_calls.clear()
    assert client.drain_spool().remaining==0
    assert len(client.graphql_calls)==1
    assert all(client.remote_runs['run']['summary'][f'skynet/artifact/checkpoint-{i}']==f'/run/checkpoint-{i}' for i in range(57))
    client.log_artifact_links('run', links)
    assert client.drain_spool().attempted==0
