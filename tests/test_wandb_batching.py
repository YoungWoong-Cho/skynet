"""Metric uploads retain every sample and back off across bridge instances."""
import io
import json
import urllib.error
from email.message import Message

from test_tracking import FakeWandBBridge
from skynet_app.tracking import WandBBridge, WandBSettings


class WireBridge(FakeWandBBridge):
    _append_history_rows = WandBBridge._append_history_rows


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
    real_upsert = client._upsert_run
    def fail_summary(**kwargs):
        raise TrackingRequestError("busy", status_code=429)
    monkeypatch.setattr(client, "_upsert_run", fail_summary)
    client.log_metrics("run", {"loss": 0.5}, step=1, timestamp_ms=100001)
    assert client.drain_spool().remaining == 1
    assert client.binding("run")["history_offset"] == 0
    monkeypatch.setattr(client, "_upsert_run", real_upsert)
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
    assert calls == ["wandb"]
