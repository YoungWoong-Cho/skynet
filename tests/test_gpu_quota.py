import json
from types import SimpleNamespace

import pytest

from skynet_app.adapters import ManifestAdapter, builtin_adapter_manifests
from skynet_app.experiments import ExperimentSpec
from skynet_app.gpu_quota import account_gpu_quota, idle_partition_quota
from skynet_app.pipeline_api import PipelineService


def request():
    spec = ExperimentSpec.model_validate(dict(
        identity={"project": "qa", "experiment": "idle-quota"},
        source={"repository": "https://github.com/example/test", "revision": "a" * 40, "adapter": "generic"},
        runtime={"backend": "existing"}, native={"argv": ["python", "train.py"], "resume_argv": ["--resume"]},
        resources={"queue_policy": "auto", "gpu": {"mode": "explicit", "count": 1, "type": "any"}},
    ))
    manifest = next(item for item in builtin_adapter_manifests() if item.slug == "generic")
    return spec, ManifestAdapter(manifest).resolve(spec)


def test_gpu_quota_uses_header_names_and_legacy_column_positions():
    output = "| Account | a40 | l40s | rtx_6000 |\n| rl2-lab | 8 / 8 | 0 / 8 | 0 / 0 |"
    assert account_gpu_quota(output, "rl2-lab", ["l40s"]) == {"l40s": (0, 8)}
    legacy = "| rl2-lab | 0 / 8 | 8 / 8 | 0 / 0 |"
    assert account_gpu_quota(legacy, "rl2-lab", ["a40"], fallback_columns=["l40s", "a40", "rtx_6000"]) == {"a40": (8, 8)}


def test_auto_queue_checks_the_gpu_type_used_by_the_resolved_plan():
    spec, plan = request()
    plan.resolved_gpu_type = "a40"
    service = PipelineService.__new__(PipelineService)
    service.cluster = SimpleNamespace(run_with_fallback=lambda *a, **k: ("sky2", "| rl2-lab | 0 / 8 | 8 / 8 | 0 / 0 |"))
    resolved, _ = service._auto_queue(spec, plan, "sky2", record_snapshot=False)
    assert resolved.resources.account == "overcap", "free l40s quota cannot satisfy a resolved a40 request"


@pytest.mark.parametrize("limit,expected", [(8, "rl2-lab"), (0, "overcap")])
def test_idle_account_requires_verified_live_quota_without_persisting(limit, expected):
    spec, plan = request()
    calls = []
    def read(command, gateway, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            return "sky2", "| Account | l40s | a40 | rtx_6000 |"
        return "sky2", json.dumps(dict(account="rl2-lab", partition="rl2-lab", qos="different-qos",
                                       active_allocations=0, limits={"gpu:l40s": limit, "gpu:a40": 0, "gpu:rtx_6000": 0}))
    service = PipelineService.__new__(PipelineService)
    service.cluster = SimpleNamespace(run_with_fallback=read)
    resolved, snapshot = service._auto_queue(spec, plan, "sky2", record_snapshot=False)
    assert resolved.resources.account == expected and snapshot is None
    assert len(calls) == 2


@pytest.mark.parametrize("response,reason", [
    ({"account": "rl2-lab", "partition": "rl2-lab", "active_allocations": 1}, "idle normal account"),
    ({"account": "rl2-lab", "partition": "rl2-lab", "active_allocations": 0, "limits": {}}, "explicit GPU limit"),
])
def test_missing_account_is_not_assumed_to_have_zero_usage(response, reason):
    spec, plan = request()
    responses = iter(["No running jobs found", json.dumps(response)])
    service = PipelineService.__new__(PipelineService)
    service.cluster = SimpleNamespace(run_with_fallback=lambda *a, **k: ("sky2", next(responses)))
    with pytest.raises(ValueError, match=reason):
        service._auto_queue(spec, plan, "sky2", record_snapshot=False)


@pytest.mark.parametrize("active", [False, True])
def test_idle_probe_resolves_partition_qos_and_checks_current_allocations(monkeypatch, active):
    calls = []
    def read(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "scontrol":
            return "PartitionName=normal QoS=research-budget"
        if argv[0] == "sacctmgr":
            assert "name=research-budget" in argv
            assert "format=Name,MaxTRESPU,GrpTRES" in argv
            return "research-budget|gres/gpu:l40s=8,gres/gpu:a40=8|gres/gpu:l40s=4|"
        assert "--partition=normal" in argv
        assert not any(argument.startswith("--account=") for argument in argv)
        assert "COMPLETING" in " ".join(argv) and "SUSPENDED" in " ".join(argv)
        return "123" if active else ""
    monkeypatch.setattr("subprocess.check_output", read)
    if active:
        with pytest.raises(ValueError, match="active allocations"):
            idle_partition_quota("research", "normal")
    else:
        result = idle_partition_quota("research", "normal")
        assert result["qos"] == "research-budget"
        assert result["limits"] == {"gpu:l40s": 4, "gpu:a40": 8}
    assert len(calls) == 3
