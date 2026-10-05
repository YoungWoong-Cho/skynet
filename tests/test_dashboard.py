from __future__ import annotations

import json

import pytest

from skynet_app.cluster_config import CLUSTER, ClusterCommands
from skynet_app.main import QUERY_COMMAND, _parse_account_usage, _parse_snapshot


def test_gpu_usage_command_uses_configured_interpreter():
    commands = ClusterCommands(
        slurm_bin="/opt/slurm/bin",
        gpu_usage="/cluster tools/gpu_usage",
        gpu_usage_interpreter="/usr/bin/python3",
    )

    assert commands.gpu_usage_shell_command("-l") == (
        "/usr/bin/python3 '/cluster tools/gpu_usage' -l"
    )
    assert CLUSTER.commands.gpu_usage_shell_command("-l") in QUERY_COMMAND


def test_account_usage_parser_accepts_live_table_shape():
    output = """
|      Account      |  l40s |   a40  | rtx_6000 |    CPU    | Total GPUs |
|      rl2-lab      | 8 / 8 | 8 / 8 |  0 / 0   | 108 / 424 | 16 / 16   |
| overcap/scavenger |   0   |  126  |    1     |   1751    |   127     |
"""

    assert _parse_account_usage(output) == [
        {
            "account": "rl2-lab",
            "l40s": {"usage": 8, "limit": 8},
            "a40": {"usage": 8, "limit": 8},
            "rtx_6000": {"usage": 0, "limit": 0},
            "cpu": {"usage": 108, "limit": 424},
            "total_gpus": {"usage": 16, "limit": 16},
        },
        {
            "account": "overcap/scavenger",
            "l40s": {"usage": 0, "limit": None},
            "a40": {"usage": 126, "limit": None},
            "rtx_6000": {"usage": 1, "limit": None},
            "cpu": {"usage": 1751, "limit": None},
            "total_gpus": {"usage": 127, "limit": None},
        },
    ]


def dashboard_output(usage, receipts):
    return (
        "17|someone|RUNNING|overcap|1|grom|gres/gpu:a40:1|01:00|04:00:00|train|None|rl2-lab\n"
        "__SKYNET_JOBS__\n__SKYNET_USAGE__\n" + usage
        + "\n__SKYNET_USER_USAGE__\n| someone | 0/0/0(0) | 0/1/0(1) | 0/0/0(0) |\n"
        + "__SKYNET_IDLE_QUOTAS__\n" + json.dumps(receipts)
    )


def idle_receipt(**overrides):
    return dict({"account": "rl2-lab", "partition": "rl2-lab", "qos": "normal-budget",
                 "active_allocations": 0, "limits": {"gpu:l40s": 8, "gpu:a40": 8, "gpu:rtx_6000": 0}},
                **overrides)


def test_dashboard_adds_verified_idle_account_without_counting_overcap_usage():
    output = dashboard_output(
        "| Account | l40s | a40 | rtx_6000 | CPU | Total GPUs |\n"
        "| overcap/scavenger | 0 | 1 | 0 | 4 | 1 |",
        [idle_receipt()],
    )
    snapshot = _parse_snapshot(output, "sky2")
    own = next(row for row in snapshot["account_usage"] if row["account"] == "rl2-lab")
    assert own["l40s"] == {"usage": 0, "limit": 8, "users": []}
    assert own["a40"] == {"usage": 0, "limit": 8, "users": []}
    assert own["rtx_6000"] == {"usage": 0, "limit": 0, "users": []}
    assert own["total_gpus"]["limit"] == 16
    overflow = snapshot["account_usage"][0]
    assert overflow["a40"]["usage"] == 1
    assert overflow["a40"]["users"][0]["name"] == "someone"
    assert snapshot["summary"]["gpu_total"] == 16
    assert snapshot["quota_errors"] == []


def test_dashboard_preserves_present_account_and_never_replaces_it_with_idle_receipt():
    snapshot = _parse_snapshot(dashboard_output(
        "| Account | l40s | a40 | rtx_6000 | CPU | Total GPUs |\n"
        "| rl2-lab | 2 / 8 | 3 / 8 | 0 / 0 | 12 / 424 | 5 / 16 |", [idle_receipt()],
    ), "sky2")
    assert len(snapshot["account_usage"]) == 1
    assert snapshot["account_usage"][0]["l40s"]["usage"] == 2
    assert snapshot["account_usage"][0]["a40"]["usage"] == 3
    assert snapshot["quota_errors"] == []


@pytest.mark.parametrize("receipts", [
    [idle_receipt(error="Slurm quota lookup timed out")],
    [idle_receipt(active_allocations=1)],
    [idle_receipt(limits={})],
    [idle_receipt(limits={"gpu:a40": -1})],
    [idle_receipt(account="another-lab")],
    [],
    {"error": "invalid response"},
])
def test_dashboard_idle_probe_failure_preserves_jobs_and_other_accounts(receipts):
    snapshot = _parse_snapshot(dashboard_output(
        "| Account | l40s | a40 | rtx_6000 | CPU | Total GPUs |\n"
        "| another-lab | 1 / 2 | 0 / 8 | 0 / 0 | 4 / 32 | 1 / 10 |", receipts,
    ), "sky2")
    assert [row["account"] for row in snapshot["account_usage"]] == ["another-lab"]
    assert snapshot["jobs"][0]["id"] == "17"
    assert snapshot["quota_errors"]


def test_dashboard_malformed_idle_probe_response_is_nonfatal():
    output = dashboard_output("", []).replace("__SKYNET_IDLE_QUOTAS__\n[]", "__SKYNET_IDLE_QUOTAS__\nnot JSON")
    snapshot = _parse_snapshot(output, "sky2")
    assert snapshot["jobs"][0]["id"] == "17"
    assert snapshot["account_usage"] == []
    assert snapshot["quota_errors"]
