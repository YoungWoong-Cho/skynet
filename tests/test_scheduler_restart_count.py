from __future__ import annotations

import pytest

from skynet_app.cluster_runtime import ClusterClient


class RestartAccountingClient(ClusterClient):
    def __init__(self, output: str) -> None:
        super().__init__(("sky2",))
        self.output = output
        self.accounting_calls: list[tuple[str, str, int]] = []
        self.queue_calls: list[tuple[str, str]] = []

    def run_with_fallback(self, command, gateway="auto", *, timeout=30, **kwargs):
        self.accounting_calls.append((command, gateway, timeout))
        return "sky2", self.output

    def ssh(self, host, command, **kwargs):
        self.queue_calls.append((host, command))
        return "102|PENDING|Resources\n"


def accounting_row(job: str, state: str, restarts: str | None) -> str:
    row = f"{job}|{state}|0:0|None|node1|00:00:10|1789060000|Unknown|gres/gpu=1|rl2-lab|rl2-lab"
    return row + ("" if restarts is None else "|" + restarts) + "\n"


def test_restart_counts_share_one_batched_accounting_query() -> None:
    client = RestartAccountingClient(
        accounting_row("101", "RUNNING", "2")
        + accounting_row("102", "PENDING", "0")
        + accounting_row("103_7", "COMPLETED", "3")
    )

    host, rows = client.job_statuses(["101", "102", "103_7", "101"])

    assert host == "sky2"
    assert {job: row["Restarts"] for job, row in rows.items()} == {
        "101": "2", "102": "0", "103_7": "3",
    }
    assert len(client.accounting_calls) == 1
    command, _, _ = client.accounting_calls[0]
    assert "-j 101,102,103_7" in command
    assert "Partition,Account,Restarts" in command
    assert len(client.queue_calls) == 1
    assert rows["102"]["Reason"] == "Resources"
    assert all("scontrol" not in call[0] for call in client.accounting_calls)


@pytest.mark.parametrize("value", [None, "", "Unknown", "N/A", "-1", "1.5", "0 bad"])
def test_unavailable_or_invalid_restart_count_does_not_become_zero(value) -> None:
    client = RestartAccountingClient(accounting_row("101", "RUNNING", value))

    _, rows = client.job_statuses(["101"])

    assert rows["101"]["State"] == "RUNNING"
    assert "Restarts" not in rows["101"]
    assert client.queue_calls == []


@pytest.mark.parametrize("value, expected", [("0", "0"), (" 02 ", "2")])
def test_explicit_restart_zero_and_count_are_preserved(value, expected) -> None:
    client = RestartAccountingClient(accounting_row("101", "RUNNING", value))

    _, rows = client.job_statuses(["101"])

    assert rows["101"]["Restarts"] == expected


def test_partial_accounting_row_is_not_treated_as_restart_evidence() -> None:
    client = RestartAccountingClient("101|RUNNING|0:0|2\n")

    _, rows = client.job_statuses(["101"])

    assert rows == {}
