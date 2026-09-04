from __future__ import annotations

from skynet_app.cluster_config import ClusterCommands
from skynet_app.main import QUERY_COMMAND, _parse_account_usage


def test_gpu_usage_command_uses_configured_interpreter():
    commands = ClusterCommands(
        slurm_bin="/opt/slurm/bin",
        gpu_usage="/cluster tools/gpu_usage",
        gpu_usage_interpreter="/usr/bin/python3",
    )

    assert commands.gpu_usage_shell_command("-l") == (
        "/usr/bin/python3 '/cluster tools/gpu_usage' -l"
    )
    assert "/usr/bin/python3 /coc/testnvme/admin/tools/skynet-utilities/gpu_usage -l" in QUERY_COMMAND


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
