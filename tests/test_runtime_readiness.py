from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from skynet_app.cluster_runtime import Submission
from skynet_app.pipeline_api import PipelineService
from skynet_app.runtime_readiness import (
    _content_sha256,
    _run_streaming_subprocess,
    build_readiness_contract,
    execute_compute_probe,
    main,
    render_readiness_sbatch,
    submit_readiness_sbatch,
)


PROFILE = "groot-isaacsim-5.0.0_isaaclab-2.2.0_py311"
SUITE = "groot_gr1_isaaclab_evaltasks"


def test_groot_readiness_contract_binds_profile_suite_and_adapter_hook() -> None:
    contract, source = build_readiness_contract(PROFILE, SUITE)

    assert contract["profile_id"] == PROFILE
    assert len(contract["profile_snapshot_sha256"]) == 64
    assert contract["suite"]["suite"] == SUITE
    assert contract["suite"]["version"] == "catalog-v2-git-460f2878bdcb"
    assert contract["suite"]["task_environments"] == [
        {
            "id": "nutpouring",
            "environment_id": "Isaac-NutPour-GR1T2-ClosedLoop-v0",
            "simulation_device": "cuda:0",
        },
        {
            "id": "pipesorting",
            "environment_id": "Isaac-ExhaustPipe-GR1T2-ClosedLoop-v0",
            "simulation_device": "cuda:0",
        },
    ]
    assert contract["hook"]["result_schema"] == "skynet.evaluator-readiness/v1"
    assert contract["verification"]["compute_smoke"]["resources"] == {
        "queue_policy": "normal",
        "gpu_type": "l40s",
        "gpu_count": 1,
        "cpus_per_task": 12,
        "memory_gb": 64,
        "time_limit": "02:00:00",
    }
    assert hashlib.sha256(source.encode("utf-8")).hexdigest() == contract["hook"][
        "capsule_sha256"
    ]


def test_rendered_readiness_job_requests_one_l40s_without_accepting_eula() -> None:
    script = render_readiness_sbatch(PROFILE, SUITE)

    assert "#SBATCH --partition=rl2-lab" in script
    assert "#SBATCH --account=rl2-lab" in script
    assert "#SBATCH --gres=gpu:l40s:1" in script
    assert "#SBATCH --cpus-per-task=12" in script
    assert "#SBATCH --mem=64G" in script
    assert "#SBATCH --time=02:00:00" in script
    assert "#SBATCH --export=ALL" in script
    assert 'export TMPDIR="$probe_root/tmp"' in script
    assert "#SBATCH --nodelist=grom" in script
    assert re.search(
        r"^export OMNI_KIT_ACCEPT_EULA=YES$", script, flags=re.MULTILINE
    ) is None
    assert '${OMNI_KIT_ACCEPT_EULA:-}' in script
    assert "/.skynet/compute-readiness-v1.json" in script


def test_readiness_rejects_resources_below_profile_minimum() -> None:
    with pytest.raises(ValueError, match="requires at least 12 CPUs"):
        render_readiness_sbatch(PROFILE, SUITE, cpus_per_task=4)
    with pytest.raises(ValueError, match="requires at least 64 GB"):
        render_readiness_sbatch(PROFILE, SUITE, memory_gb=32)
    with pytest.raises(ValueError, match="time limit of at least 02:00:00"):
        render_readiness_sbatch(PROFILE, SUITE, time_limit="00:30:00")


def test_hook_streams_output_before_exit_and_retains_attestation_tails() -> None:
    class LiveStream:
        def __init__(self, marker: str) -> None:
            self.marker = marker
            self.value = ""
            self.seen = threading.Event()

        def write(self, value: str) -> None:
            self.value += value
            if self.marker in self.value:
                self.seen.set()

        def flush(self) -> None:
            return None

    stdout = LiveStream("stdout-live")
    stderr = LiveStream("stderr-live")
    command = [
        sys.executable,
        "-u",
        "-c",
        (
            "import sys,time; "
            "print('stdout-live', flush=True); "
            "print('stderr-live', file=sys.stderr, flush=True); "
            "time.sleep(0.4); "
            "print('finished', flush=True)"
        ),
    ]

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            _run_streaming_subprocess,
            command,
            environment=dict(os.environ),
            timeout_seconds=5,
            stdout_stream=stdout,
            stderr_stream=stderr,
        )
        assert stdout.seen.wait(timeout=1)
        assert stderr.seen.wait(timeout=1)
        assert not future.done()
        result = future.result(timeout=2)

    assert result == {
        "returncode": 0,
        "stdout": "stdout-live\nfinished\n",
        "stderr": "stderr-live\n",
        "timed_out": False,
    }


def test_readiness_cli_renders_validated_manual_node_placement(tmp_path: Path) -> None:
    output = tmp_path / "readiness.sbatch"

    assert main(
        [
            "render-sbatch",
            "--profile",
            PROFILE,
            "--suite",
            SUITE,
            "--node",
            "grom",
            "--output",
            str(output),
        ]
    ) == 0

    assert "#SBATCH --nodelist=grom" in output.read_text(encoding="utf-8")


def test_readiness_submission_uses_gateway_and_idempotent_environment_handoff(
    monkeypatch,
) -> None:
    class RecordingReadinessCluster:
        def __init__(self) -> None:
            self.submission = None

        def test_script(self, script, gateway="auto"):
            return "sky2", "valid"

        def submit_script(self, script, run_id, gateway="auto", **options):
            self.submission = (script, run_id, gateway, options)
            return Submission("12345", "12345", gateway, "/job.sbatch", "/run")

    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")
    cluster = RecordingReadinessCluster()

    submission = submit_readiness_sbatch(
        PROFILE,
        SUITE,
        run_id="readiness-123",
        gateway="sky2",
        node="grom",
        cluster=cluster,
    )

    assert submission.job_id == "12345"
    assert cluster.submission is not None
    script, run_id, gateway, options = cluster.submission
    assert "#SBATCH --nodelist=grom" in script
    assert run_id == "readiness-123"
    assert gateway == "sky2"
    assert options == {
        "submission_key": "readiness-123",
        "forwarded_environment": {"OMNI_KIT_ACCEPT_EULA": "YES"},
    }


def test_hook_runs_and_records_both_streams_despite_preflight_failure(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    hook = tmp_path / "hook.py"
    hook.write_text(
        """import argparse, json, sys
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
a = p.parse_args()
print('hook stdout evidence')
print('hook stderr evidence', file=sys.stderr)
Path(a.out).write_text(json.dumps({
    'schema_version': 'skynet.evaluator-readiness/v1',
    'ready': True,
    'errors': [],
    'actual': {},
}), encoding='utf-8')
""",
        encoding="utf-8",
    )
    profile_snapshot = {
        "schema_version": 1,
        "id": "test-runtime",
        "backend": "conda",
        "environment_path": sys.prefix,
        "environment": {},
        "environment_operations": [],
        "source_prerequisites": [],
    }
    contract = {
        "schema_version": "skynet.runtime-readiness-contract/v1",
        "profile_id": "test-runtime",
        "profile_snapshot": profile_snapshot,
        "profile_snapshot_sha256": _content_sha256(profile_snapshot),
        "suite_contract_sha256": "c" * 64,
        "verification": {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "distributions": {},
            "required_distributions": [],
            "python_imports": [],
            "executables": ["definitely-not-installed-skynet-test"],
            "executable_providers": [],
            "shared_libraries": [],
            "gym_registrations": [],
            "platform_system": platform.system(),
            "glibc_minimum": None,
            "requires_gpu": False,
            "pip_check": False,
            "compute_smoke": {"required_checks": []},
        },
        "runtime": {
            "environment_path": sys.prefix,
            "python_executable": sys.executable,
            "source_dir": str(tmp_path),
        },
        "hook": {
            "capsule_sha256": hashlib.sha256(hook.read_bytes()).hexdigest(),
            "argv": [
                "{{runtime.python_executable}}",
                "{{smoke.asset_path}}",
                "--out",
                "{{smoke.result_path}}",
            ],
            "environment": {},
            "result_schema": "skynet.evaluator-readiness/v1",
            "timeout_seconds": 30,
        },
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    attestation_path = tmp_path / "attestation.json"
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")

    status = execute_compute_probe(contract_path, hook, attestation_path)

    assert status == 1
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    assert attestation["ready"] is False
    assert attestation["actual"]["hook"] == {
        "attempted": True,
        "returncode": 0,
        "stdout": "hook stdout evidence\n",
        "stderr": "hook stderr evidence\n",
        "timed_out": False,
    }
    assert not any(
        "did not write its declared result" in error
        for error in attestation["errors"]
    )
    assert (
        "required executable definitely-not-installed-skynet-test was not found"
        in capsys.readouterr().err
    )


def test_adapter_report_is_wrapped_as_atomic_core_attestation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    hook = tmp_path / "hook.py"
    hook.write_text(
        """import argparse, json, os, tempfile
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
a = p.parse_args()
value = {
    'schema_version': 'skynet.evaluator-readiness/v1',
    'ready': True,
    'errors': [],
    'actual': {
        'imports': {'json': True},
        'gym_registrations': {
            'fake.tasks': {
                'module_imported': True,
                'ids': {'Fake-Task-v0': True},
            }
        },
        'gpu': {'cuda_available': True, 'device_count': 1, 'devices': ['L40S']},
        'checks': {'environment_reset': True, 'media_nonempty': True},
    },
}
path = Path(a.out)
path.write_text(json.dumps(value), encoding='utf-8')
""",
        encoding="utf-8",
    )
    profile_snapshot = {
        "schema_version": 1,
        "id": "test-runtime",
        "backend": "conda",
        "environment_path": sys.prefix,
        "environment": {},
        "environment_operations": [],
        "source_prerequisites": [],
    }
    verification = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "distributions": {},
        "required_distributions": [],
        "python_imports": ["json"],
        "executables": [],
        "executable_providers": [],
        "shared_libraries": [],
        "gym_registrations": [
            {"module": "fake.tasks", "ids": ["Fake-Task-v0"]}
        ],
        "platform_system": platform.system(),
        "glibc_minimum": None,
        "requires_gpu": True,
        "pip_check": False,
        "compute_smoke": {
            "required_checks": ["environment_reset", "media_nonempty"]
        },
    }
    suite_sha = "b" * 64
    contract = {
        "schema_version": "skynet.runtime-readiness-contract/v1",
        "profile_id": "test-runtime",
        "profile_snapshot": profile_snapshot,
        "profile_snapshot_sha256": _content_sha256(profile_snapshot),
        "suite_contract_sha256": suite_sha,
        "verification": verification,
        "runtime": {
            "environment_path": sys.prefix,
            "python_executable": sys.executable,
            "source_dir": str(tmp_path),
        },
        "hook": {
            "capsule_sha256": hashlib.sha256(hook.read_bytes()).hexdigest(),
            "argv": [
                "{{runtime.python_executable}}",
                "{{smoke.asset_path}}",
                "--out",
                "{{smoke.result_path}}",
            ],
            "environment": {},
            "result_schema": "skynet.evaluator-readiness/v1",
            "timeout_seconds": 30,
        },
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    attestation_path = tmp_path / "nested" / "attestation.json"
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    monkeypatch.setenv("SLURMD_NODENAME", "compute-l40s")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "rl2-lab")
    monkeypatch.setenv("SKYNET_RUNTIME_READINESS_GPU_TYPE", "l40s")
    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")

    status = execute_compute_probe(contract_path, hook, attestation_path)

    assert status == 0
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    assert attestation["schema_version"] == "skynet.runtime-readiness/v1"
    assert attestation["ready"] is True
    assert attestation["profile_snapshot_sha256"] == _content_sha256(
        profile_snapshot
    )
    assert attestation["suite_contract_sha256"] == suite_sha
    assert attestation["execution"]["slurm_job_id"] == "123456"
    expected = {
        "profile_id": "test-runtime",
        "profile_snapshot_sha256": _content_sha256(profile_snapshot),
        "suite_contract_sha256": suite_sha,
        "python_version": verification["python_version"],
        "distributions": {},
        "required_distributions": [],
        "python_imports": ["json"],
        "executables": [],
        "executable_providers": [],
        "shared_libraries": [],
        "gym_registrations": verification["gym_registrations"],
        "platform_system": platform.system(),
        "glibc_minimum": None,
        "requires_gpu": True,
        "required_checks": ["environment_reset", "media_nonempty"],
        "pip_check": False,
    }
    assert PipelineService._validate_compute_runtime_attestation(
        attestation, expected
    ) == []
