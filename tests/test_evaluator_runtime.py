from __future__ import annotations

from types import SimpleNamespace

from skynet_app.adapters import builtin_adapter_manifests
from skynet_app.cluster_config import CLUSTER
from skynet_app.database import content_sha256
from skynet_app.pipeline_api import PipelineService


EVAL_TASKS_COMMIT = "460f2878bdcb4db2d21913db789174fb316b73e2"
GROOT_RUNTIME_PROFILE = "groot-isaacsim-5.0.0_isaaclab-2.2.0_py311"


def test_groot_isaaclab_declaration_owns_exact_runtime_and_suite() -> None:
    manifest = next(item for item in builtin_adapter_manifests() if item.slug == "groot")
    declaration = next(
        item
        for item in manifest.evaluations
        if item.environment == "isaac_sim"
        and item.suites == ["groot_gr1_isaaclab_evaltasks"]
    )

    assert declaration.runtime_profile_id == GROOT_RUNTIME_PROFILE
    assert declaration.enabled_when == {"native.config.embodiment_tag": ["GR1"]}
    assert declaration.command is not None
    assert (
        declaration.command.capsule_files["adapter-support/groot-isaaclab.py"]
    )


def test_dual_evaluation_runtime_preserves_policy_pin_and_snapshots_evaluator(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")
    policy_runtime = {
        "backend": "uv",
        "profile": "repository",
        "lock_file": "uv.lock",
        "lock_sha256": "a" * 64,
        "resolution": {"commit": "9b37aa1ce69c73c6d165233fa88128283bba4508"},
    }
    suite_config = {
        "task_catalog_provenance": {
            "repository": "https://github.com/isaac-sim/IsaacLabEvalTasks",
            "revision": EVAL_TASKS_COMMIT,
            "source_path": "scripts/config/args.py",
        }
    }

    runtimes, blockers = PipelineService._resolve_dual_evaluation_runtimes(
        policy_runtime,
        GROOT_RUNTIME_PROFILE,
        suite_config,
    )

    assert blockers == []
    assert runtimes["policy_runtime"] == policy_runtime
    evaluator = runtimes["evaluator_runtime"]
    assert evaluator["profile"] == GROOT_RUNTIME_PROFILE
    assert evaluator["backend"] == "conda"
    assert evaluator["python_executable"] == (
        "/coc/flash7/ycho420/envs/"
        "groot-isaacsim-5.0.0_isaaclab-2.2.0_py311/bin/python"
    )
    assert "OMNI_KIT_ACCEPT_EULA" not in evaluator["environment"]
    assert evaluator["operator_environment"] == {
        "OMNI_KIT_ACCEPT_EULA": "YES"
    }
    assert evaluator["source"]["revision"] == EVAL_TASKS_COMMIT
    assert evaluator["source_dir"].endswith(EVAL_TASKS_COMMIT)
    assert evaluator["versions"] == {
        "python": "3.11",
        "isaac_sim": "5.0.0",
        "isaac_lab": "2.2.0",
        "isaac_lab_eval_tasks": EVAL_TASKS_COMMIT,
    }
    assert len(evaluator["profile_snapshot_sha256"]) == 64
    verification = evaluator["profile_snapshot"]["verification"]
    assert verification["distributions"] == {"isaacsim": "5.0.0"}
    assert set(verification["required_distributions"]) == {
        "imageio",
        "imageio-ffmpeg",
    }
    assert "isaaclab_assets" in verification["python_imports"]
    assert "pinocchio" in verification["python_imports"]
    assert verification["python_imports"].index("pinocchio") < verification[
        "python_imports"
    ].index("isaaclab")
    assert verification["executables"] == ["nvidia-smi"]
    assert verification["executable_providers"] == [
        {
            "name": "imageio-ffmpeg encoder",
            "module": "imageio_ffmpeg",
            "resolver": "get_ffmpeg_exe",
        }
    ]
    assert verification["shared_libraries"] == ["libSM.so.6", "libXext.so.6"]
    assert verification["gym_registrations"] == [
        {
            "module": "isaaclab_eval_tasks.tasks",
            "ids": [
                "Isaac-NutPour-GR1T2-ClosedLoop-v0",
                "Isaac-ExhaustPipe-GR1T2-ClosedLoop-v0",
            ],
        }
    ]
    assert verification["platform_system"] == "Linux"
    assert verification["glibc_minimum"] == "2.35"
    assert verification["requires_gpu"] is True
    assert verification["compute_attestation_path"].endswith(
        "/.skynet/compute-readiness-v1.json"
    )
    assert verification["pip_check"] is True
    assert verification["compute_smoke"] == {
        "suite": "groot_gr1_isaaclab_evaltasks",
        "adapter": "groot",
        "resources": {
            "queue_policy": "normal",
            "gpu_type": "l40s",
            "gpu_count": 1,
            "cpus_per_task": 12,
            "memory_gb": 64,
            "time_limit": "02:00:00",
        },
        "capsule_asset": "adapter-support/groot-isaaclab.py",
        "capsule_sha256": "13ff47cc8531c5e8834b6752412a984d6c015c2335a989e36fdf16907f225d54",
        "argv": [
            "{{runtime.python_executable}}",
            "{{smoke.asset_path}}",
            "--source-dir",
            "{{runtime.source_dir}}",
            "--readiness-smoke-output",
            "{{smoke.result_path}}",
            "--simulation-device",
            "cuda:0",
        ],
        "environment": {
            "PYTHONPATH": "{{runtime.source_dir}}/source/isaaclab_eval_tasks"
        },
        "result_schema": "skynet.evaluator-readiness/v1",
        "timeout_seconds": 1800,
        "required_checks": [
            "environment_constructed",
            "environment_reset",
            "media_writer",
            "media_nonempty",
        ],
    }


def test_dual_evaluation_runtime_blocks_missing_profile_without_changing_policy() -> None:
    policy_runtime = {"backend": "uv", "lock_file": "uv.lock"}

    runtimes, blockers = PipelineService._resolve_dual_evaluation_runtimes(
        policy_runtime,
        "missing-evaluator-profile",
        {
            "task_catalog_provenance": {
                "repository": "https://example.invalid/evaluator",
                "revision": "b" * 40,
            }
        },
    )

    assert runtimes == {"policy_runtime": policy_runtime}
    assert blockers
    assert "is not configured" in blockers[0]


def test_dual_evaluation_runtime_requires_external_operator_eula_acceptance(
    monkeypatch,
) -> None:
    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)

    runtimes, blockers = PipelineService._resolve_dual_evaluation_runtimes(
        {"backend": "uv", "lock_file": "uv.lock"},
        GROOT_RUNTIME_PROFILE,
        {
            "task_catalog_provenance": {
                "repository": "https://github.com/isaac-sim/IsaacLabEvalTasks",
                "revision": EVAL_TASKS_COMMIT,
            }
        },
    )

    assert runtimes["evaluator_runtime"]["operator_environment"] == {}
    assert any("NVIDIA Isaac Sim EULA" in blocker for blocker in blockers)
    assert any("OMNI_KIT_ACCEPT_EULA=YES" in blocker for blocker in blockers)


def test_operational_readiness_probe_returns_actionable_blocker(monkeypatch) -> None:
    service = PipelineService.__new__(PipelineService)
    monkeypatch.setattr(
        service,
        "_probe_runtime_profile",
        lambda profile, gateway, operator_environment=None, suite_config=None: {
            "runtime_verified": False,
            "verification": {
                "checked": True,
                "gateway": gateway,
                "errors": [
                    "distribution isaacsim is not installed",
                    "required source prerequisite is missing: /shared/IsaacLabEvalTasks",
                ],
            },
        },
    )

    readiness, blockers = service._verify_evaluator_runtime_readiness(
        GROOT_RUNTIME_PROFILE,
        "sky2",
        operator_environment={"OMNI_KIT_ACCEPT_EULA": "YES"},
    )

    assert readiness["checked"] is True
    assert readiness["gateway"] == "sky2"
    assert blockers == [
        "evaluation runtime profile "
        f"{GROOT_RUNTIME_PROFILE} is not ready: "
        "distribution isaacsim is not installed; required source prerequisite is missing: "
        "/shared/IsaacLabEvalTasks"
    ]


def test_validation_requires_runtime_readiness_and_surfaces_blocker(monkeypatch) -> None:
    service = PipelineService.__new__(PipelineService)
    run = {"id": "run-1"}
    checkpoint = {"id": "checkpoint-1", "path": "/checkpoint"}
    suite = {
        "id": "suite-1",
        "name": "groot_gr1_isaaclab_evaltasks",
        "config_json": {},
    }
    validation = {
        "valid": True,
        "run_valid": True,
        "checkpoint_valid": True,
        "errors": {},
    }
    monkeypatch.setattr(
        service,
        "_resolve_evaluation_target",
        lambda run_id, checkpoint_path: (validation.copy(), run, checkpoint),
    )
    monkeypatch.setattr(
        service,
        "_resolve_evaluation_suite_selection",
        lambda suite_id, environment, tasks: (suite, "isaac_sim", ["nutpouring"], {}),
    )
    observed = {}

    def resolve(*args, **kwargs):
        observed.update(kwargs)
        return (
            None,
            SimpleNamespace(
                argv=["python", "adapter-support/groot-isaaclab.py"],
                blockers=[
                    "evaluation runtime profile groot-runtime is not ready: "
                    "compute readiness attestation is stale"
                ],
            ),
            {},
            {},
            {
                "id": "adapter-1",
                "version_id": "version-1",
                "slug": "groot",
                "version": 1,
                "manifest_sha256": "a" * 64,
            },
            "registered_adapter",
        )

    monkeypatch.setattr(service, "_resolve_evaluation_implementation", resolve)

    result = service.validate_evaluation_target(
        "run-1", "/checkpoint", suite_id="suite-1", gateway="sky2"
    )

    assert observed["verify_evaluator_runtime"] is True
    assert observed["evaluator_runtime_gateway"] == "sky2"
    assert result["plan_valid"] is False
    assert result["valid"] is False
    assert "attestation is stale" in result["plan_message"]


def test_runtime_readiness_validation_cache_avoids_repeated_ssh(monkeypatch) -> None:
    service = PipelineService.__new__(PipelineService)
    calls = []
    monkeypatch.setattr(
        service,
        "_probe_runtime_profile",
        lambda profile, gateway, operator_environment=None, suite_config=None: (
            calls.append((profile["id"], gateway))
            or {
                "runtime_verified": False,
                "verification": {
                    "checked": True,
                    "gateway": gateway,
                    "errors": ["compute readiness attestation is missing"],
                },
            }
        ),
    )

    first = service._verify_evaluator_runtime_readiness(
        GROOT_RUNTIME_PROFILE, "sky2", suite_config={"suite": "test"}
    )
    second = service._verify_evaluator_runtime_readiness(
        GROOT_RUNTIME_PROFILE, "sky2", suite_config={"suite": "test"}
    )
    refreshed = service._verify_evaluator_runtime_readiness(
        GROOT_RUNTIME_PROFILE,
        "sky2",
        suite_config={"suite": "test"},
        refresh=True,
    )

    assert first == second == refreshed
    assert calls == [
        (GROOT_RUNTIME_PROFILE, "sky2"),
        (GROOT_RUNTIME_PROFILE, "sky2"),
    ]


def test_runtime_probe_carries_full_contract_and_explicit_operator_environment() -> None:
    class ProbeCluster:
        command = ""
        timeout = 0

        def run_with_fallback(self, command, gateway, *, timeout):
            self.command = command
            self.timeout = timeout
            profile_snapshot = CLUSTER.runtime_profile_snapshot(GROOT_RUNTIME_PROFILE)
            verification = profile_snapshot["verification"]
            attestation = {
                "schema_version": "skynet.runtime-readiness/v1",
                "profile_id": GROOT_RUNTIME_PROFILE,
                "profile_snapshot_sha256": content_sha256(profile_snapshot),
                "ready": True,
                "errors": [],
                "execution": {
                    "slurm_job_id": "123456",
                    "node": "compute-l40s",
                    "gpu_type": "l40s",
                },
                "actual": {
                    "python": "3.11.11",
                    "distributions": {
                        **verification["distributions"],
                        **{
                            name: "1.0"
                            for name in verification["required_distributions"]
                        },
                    },
                    "imports": {
                        name: True for name in verification["python_imports"]
                    },
                    "executables": {
                        name: f"/usr/bin/{name}"
                        for name in verification["executables"]
                    },
                    "executable_providers": {
                        item["name"]: {"path": "/tmp/imageio-ffmpeg"}
                        for item in verification["executable_providers"]
                    },
                    "shared_libraries": {
                        name: True for name in verification["shared_libraries"]
                    },
                    "gym_registrations": {
                        item["module"]: {
                            "module_imported": True,
                            "ids": {task_id: True for task_id in item["ids"]},
                        }
                        for item in verification["gym_registrations"]
                    },
                    "platform": {
                        "system": "Linux",
                        "libc": "glibc",
                        "libc_version": "2.39",
                    },
                    "gpu": {"cuda_available": True, "device_count": 1},
                    "checks": {
                        name: True
                        for name in verification["compute_smoke"]["required_checks"]
                    },
                    "pip_check": {"ok": True, "output": "No broken requirements found."},
                },
            }
            source_lines = []
            for index, prerequisite in enumerate(profile_snapshot["source_prerequisites"]):
                source_lines.extend(
                    [
                        f"SKYNET_RUNTIME_SOURCE_{index}_PRESENT=1",
                        f"SKYNET_RUNTIME_SOURCE_{index}_HEAD={prerequisite['revision']}",
                        f"SKYNET_RUNTIME_SOURCE_{index}_EXPECTED={prerequisite['revision']}",
                    ]
                )
            return gateway, "\n".join(
                [
                    "SKYNET_RUNTIME_ENVIRONMENT_PRESENT=1",
                    "SKYNET_RUNTIME_PYTHON_PRESENT=1",
                    "SKYNET_RUNTIME_ATTESTATION_BEGIN",
                    __import__("json").dumps(attestation),
                    "SKYNET_RUNTIME_ATTESTATION_END",
                    *source_lines,
                ]
            )

    service = PipelineService.__new__(PipelineService)
    service.cluster = ProbeCluster()
    profile = next(
        item
        for item in CLUSTER.public_runtime_profiles()
        if item["id"] == GROOT_RUNTIME_PROFILE
    )

    result = service._probe_runtime_profile(
        profile,
        "sky2",
        operator_environment={"OMNI_KIT_ACCEPT_EULA": "YES"},
    )

    assert result["runtime_verified"] is True
    assert service.cluster.timeout == 30
    assert "OMNI_KIT_ACCEPT_EULA" not in service.cluster.command
    assert "bin/python - <<" not in service.cluster.command
    assert "compute-readiness-v1.json" in service.cluster.command
