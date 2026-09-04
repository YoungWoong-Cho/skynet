from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from skynet_app.adapters.groot_isaaclab_bridge import (
    GROOT_ISAACLAB_BRIDGE_SOURCE,
    ISAAC_KIT_ARGS,
    ISAAC_CLIENT_SOURCE,
    OMNI_WARP_BINARY_SHA256,
    OMNI_WARP_VERSION,
    TASK_CATALOG_HASH,
    TASK_DEFINITIONS,
    _evaluator_readiness_result,
    _launch_isaac_app,
    append_jsonl,
    build_isaac_environment,
    build_policy_server_command,
    load_completed,
    run_evaluator_readiness_smoke,
    validate_checkpoint_contract,
)


def _processor_config() -> dict:
    keys = ["left_arm", "right_arm", "left_hand", "right_hand"]
    return {
        "processor_class": "Gr00tN1d6Processor",
        "processor_kwargs": {
            "modality_configs": {
                "gr1": {
                    "video": {"modality_keys": ["ego_view"], "delta_indices": [0]},
                    "state": {
                        "modality_keys": keys,
                        "delta_indices": [0],
                        "sin_cos_embedding_keys": keys,
                    },
                    "action": {
                        "modality_keys": keys,
                        "delta_indices": list(range(16)),
                        "action_configs": [
                            {
                                "rep": "ABSOLUTE",
                                "type": "NON_EEF",
                                "format": "DEFAULT",
                                "state_key": None,
                            }
                            for _ in keys
                        ],
                    },
                    "language": {
                        "modality_keys": ["annotation.human.action.task_description"],
                        "delta_indices": [0],
                    },
                }
            }
        },
    }


def _checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "processor_config.json").write_text(
        json.dumps(_processor_config()), encoding="utf-8"
    )
    (checkpoint / "embodiment_id.json").write_text(
        json.dumps({"gr1": 20}), encoding="utf-8"
    )
    return checkpoint


def test_checkpoint_contract_accepts_exact_n16_gr1_profile(tmp_path: Path) -> None:
    validate_checkpoint_contract(_checkpoint(tmp_path))


def test_checkpoint_contract_rejects_changed_action_horizon(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    processor = _processor_config()
    processor["processor_kwargs"]["modality_configs"]["gr1"]["action"]["delta_indices"] = list(
        range(8)
    )
    (checkpoint / "processor_config.json").write_text(json.dumps(processor), encoding="utf-8")

    with pytest.raises(RuntimeError, match="action contract is incompatible"):
        validate_checkpoint_contract(checkpoint)


def test_suite_catalog_is_exact_and_client_capsule_compiles() -> None:
    suite_path = Path("config/evaluation_suites/groot-isaaclab-evaltasks.json")
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    canonical_tasks = json.dumps(suite["tasks"], sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical_tasks.encode("utf-8")).hexdigest() == TASK_CATALOG_HASH
    assert suite["tasks"] == list(TASK_DEFINITIONS)
    assert [item["metadata"]["native"] for item in suite["task_options"]] == list(
        TASK_DEFINITIONS.values()
    )
    assert {item["metadata"]["native"]["simulation_device"] for item in suite["task_options"]} == {
        "cuda:0"
    }
    compile(ISAAC_CLIENT_SOURCE, "skynet-isaac-client.py", "exec")
    assert 'parser.add_argument("--simulation-device", default="cuda:0")' in ISAAC_CLIENT_SOURCE
    assert (
        'KIT_ARGS = "--/rtx/verifyDriverVersion/enabled=false"'
        in ISAAC_CLIENT_SOURCE
    )
    assert "kit_args=kit_args(args.kit_extension_folder)" in ISAAC_CLIENT_SOURCE
    assert "launcher, simulation_app = _launch_isaac_app(" in ISAAC_CLIENT_SOURCE
    assert '"schema_version": "skynet.isaac-launch-phase/v1"' in ISAAC_CLIENT_SOURCE
    assert ISAAC_CLIENT_SOURCE.index("import isaacsim") < ISAAC_CLIENT_SOURCE.index("import pinocchio")
    assert ISAAC_CLIENT_SOURCE.index("import pinocchio") < ISAAC_CLIENT_SOURCE.index(
        "from isaaclab.app import AppLauncher"
    )
    assert suite["version"] == "catalog-v2-git-460f2878bdcb"
    assert "GROOT_ISAACLAB_BRIDGE_SOURCE" in GROOT_ISAACLAB_BRIDGE_SOURCE


def test_progress_resume_requires_same_identity_and_video(tmp_path: Path) -> None:
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"not-empty")
    episode = {
        "task": "nutpouring",
        "seed": 4,
        "episode_index": 0,
        "success": True,
        "reward": None,
        "episode_length": 480,
        "status": "SUCCEEDED",
        "metrics": {"effective_seed": 4.0},
        "video_path": str(video),
        "failure_reason": None,
    }
    progress = tmp_path / "progress.jsonl"
    append_jsonl(
        progress,
        {
            "record_type": "episode",
            "identity": "pinned-a",
            "status": "SUCCEEDED",
            "task": "nutpouring",
            "seed": 4,
            "episode_index": 0,
            "episode": episode,
        },
    )
    assert load_completed(progress, "pinned-a") == {("nutpouring", 4, 0): episode}
    with pytest.raises(RuntimeError, match="different pinned evaluation"):
        load_completed(progress, "pinned-b")


def test_policy_server_command_uses_outer_interpreter_and_exact_wrapper(tmp_path: Path) -> None:
    command = build_policy_server_command(tmp_path / "groot", tmp_path / "checkpoint", 4567)
    assert command[0]
    assert command[-1] == "--use-sim-policy-wrapper"
    assert command[command.index("--embodiment-tag") + 1] == "GR1"
    assert command[command.index("--port") + 1] == "4567"


def test_isaac_eula_consent_must_come_from_operator_context(tmp_path: Path) -> None:
    runtime = {
        "environment_path": str(tmp_path / "isaac-env"),
        "source_dir": str(tmp_path / "eval-tasks"),
        "profile_snapshot": {"environment": {"OMNI_KIT_ACCEPT_EULA": "YES"}},
    }
    with pytest.raises(RuntimeError, match="operator_environment.*NVIDIA Isaac Sim EULA"):
        build_isaac_environment(runtime)

    runtime["operator_environment"] = {"OMNI_KIT_ACCEPT_EULA": "YES"}
    environment = build_isaac_environment(runtime)
    assert environment["OMNI_KIT_ACCEPT_EULA"] == "YES"


def test_evaluator_environment_applies_profile_operations_without_encoding_consent(
    tmp_path: Path,
) -> None:
    runtime = {
        "environment_path": str(tmp_path / "isaac-env"),
        "source_dir": str(tmp_path / "eval-tasks"),
        "operator_environment": {"OMNI_KIT_ACCEPT_EULA": "YES"},
        "profile_snapshot": {
            "environment": {},
            "environment_operations": [
                {
                    "name": "LD_LIBRARY_PATH",
                    "operation": "prepend",
                    "value": "{{runtime.environment_path}}/.skynet/compat/libglu",
                    "separator": ":",
                }
            ],
        },
    }
    environment = build_isaac_environment(runtime)
    assert environment["LD_LIBRARY_PATH"].split(":")[0] == str(
        tmp_path / "isaac-env" / ".skynet" / "compat" / "libglu"
    )


def test_readiness_smoke_requires_external_eula_and_always_writes_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    output = tmp_path / "readiness.json"

    assert run_evaluator_readiness_smoke(
        output_path=output,
        source_dir=tmp_path / "evaltasks",
        simulation_device="cuda:0",
    ) == 1

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["schema_version"] == "skynet.evaluator-readiness/v1"
    assert result["ready"] is False
    assert "OMNI_KIT_ACCEPT_EULA=YES" in result["errors"][0]
    assert {"isaaclab", "isaaclab_assets", "isaaclab_tasks"} <= set(
        result["actual"]["imports"]
    )
    assert result["actual"]["checks"] == {
        "environment_constructed": False,
        "environment_reset": False,
        "media_writer": False,
        "media_nonempty": False,
    }
    assert "Traceback (most recent call last)" in result["actual"]["traceback"]
    assert "OMNI_KIT_ACCEPT_EULA=YES" in capsys.readouterr().err


def test_readiness_smoke_encodes_official_import_order_and_exact_tasks() -> None:
    source = inspect.getsource(run_evaluator_readiness_smoke)
    assert source.index('import_module("isaacsim")') < source.index(
        'import_module("pinocchio")'
    )
    assert source.index('import_module("pinocchio")') < source.index(
        'import_module("isaaclab.app")'
    )
    launcher_index = source.index("launcher, launcher_app = _launch_isaac_app(")
    assert ISAAC_KIT_ARGS == "--/rtx/verifyDriverVersion/enabled=false"
    assert OMNI_WARP_VERSION == "1.8.1"
    assert len(OMNI_WARP_BINARY_SHA256) == 64
    assert "--ext-folder " in ISAAC_CLIENT_SOURCE
    assert (
        "--enable omni.warp.core-1.8.1 --enable omni.warp-1.8.1"
        in ISAAC_CLIENT_SOURCE
    )
    assert "outside pinned extension" in ISAAC_CLIENT_SOURCE
    assert "kit_args=build_isaac_kit_args(kit_extension_folder)" in source[launcher_index:]
    assert source.index('actual["launch_phase"] = phase') < launcher_index
    assert source.index("environment.close()") < source.index(
        "result = _evaluator_readiness_result("
    )
    assert source.index("result = _evaluator_readiness_result(") < source.index(
        "launcher_app.close()"
    )
    for post_launch_import in (
        'import_module("isaaclab")',
        'import_module("isaaclab_assets")',
        'import_module("isaaclab_tasks")',
        'import_module("isaaclab_tasks.utils.parse_cfg")',
        "import_module(registration_module)",
    ):
        assert launcher_index < source.index(post_launch_import)
    for definition in TASK_DEFINITIONS.values():
        assert definition["environment_id"] in source or definition["environment_id"] in str(
            TASK_DEFINITIONS
        )


def test_readiness_result_is_success_before_native_app_shutdown() -> None:
    registration_module = "isaaclab_eval_tasks.tasks"
    actual = {
        "imports": {"isaacsim": True, registration_module: True},
        "gym_registrations": {
            registration_module: {
                "module_imported": True,
                "ids": {"Isaac-NutPour-GR1T2-ClosedLoop-v0": True},
            }
        },
        "gpu": {"cuda_available": True, "device_count": 1},
        "checks": {
            "environment_constructed": True,
            "environment_reset": True,
            "media_writer": True,
            "media_nonempty": True,
        },
    }

    result = _evaluator_readiness_result(
        actual=actual,
        errors=[],
        registration_module=registration_module,
    )

    assert result["ready"] is True
    assert result["errors"] == []


def test_launch_isaac_app_constructs_then_creates_stage() -> None:
    phases: list[str] = []
    launch_arguments: dict[str, object] = {}

    class FakeApp:
        pass

    fake_app = FakeApp()

    class FakeLauncher:
        app = fake_app

    def app_launcher(**kwargs):
        launch_arguments.update(kwargs)
        return FakeLauncher()

    class FakeUsdContext:
        stage = None
        create_count = 0

        def get_stage(self):
            return self.stage

        def new_stage(self):
            self.create_count += 1
            self.stage = object()

    usd_context = FakeUsdContext()

    class FakeOmniUsd:
        @staticmethod
        def get_context():
            return usd_context

    launcher, app = _launch_isaac_app(
        app_launcher,
        device="cuda:0",
        kit_args="--pinned",
        phase_marker=phases.append,
        stage_api_loader=lambda: FakeOmniUsd,
    )

    assert isinstance(launcher, FakeLauncher)
    assert app is fake_app
    assert launch_arguments == {
        "headless": True,
        "enable_cameras": True,
        "num_envs": 1,
        "device": "cuda:0",
        "kit_args": "--pinned",
    }
    assert usd_context.create_count == 1
    assert phases == [
        "app_launcher_constructor_started",
        "app_launcher_constructor_complete",
        "usd_stage_creation_started",
        "usd_stage_ready",
    ]
    assert "_wait_for_viewport" not in inspect.getsource(_launch_isaac_app)


def test_launch_isaac_app_persists_started_phase_on_constructor_failure() -> None:
    phases: list[str] = []

    def failing_launcher(**_kwargs):
        raise RuntimeError("constructor failed")

    with pytest.raises(RuntimeError, match="constructor failed"):
        _launch_isaac_app(
            failing_launcher,
            device="cuda:0",
            kit_args="--pinned",
            phase_marker=phases.append,
            stage_api_loader=lambda: None,
        )

    assert phases == ["app_launcher_constructor_started"]
