"""Shared declarations for recorded-data XPolicyLab policies."""

from pathlib import Path

SUPPORT = Path(__file__).parent


def support_files(policy):
    files = {
        "adapter-support/xpolicy_runtime.py": (
            SUPPORT / "xpolicy_runtime.py"
        ).read_text(),
        "adapter-support/artifacts.py": (
            SUPPORT.parents[1] / "ops/datasets/artifacts.py"
        ).read_text(),
    }
    files["adapter-support/skynet_" + policy + "_training.py"] = (
        SUPPORT / (policy + "_training.py")
    ).read_text()
    return files


def progress_contract():
    from . import TrainingProgressContract, TrainingProgressJsonlSource

    return TrainingProgressContract(
        unit="epoch",
        total_path="native.config.epochs",
        starts_at_zero=True,
        source=TrainingProgressJsonlSource(
            path="artifacts/logs.json.txt",
            completed_key="epoch",
            completed_offset=1,
            required_key="val_loss",
            metrics={
                "train_loss": "train/loss",
                "val_loss": "validation/loss",
                "lr": "train/learning_rate",
                "global_step": "training/global_step",
            },
        ),
    )


def evaluation(policy):
    from . import EvaluationAdapterMetadata, CommandTemplate

    files = support_files(policy)
    files.update(
        {
            "adapter-support/evaluation_workers.py": (
                SUPPORT / "evaluation_workers.py"
            ).read_text(),
            "adapter-support/dexverse_readiness.py": (
                SUPPORT / "dexverse_readiness.py"
            ).read_text(),
            "adapter-support/xpolicy_evaluation.py": (
                SUPPORT / "xpolicy_evaluation.py"
            ).read_text(),
            "adapter-support/dexverse_evaluation.py": (
                SUPPORT / "dexverse_evaluation.py"
            ).read_text(),
            "adapter-support/images.py": (
                SUPPORT.parents[1] / "ops/xr/images.py"
            ).read_text(),
            "adapter-support/wrist.py": (
                SUPPORT.parents[1] / "ops/xr/wrist.py"
            ).read_text(),
        }
    )
    argv = [
        "python",
        "{{tokens.run_dir}}/adapter-support/evaluation_workers.py",
        "--context",
        "{{tokens.run_dir}}/adapter-support/evaluation-context.json",
        "--source-dir",
        "{{tokens.source_dir}}",
    ]
    return EvaluationAdapterMetadata(
        maximum_parallelism=8,
        environment="isaac_lab",
        suites=["dexverse_recorded"],
        runtime_profile_id="isaacsim-5.1.0_isaaclab-2.3.2_py311",
        command=CommandTemplate(
            argv=argv,
            resume_argv=[],
            capsule_files=files,
            environment={"SKYNET_EVAL_RESUME_GRANULARITY": "episode"},
            required_values=[
                "evaluation.checkpoint.sha256",
                "evaluation.evaluator_runtime.python_executable",
                "evaluation.evaluator_runtime.source_dir",
                "evaluation.policy.native_config.dataset_path",
                "evaluation.policy.native_config.dataset_manifest_sha256",
            ],
        ),
    )
