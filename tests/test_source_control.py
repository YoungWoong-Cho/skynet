from __future__ import annotations

import base64
import hashlib

import pytest

from skynet_app.adapters import (
    AdapterInputField,
    RepositoryChoiceSource,
    RepositoryYamlCatalogMetadataField,
    RepositoryYamlCatalogSource,
    RepositoryYamlChoiceSource,
    RepositoryYamlMetadataField,
    builtin_adapter_manifests,
)
from skynet_app.pipeline_api import PipelineService
from skynet_app.source_control import SourceDiscovery


COMMIT = "2" * 40


class StaticFilesCluster:
    hosts = ("sky1", "sky2")

    def __init__(self, files: dict[str, str]) -> None:
        self.files = files

    def run_with_fallback(self, command: str, gateway: str = "auto", timeout: int = 60):
        lines = [f"COMMIT\t{COMMIT}"]
        for path, source in self.files.items():
            content = source.encode("utf-8")
            lines.append(
                f"FILE\t{path}\t{len(content)}\t{hashlib.sha256(content).hexdigest()}"
            )
            lines.append(f"CONTENT\t{path}\t{base64.b64encode(content).decode('ascii')}")
        return "sky1", "\n".join(lines)


def openpi_field() -> dict[str, object]:
    manifest = next(item for item in builtin_adapter_manifests() if item.slug == "openpi")
    field = next(item for item in manifest.train.input_fields if item.choice_source is not None)
    return field.model_dump(mode="json")


def egoverse_field() -> tuple[dict[str, object], object]:
    manifest = next(
        item for item in builtin_adapter_manifests() if item.slug == "egoverse"
    )
    field = next(item for item in manifest.train.input_fields if item.choice_source)
    return field.model_dump(mode="json"), manifest


def test_static_registry_discovers_direct_and_imported_factory_choices_without_execution():
    files = {
        "src/openpi/training/config.py": '''
import openpi.training.misc.extra_config as extra_config
class TrainConfig:
    batch_size: int = 32
    num_workers: int = 2
_CONFIGS = [
    TrainConfig(name="direct", batch_size=256, lr_schedule=Schedule(peak_lr=5e-5)),
    *extra_config.get_extra_configs(),
]
''',
        "src/openpi/training/misc/extra_config.py": '''
def get_extra_configs():
    raise RuntimeError("repository code must never run")
    return [TrainConfig(name="imported")]
''',
    }
    field = openpi_field()
    field["choice_source"]["metadata_fields"] = [
        {
            "source_path": "batch_size",
            "canonical_path": "train.batch.value",
            "value_map": {},
        },
        {
            "source_path": "num_workers",
            "canonical_path": "train.num_workers_per_rank",
            "value_map": {},
        },
        {
            "source_path": "lr_schedule.peak_lr",
            "canonical_path": "train.learning_rate",
            "value_map": {},
        },
    ]
    field["choice_source"]["supporting_files"] = [
        "src/openpi/training/misc/extra_config.py"
    ]
    discovery = SourceDiscovery(StaticFilesCluster(files))

    options = discovery.input_options(
        "https://github.com/Physical-Intelligence/openpi", COMMIT, [field]
    )["native.config.config_name"]

    assert options["choices"] == ["direct", "imported"]
    assert options["complete"] is True
    assert options["source"]["commit"] == COMMIT
    assert len(options["source"]["files"]) == 2
    assert options["metadata"]["direct"]["values"] == {
        "train.batch.value": 256,
        "train.learning_rate": 5e-5,
        "train.num_workers_per_rank": 2,
    }
    assert options["metadata"]["imported"]["values"] == {
        "train.batch.value": 32,
        "train.num_workers_per_rank": 2,
    }
    assert options["metadata"]["direct"]["evidence"]["train.batch.value"][
        "expression_sha256"
    ]


def test_static_registry_fails_closed_for_computed_choice():
    files = {
        "src/openpi/training/config.py": '''
def execute_me():
    raise RuntimeError("must not execute")
_CONFIGS = [TrainConfig(name=execute_me())]
''',
        "src/openpi/training/misc/roboarena_config.py": "",
        "src/openpi/training/misc/polaris_config.py": "",
    }
    discovery = SourceDiscovery(StaticFilesCluster(files))

    option = discovery.input_options(
        "https://github.com/Physical-Intelligence/openpi", COMMIT, [openpi_field()]
    )["native.config.config_name"]

    assert option["complete"] is False
    assert option["choices"] == []
    assert any("non-literal name" in warning for warning in option["warnings"])


def test_static_yaml_catalog_discovers_and_composes_each_choice_without_execution():
    field, manifest = egoverse_field()
    files = {
        "egomimic/hydra_configs/train_beta.yaml": """
defaults:
  - train_alpha
  - override model: beta
  - override data: beta
""",
        "egomimic/hydra_configs/train_alpha.yaml": """
defaults:
  - model: alpha
  - trainer: ddp
  - data: alpha
  - _self_
""",
        "egomimic/hydra_configs/evaluate.yaml": "model: ignored\n",
        "egomimic/hydra_configs/data/train_nested.yaml": "ignored: true\n",
        "egomimic/hydra_configs/model/alpha.yaml": """
optimizer:
  _target_: torch.optim.AdamW
  lr: 1e-4
""",
        "egomimic/hydra_configs/model/beta.yaml": """
optimizer:
  lr: 5e-5
""",
        "egomimic/hydra_configs/data/alpha.yaml": """
train_dataloader_params:
  alpha:
    batch_size: 32
    num_workers: 6
""",
        "egomimic/hydra_configs/data/beta.yaml": """
train_dataloader_params:
  beta_one:
    batch_size: 64
    num_workers: 8
  beta_two:
    batch_size: 64
    num_workers: 8
""",
        "egomimic/hydra_configs/trainer/ddp.yaml": """
defaults:
  - default
strategy: ddp
""",
        "egomimic/hydra_configs/trainer/default.yaml": """
_target_: lightning.pytorch.trainer.Trainer
max_epochs: 2000
precision: bf16
limit_train_batches: 100
""",
    }
    discovery = SourceDiscovery(StaticFilesCluster(files))

    option = discovery.input_options(
        "https://github.com/GaTech-RL2/EgoVerse", COMMIT, [field]
    )["native.config.config_name"]

    assert option["choices"] == ["train_alpha", "train_beta"]
    assert option["complete"] is True
    assert option["source"]["kind"] == "yaml_static_catalog"
    assert option["source"]["commit"] == COMMIT
    assert option["source"]["directory"] == "egomimic/hydra_configs"
    assert option["source"]["filename_pattern"] == "train*.yaml"
    assert option["metadata"]["train_alpha"]["values"] == {
        "train.learning_rate": 1e-4,
        "train.batch.value": 32,
        "train.num_workers_per_rank": 6,
        "train.max_epochs": 2000,
        "train.precision": "bf16",
    }
    assert option["metadata"]["train_beta"]["values"] == {
        "train.learning_rate": 5e-5,
        "train.batch.value": 64,
        "train.num_workers_per_rank": 8,
        "train.max_epochs": 2000,
        "train.precision": "bf16",
    }
    for choice in option["choices"]:
        metadata = option["metadata"][choice]
        assert metadata["unresolved"]["train.max_steps"]["status"] == "not_declared"
        assert metadata["unresolved"][
            "train.batch.gradient_accumulation_steps"
        ]["status"] == "not_declared"
        assert metadata["complete"] is False
        assert any("trainer.max_steps" in warning for warning in metadata["warnings"])
    assert manifest.defaults.hyperparameters.batch_semantics == "per_device"
    assert manifest.defaults.hyperparameters.gradient_accumulation_steps == 1
    assert manifest.train.argument_validation is None
    beta_batch = option["metadata"]["train_beta"]["evidence"]["train.batch.value"]
    assert beta_batch["collapsed_identical_matches"] is True
    assert len(beta_batch["matches"]) == 2
    assert option["metadata"]["train_alpha"]["evidence"]["train.learning_rate"][
        "file"
    ].endswith("model/alpha.yaml")
    assert option["metadata"]["train_beta"]["evidence"]["train.learning_rate"][
        "file"
    ].endswith("model/beta.yaml")


def test_repository_choice_schema_rejects_unsafe_or_conflicting_declarations():
    with pytest.raises(ValueError, match="safe relative Python paths"):
        RepositoryChoiceSource(
            kind="python_static_registry",
            entrypoint="../config.py",
            registry="_CONFIGS",
            constructor="TrainConfig",
        )
    with pytest.raises(ValueError, match="safe relative YAML paths"):
        RepositoryYamlChoiceSource(
            entrypoint="../config.yaml",
            choice="train",
            metadata_fields=[
                RepositoryYamlMetadataField(
                    source_file="config.yaml",
                    source_path="trainer.max_epochs",
                    canonical_path="train.max_epochs",
                )
            ],
        )
    with pytest.raises(ValueError, match="safe relative paths"):
        RepositoryYamlCatalogSource(
            directory="../hydra_configs",
            filename_pattern="train*.yaml",
            metadata_fields=[
                RepositoryYamlCatalogMetadataField(
                    source_path="trainer.max_epochs",
                    canonical_path="train.max_epochs",
                )
            ],
        )
    source = RepositoryChoiceSource(
        kind="python_static_registry",
        entrypoint="config.py",
        registry="_CONFIGS",
        constructor="TrainConfig",
    )
    with pytest.raises(ValueError, match="both choices and choice_source"):
        AdapterInputField(
            path="native.config.name",
            label="Name",
            kind="string",
            choices=["fixed"],
            choice_source=source,
        )


def test_normalization_validation_enforces_complete_commit_specific_choices():
    manifest = next(item for item in builtin_adapter_manifests() if item.slug == "openpi")

    class Options:
        def __init__(self, complete: bool) -> None:
            self.complete = complete
            self.calls: list[tuple[str, str]] = []

        def input_options(self, repository, revision, fields, gateway, project_subdirectory):
            self.calls.append((repository, revision))
            return {
                "native.config.config_name": {
                    "choices": ["debug"],
                    "complete": self.complete,
                    "warnings": [] if self.complete else ["dynamic registry expression"],
                }
            }

    service = PipelineService.__new__(PipelineService)
    complete = Options(True)
    service._repository_input_options = lambda source, _manifest, gateway: complete.input_options(
        source["repository"],
        source["revision"],
        [],
        gateway,
        source["project_subdirectory"],
    )
    source = {
        "repository": "https://github.com/Physical-Intelligence/openpi",
        "revision": COMMIT,
        "project_subdirectory": ".",
    }
    document = {"native": {"config": {"config_name": "unknown"}}}

    with pytest.raises(ValueError, match="not registered at source commit"):
        service._validate_repository_input_choices(document, source, manifest, "auto")
    assert complete.calls == [(source["repository"], COMMIT)]

    incomplete = Options(False)
    service._repository_input_options = lambda source, _manifest, gateway: incomplete.input_options(
        source["repository"],
        source["revision"],
        [],
        gateway,
        source["project_subdirectory"],
    )
    with pytest.raises(ValueError, match="choice discovery is incomplete"):
        service._validate_repository_input_choices(
            {"native": {"config": {"config_name": "debug"}}},
            source,
            manifest,
            "auto",
        )


def test_egoverse_repository_catalog_rejects_custom_config_names():
    manifest = next(
        item for item in builtin_adapter_manifests() if item.slug == "egoverse"
    )

    class Options:
        def input_options(
            self, repository, revision, fields, gateway, project_subdirectory
        ):
            return {
                "native.config.config_name": {
                    "choices": ["train_zarr_cartesian"],
                    "complete": True,
                    "warnings": [],
                }
            }

    service = PipelineService.__new__(PipelineService)
    service._repository_input_options = lambda *_args: Options().input_options(
        None, None, None, None, None
    )
    source = {
        "repository": "https://github.com/GaTech-RL2/EgoVerse",
        "revision": COMMIT,
        "project_subdirectory": ".",
    }

    with pytest.raises(ValueError, match="not registered at source commit"):
        service._validate_repository_input_choices(
            {"native": {"config": {"config_name": "train_zarr_keypoints"}}},
            source,
            manifest,
            "auto",
        )
