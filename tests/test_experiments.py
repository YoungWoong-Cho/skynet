from __future__ import annotations

import pytest
from pydantic import ValidationError

from skynet_app.adapters import resolve_adapter_plan
from skynet_app.experiments import (
    CheckpointReference,
    EnvironmentReference,
    EpisodeResult,
    EvaluatorReference,
    ExperimentSpec,
    build_canonical_result,
    expand_sweep,
    get_evaluation_catalog,
)


COMMIT = "a" * 40


def make_spec(**overrides):
    payload = {
        "identity": {"project": "tests", "experiment": "smoke"},
        "source": {
            "repository": "https://github.com/example/repository.git",
            "revision": COMMIT,
            "adapter": "generic",
        },
        "runtime": {"backend": "existing", "bootstrap_uv": False},
        "resources": {
            "account": "rl2-lab",
            "partition": "rl2-lab",
            "gpu": {"mode": "explicit", "count": 2, "type": "a40"},
            "time_limit": "04:00:00",
        },
        "native": {
            "argv": ["python", "train.py", "--seed", "42"],
            "resume_argv": ["--resume", "{{SKYNET_RESUME_CHECKPOINT}}"],
        },
    }
    payload.update(overrides)
    return ExperimentSpec.model_validate(payload)


def test_account_partition_are_atomic():
    with pytest.raises(ValidationError, match="partition and account"):
        make_spec(
            resources={
                "account": "overcap",
                "partition": "rl2-lab",
                "gpu": {"mode": "explicit", "count": 1},
            }
        )


def test_partition_specific_time_limits():
    with pytest.raises(ValidationError, match="04:00:00"):
        make_spec(
            resources={
                "account": "rl2-lab",
                "partition": "rl2-lab",
                "time_limit": "04:00:01",
                "gpu": {"mode": "explicit", "count": 1},
            }
        )
    spec = make_spec(
        resources={
            "account": "overcap",
            "partition": "overcap",
            "time_limit": "2-00:00:00",
            "gpu": {"mode": "explicit", "count": 1},
        }
    )
    assert spec.resources.time_limit == "2-00:00:00"


def test_single_node_allows_multiple_gpus_but_not_multiple_nodes():
    assert make_spec().resources.gpu.count == 2
    with pytest.raises(ValidationError):
        make_spec(
            resources={
                "nodes": 2,
                "account": "rl2-lab",
                "partition": "rl2-lab",
                "gpu": {"mode": "explicit", "count": 2},
            }
        )


def test_grid_sweep_expands_variants_and_seeds():
    spec = make_spec(
        sweep={
            "strategy": "grid",
            "axes": {"train.learning_rate": [0.001, 0.0001]},
            "seeds": [41, 42],
        }
    )
    variants = expand_sweep(spec)
    assert len(variants) == 4
    assert {variant.seed for variant in variants} == {41, 42}
    assert len({variant.resolved_spec_sha256 for variant in variants}) == 4


def test_generic_adapter_uses_structured_argv():
    plan = resolve_adapter_plan(make_spec())
    assert plan.runnable
    assert plan.argv == ["python", "train.py", "--seed", "42"]


def test_openpi_rejects_gradient_accumulation():
    spec = make_spec(
        source={
            "repository": "https://github.com/Physical-Intelligence/openpi.git",
            "revision": COMMIT,
            "adapter": "openpi",
        },
        train={"batch": {"value": 8, "gradient_accumulation_steps": 2}},
        native={"config": {"config_name": "pi0_libero"}},
    )
    plan = resolve_adapter_plan(spec)
    assert any("gradient accumulation" in blocker for blocker in plan.blockers)


def test_openpi_inherited_batch_does_not_block_or_emit_on_multiple_gpus():
    from skynet_app.adapters import ManifestAdapter, builtin_adapter_manifests

    manifest = next(
        item for item in builtin_adapter_manifests() if item.slug == "openpi"
    )
    spec = make_spec(
        source={
            "repository": "https://github.com/Physical-Intelligence/openpi.git",
            "revision": COMMIT,
            "adapter": "openpi",
        },
        native={"config": {"config_name": "debug"}},
    )
    round_trip = ExperimentSpec.model_validate(
        spec.model_dump(mode="json", by_alias=True)
    )
    plan = ManifestAdapter(manifest).resolve(round_trip)

    assert round_trip.intent.explicit_parameters == []
    assert plan.runnable is True
    assert "--fsdp-devices" in plan.argv
    for flag in (
        "--batch-size",
        "--num-workers",
        "--num-train-steps",
        "--pytorch-training-precision",
        "--save-interval",
        "--seed",
    ):
        assert flag not in plan.argv


def test_openpi_explicit_batch_still_maps_and_validates_on_multiple_gpus():
    from skynet_app.adapters import ManifestAdapter, builtin_adapter_manifests

    manifest = next(
        item for item in builtin_adapter_manifests() if item.slug == "openpi"
    )

    def plan_for(batch: dict):
        return ManifestAdapter(manifest).resolve(
            make_spec(
                source={
                    "repository": "https://github.com/Physical-Intelligence/openpi.git",
                    "revision": COMMIT,
                    "adapter": "openpi",
                },
                train={"batch": batch},
                native={"config": {"config_name": "debug"}},
            )
        )

    per_device = plan_for({"value": 4})
    assert per_device.argv[per_device.argv.index("--batch-size") + 1] == "4"
    assert any("--batch-size is global" in reason for reason in per_device.blockers)

    indivisible_global = plan_for(
        {"declared_semantics": "global_before_accumulation", "value": 3}
    )
    assert (
        indivisible_global.argv[indivisible_global.argv.index("--batch-size") + 1]
        == "3"
    )
    assert any("must be divisible" in reason for reason in indivisible_global.blockers)


def test_openpi_batch_compatibility_manifest_schema_round_trip():
    from skynet_app.adapters import AdapterManifest, builtin_adapter_manifests

    manifests = {
        manifest.slug: manifest for manifest in builtin_adapter_manifests()
    }
    payload = manifests["openpi"].model_dump(mode="json")
    assert payload["train"]["batch_compatibility"] == {
        "schema_version": "skynet.batch-compatibility/v1",
        "allowed_semantics": [
            "per_device",
            "global_before_accumulation",
            "global_effective",
        ],
        "multi_gpu_allowed_semantics": [
            "global_before_accumulation",
            "global_effective",
        ],
        "supports_gradient_accumulation": False,
        "batch_size_divisible_by": "resolved_gpu_count",
    }

    restored = AdapterManifest.model_validate(payload)
    assert restored.train.batch_compatibility == (
        manifests["openpi"].train.batch_compatibility
    )
    assert manifests["generic"].train.batch_compatibility is None


@pytest.mark.parametrize(
    ("gpu_count", "semantics", "batch_size", "expected_blocker"),
    [
        (1, "per_device", 3, None),
        (2, "global_before_accumulation", 4, None),
        (4, "global_effective", 8, None),
        (2, "per_device", 4, "--batch-size is global"),
        (4, "per_device", 8, "--batch-size is global"),
        (2, "global_before_accumulation", 3, "must be divisible"),
        (4, "global_effective", 6, "must be divisible"),
    ],
)
def test_openpi_declared_batch_compatibility_for_one_two_and_four_gpus(
    gpu_count, semantics, batch_size, expected_blocker
):
    from skynet_app.adapters import ManifestAdapter, builtin_adapter_manifests

    manifest = next(
        item for item in builtin_adapter_manifests() if item.slug == "openpi"
    )
    spec = make_spec(
        source={
            "repository": "https://github.com/Physical-Intelligence/openpi.git",
            "revision": COMMIT,
            "adapter": "openpi",
        },
        resources={
            "account": "rl2-lab",
            "partition": "rl2-lab",
            "gpu": {"mode": "explicit", "count": gpu_count, "type": "l40s"},
            "time_limit": "04:00:00",
        },
        train={
            "batch": {
                "declared_semantics": semantics,
                "value": batch_size,
                "gradient_accumulation_steps": 1,
            }
        },
        native={"config": {"config_name": "debug"}},
    )

    plan = ManifestAdapter(manifest).resolve(spec)

    if expected_blocker is None:
        assert plan.runnable is True
    else:
        assert plan.runnable is False
        assert any(expected_blocker in blocker for blocker in plan.blockers)


@pytest.mark.parametrize("gpu_count", [1, 2, 4])
def test_openpi_declared_batch_compatibility_rejects_accumulation(gpu_count):
    from skynet_app.adapters import ManifestAdapter, builtin_adapter_manifests

    manifest = next(
        item for item in builtin_adapter_manifests() if item.slug == "openpi"
    )
    spec = make_spec(
        source={
            "repository": "https://github.com/Physical-Intelligence/openpi.git",
            "revision": COMMIT,
            "adapter": "openpi",
        },
        resources={
            "account": "rl2-lab",
            "partition": "rl2-lab",
            "gpu": {"mode": "explicit", "count": gpu_count, "type": "l40s"},
            "time_limit": "04:00:00",
        },
        train={
            "batch": {
                "declared_semantics": "global_before_accumulation",
                "value": 4,
                "gradient_accumulation_steps": 2,
            }
        },
        native={"config": {"config_name": "debug"}},
    )

    plan = ManifestAdapter(manifest).resolve(spec)

    assert plan.runnable is False
    assert any("gradient accumulation" in blocker for blocker in plan.blockers)


def test_native_text_config_maps_to_canonical_native_config():
    from skynet_app.pipeline_api import _parse_native

    config, overrides, argv, resume_argv = _parse_native(
        ['config.config_name="debug"']
    )

    assert config == {"config_name": "debug"}
    assert overrides == {}
    assert argv == []
    assert resume_argv == []


def test_openpi_missing_config_blocks_and_debug_is_runnable():
    source = {
        "repository": "https://github.com/Physical-Intelligence/openpi.git",
        "revision": COMMIT,
        "adapter": "openpi",
    }
    train = {
        "batch": {
            "declared_semantics": "global_before_accumulation",
            "value": 12,
            "gradient_accumulation_steps": 1,
        }
    }
    blocked = resolve_adapter_plan(
        make_spec(source=source, train=train, native={"config": {}})
    )
    runnable = resolve_adapter_plan(
        make_spec(
            source=source,
            train=train,
            native={"config": {"config_name": "debug"}},
        )
    )

    assert blocked.runnable is False
    assert "openpi requires native.config.config_name" in blocked.blockers
    assert runnable.runnable is True
    assert runnable.argv[:3] == ["python", "scripts/train.py", "debug"]


def test_train_precision_is_canonical_and_rejects_unknown_values():
    assert make_spec(train={"precision": "bf16"}).train.precision == "bf16"
    with pytest.raises(ValidationError, match="precision"):
        make_spec(train={"precision": "float16"})


def test_generic_manifest_value_mapping_is_deterministic_and_strict():
    from skynet_app.adapters import (
        ArgumentBinding,
        CommandTemplate,
        ManifestAdapter,
        builtin_adapter_manifests,
    )

    base = next(manifest for manifest in builtin_adapter_manifests() if manifest.slug == "generic")
    binding = ArgumentBinding(
        flag="--backend-dtype",
        value_map={"fp32": "float32", "bf16": "bfloat16"},
    )
    assert list(binding.value_map) == ["bf16", "fp32"]
    manifest = base.model_copy(
        update={
            "legacy_handler": None,
            "train": CommandTemplate(
                argv=["python", "train.py"],
                parameter_flags={"train.precision": binding},
                retry_clean_argv=["--fresh"],
            ),
        },
        deep=True,
    )

    supported = ManifestAdapter(manifest).resolve(
        make_spec(train={"precision": "bf16"}, native={})
    )
    unsupported = ManifestAdapter(manifest).resolve(
        make_spec(train={"precision": "fp16"}, native={})
    )

    assert supported.argv == ["python", "train.py", "--backend-dtype", "bfloat16"]
    assert supported.retry_clean_argv == ["--fresh"]
    round_trip = type(manifest).model_validate(manifest.model_dump(mode="json"))
    assert round_trip.train.retry_clean_argv == ["--fresh"]
    assert supported.runnable is True
    assert unsupported.argv == ["python", "train.py"]
    assert unsupported.runnable is False
    assert any("does not support train.precision=fp16" in reason for reason in unsupported.blockers)


def test_manifest_canonical_flags_only_emit_for_explicit_parameters():
    from skynet_app.adapters import (
        ArgumentBinding,
        CommandTemplate,
        ManifestAdapter,
        builtin_adapter_manifests,
    )

    base = next(
        manifest for manifest in builtin_adapter_manifests() if manifest.slug == "generic"
    )
    bindings = {
        "train.batch.value": ArgumentBinding(flag="--batch-size"),
        "train.checkpoint.save_every_steps": ArgumentBinding(flag="--save-every"),
        "train.max_steps": ArgumentBinding(flag="--max-steps"),
        "train.num_workers_per_rank": ArgumentBinding(flag="--num-workers"),
        "train.seed": ArgumentBinding(flag="--seed"),
    }
    manifest = base.model_copy(
        update={
            "legacy_handler": None,
            "train": CommandTemplate(
                argv=["python", "train.py"], parameter_flags=bindings
            ),
        },
        deep=True,
    )

    inherited = ManifestAdapter(manifest).resolve(make_spec(native={}))
    assert inherited.argv == ["python", "train.py"]
    assert inherited.runnable is True

    explicit_spec = make_spec(
        train={
            "batch": {"value": 1},
            "checkpoint": {"save_every_steps": 1000},
            "max_steps": 1000,
            "num_workers_per_rank": 4,
            "seed": 42,
        },
        native={},
    )
    explicit = ManifestAdapter(manifest).resolve(
        ExperimentSpec.model_validate(
            explicit_spec.model_dump(mode="json", by_alias=True)
        )
    )
    for path, binding in bindings.items():
        flag_position = explicit.argv.index(binding.flag)
        value = explicit_spec
        for part in path.split("."):
            value = getattr(value, part)
        assert explicit.argv[flag_position + 1] == str(value)
    assert explicit.runnable is True


def test_egoverse_checkpoint_steps_are_a_new_manifest_only_mapping():
    from skynet_app.adapters import ManifestAdapter, builtin_adapter_manifests

    canonical_path = "train.checkpoint.save_every_steps"
    manifest = next(
        item for item in builtin_adapter_manifests() if item.slug == "egoverse"
    )
    binding = manifest.train.parameter_flags[canonical_path]
    assert binding.flag == "+callbacks.model_checkpoint.every_n_train_steps"
    assert binding.style == "hydra"
    assert binding.companion_arguments == [
        "callbacks.model_checkpoint.every_n_epochs=null"
    ]
    assert canonical_path in manifest.train.supported_canonical_fields

    source = {
        "repository": "https://github.com/GaTech-RL2/EgoVerse",
        "revision": COMMIT,
        "adapter": "egoverse",
    }
    inherited = ManifestAdapter(manifest).resolve(make_spec(source=source, native={}))
    assert not any("every_n_train_steps" in argument for argument in inherited.argv)
    assert "callbacks.model_checkpoint.every_n_epochs=null" not in inherited.argv

    explicit_spec = make_spec(
        source=source,
        train={"checkpoint": {"save_every_steps": 37}},
        native={},
    )
    explicit = ManifestAdapter(manifest).resolve(explicit_spec)
    assert "+callbacks.model_checkpoint.every_n_train_steps=37" in explicit.argv
    assert "callbacks.model_checkpoint.every_n_epochs=null" in explicit.argv

    old_train = manifest.train.model_copy(
        update={
            "parameter_flags": {},
            "supported_canonical_fields": [
                field
                for field in manifest.train.supported_canonical_fields
                if field != canonical_path
            ],
        },
        deep=True,
    )
    old_manifest = manifest.model_copy(update={"train": old_train}, deep=True)
    old_plan = ManifestAdapter(old_manifest).resolve(explicit_spec)
    assert not any("every_n_train_steps" in argument for argument in old_plan.argv)
    assert any(canonical_path in blocker for blocker in old_plan.blockers)


def test_egoverse_manifest_emits_exact_hydra_argv_without_runtime_validation():
    from skynet_app.adapters import (
        ManifestAdapter,
        builtin_adapter_manifests,
    )

    manifest = next(
        item for item in builtin_adapter_manifests() if item.slug == "egoverse"
    )
    spec = make_spec(
        source={
            "repository": "https://github.com/GaTech-RL2/EgoVerse",
            "revision": "e17cf98fe4bc234c564b37abc9e155f25e76d566",
            "adapter": "egoverse",
        },
        resources={
            "account": "rl2-lab",
            "partition": "rl2-lab",
            "gpu": {"mode": "explicit", "count": 1, "type": "a40"},
            "time_limit": "04:00:00",
        },
        train={
            "batch": {"gradient_accumulation_steps": 1},
            "checkpoint": {"save_every_steps": 50000},
            "max_steps": 150000,
            "seed": 42,
        },
        native={"config": {"config_name": "train_zarr_cartesian"}},
    )

    plan = ManifestAdapter(manifest).resolve(spec)

    assert plan.runnable is True
    assert plan.argv == [
        "python",
        "egomimic/trainHydra.py",
        "--config-name",
        "train_zarr_cartesian",
        "hydra/launcher=basic",
        "trainer.devices=1",
        "seed=42",
        "paths.output_dir={{SKYNET_RUN_DIR}}/artifacts",
        "+trainer.accumulate_grad_batches=1",
        "+callbacks.model_checkpoint.every_n_train_steps=50000",
        "callbacks.model_checkpoint.every_n_epochs=null",
        "+trainer.max_steps=150000",
    ]
    assert manifest.train.argument_validation is None


def test_repository_argument_validation_schema_and_canonical_compatibility():
    from skynet_app.adapters import (
        AdapterManifest,
        CommandTemplate,
        RepositoryArgumentValidation,
        builtin_adapter_manifests,
        canonical_adapter_manifest,
    )

    for invalid in (
        ["python", "--cfg", "job"],
        ["{{generated_args}}", "{{generated_args}}"],
        ["prefix={{generated_args}}"],
    ):
        with pytest.raises(ValidationError, match="exactly one"):
            RepositoryArgumentValidation(argv=invalid)
    with pytest.raises(ValidationError):
        RepositoryArgumentValidation(
            argv=["{{generated_args}}"], timeout_seconds=301
        )

    generic = next(
        item for item in builtin_adapter_manifests() if item.slug == "generic"
    )
    canonical = canonical_adapter_manifest(generic)
    assert "argument_validation" not in canonical["train"]
    restored = AdapterManifest.model_validate(canonical)
    assert restored.train.argument_validation is None
    assert CommandTemplate().argument_validation is None


def test_boolean_argument_binding_supports_explicit_positive_and_negative_flags():
    from skynet_app.adapters import ArgumentBinding, CommandTemplate, ManifestAdapter, builtin_adapter_manifests

    base = next(manifest for manifest in builtin_adapter_manifests() if manifest.slug == "generic")
    manifest = base.model_copy(
        update={
            "legacy_handler": None,
            "train": CommandTemplate(
                argv=["python", "train.py"],
                parameter_flags={
                    "native.config.tracking_enabled": ArgumentBinding(
                        flag="--tracking-enabled",
                        false_flag="--no-tracking-enabled",
                        style="boolean",
                    )
                },
            ),
        },
        deep=True,
    )

    enabled = ManifestAdapter(manifest).resolve(
        make_spec(native={"config": {"tracking_enabled": True}})
    )
    disabled = ManifestAdapter(manifest).resolve(
        make_spec(native={"config": {"tracking_enabled": False}})
    )

    assert enabled.argv[-1] == "--tracking-enabled"
    assert disabled.argv[-1] == "--no-tracking-enabled"


def test_openpi_manifest_maps_supported_precision_and_blocks_fp16():
    from skynet_app.adapters import ManifestAdapter, builtin_adapter_manifests

    manifest = next(manifest for manifest in builtin_adapter_manifests() if manifest.slug == "openpi")
    assert manifest.train.retry_clean_argv == ["--overwrite"]

    def plan_for(precision=None, overrides=None):
        train = {
            "batch": {
                "declared_semantics": "global_before_accumulation",
                "value": 8,
                "gradient_accumulation_steps": 1,
            }
        }
        if precision is not None:
            train["precision"] = precision
        return ManifestAdapter(manifest).resolve(
            make_spec(
                source={
                    "repository": "https://github.com/Physical-Intelligence/openpi.git",
                    "revision": COMMIT,
                    "adapter": "openpi",
                },
                train=train,
                native={
                    "config": {"config_name": "debug"},
                    "overrides": overrides or {},
                },
            )
        )

    for canonical, native in (("bf16", "bfloat16"), ("fp32", "float32")):
        plan = plan_for(canonical)
        assert plan.retry_clean_argv == ["--overwrite"]
        assert "--overwrite" not in plan.argv
        assert plan.runnable is True
        assert "--precision" not in plan.argv
        position = plan.argv.index("--pytorch-training-precision")
        assert plan.argv[position + 1] == native

    default = plan_for()
    assert default.runnable is True
    assert "--precision" not in default.argv
    assert "--pytorch-training-precision" not in default.argv

    unsupported = plan_for("fp16")
    assert unsupported.runnable is False
    assert "--precision" not in unsupported.argv
    assert "--pytorch-training-precision" not in unsupported.argv
    assert any("does not support train.precision=fp16" in reason for reason in unsupported.blockers)

    conflicting = plan_for("bf16", {"pytorch_training_precision": "float32"})
    assert conflicting.argv.count("--pytorch-training-precision") == 1
    assert conflicting.runnable is False
    assert any("conflicts with existing argument" in reason for reason in conflicting.blockers)


def test_openpi_manifest_maps_only_verified_trainconfig_fields():
    from skynet_app.adapters import ManifestAdapter, builtin_adapter_manifests

    manifest = next(manifest for manifest in builtin_adapter_manifests() if manifest.slug == "openpi")
    plan = ManifestAdapter(manifest).resolve(
        make_spec(
            source={
                "repository": "https://github.com/Physical-Intelligence/openpi.git",
                "revision": COMMIT,
                "adapter": "openpi",
            },
            train={
                "learning_rate": 3e-4,
                "batch": {
                    "declared_semantics": "global_before_accumulation",
                    "value": 32,
                    "gradient_accumulation_steps": 1,
                },
                "num_workers_per_rank": 8,
                "max_steps": 10,
                "seed": 7,
                "checkpoint": {"save_every_steps": 5, "auto_resume": False},
            },
            native={"config": {"config_name": "pi05_libero", "wandb_enabled": False}},
            tracking={
                "providers": [
                    {
                        "provider": "wandb",
                        "enabled": True,
                        "entity": "team",
                        "project": "central-project",
                    }
                ]
            },
        )
    )

    expected = {
        "--batch-size": "32",
        "--num-workers": "8",
        "--num-train-steps": "10",
        "--save-interval": "5",
        "--seed": "7",
    }
    for flag, value in expected.items():
        position = plan.argv.index(flag)
        assert plan.argv[position + 1] == value
    assert plan.argv.count("--wandb-enabled") == 1
    assert "--no-wandb-enabled" not in plan.argv
    project_flag = plan.argv.index("--project-name")
    assert plan.argv[project_flag + 1] == "central-project"
    assert plan.native_config["wandb_enabled"] is True
    assert plan.native_config["project_name"] == "central-project"
    assert [item.provider for item in plan.native_tracking] == ["wandb"]
    assert plan.native_tracking[0].run_id_file == (
        "{{SKYNET_RUN_DIR}}/checkpoints/pi05_libero/smoke/wandb_id.txt"
    )
    assert "native.config.wandb_enabled" not in {
        field.path for field in manifest.train.input_fields
    }
    assets_flag = plan.argv.index("--assets-base-dir")
    assert plan.argv[assets_flag + 1] == (
        "{{SKYNET_RUN_DIR}}/artifacts/preparation/openpi/pi05-libero/assets"
    )

    disabled = ManifestAdapter(manifest).resolve(
        make_spec(
            source={
                "repository": "https://github.com/Physical-Intelligence/openpi.git",
                "revision": COMMIT,
                "adapter": "openpi",
            },
            native={"config": {"config_name": "debug", "wandb_enabled": True}},
        )
    )
    assert "--no-wandb-enabled" in disabled.argv
    assert "--wandb-enabled" not in disabled.argv
    assert disabled.native_config["wandb_enabled"] is False
    assert len(plan.preparation_steps) == 1
    preparation = plan.preparation_steps[0]
    assert preparation.id == "openpi-pi05-libero-norm-stats"
    assert preparation.argv == [
        "curl",
        "--fail",
        "--location",
        "--create-dirs",
        "--output",
        "assets/pi05_libero/physical-intelligence/libero/norm_stats.json",
        "https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json",
    ]
    assert preparation.working_directory == "artifacts/preparation/openpi/pi05-libero"
    assert preparation.output_globs == [
        "assets/pi05_libero/physical-intelligence/libero/norm_stats.json"
    ]
    assert preparation.expected_output_sha256 == {
        "assets/pi05_libero/physical-intelligence/libero/norm_stats.json": (
            "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84"
        )
    }
    assert not any(argument.startswith("--lr-schedule") for argument in plan.argv)
    assert any(
        "does not map canonical value train.learning_rate" in blocker
        for blocker in plan.blockers
    )
    assert plan.runnable is False


def test_pinned_openpi_native_tracking_migration_is_explicit_and_audited(monkeypatch):
    import skynet_app.adapters as adapters
    from skynet_app.adapters import (
        AdapterPlan,
        ArgumentBinding,
        ManifestAdapter,
        NativeTrackingIntegration,
        apply_pinned_adapter_plan_compatibility,
        builtin_adapter_manifests,
        canonical_adapter_manifest,
    )
    from skynet_app.experiments import canonical_sha256

    current = next(
        manifest for manifest in builtin_adapter_manifests()
        if manifest.slug == "openpi"
    )
    flags = dict(current.train.parameter_flags)
    flags.pop("native.config.project_name")
    legacy = current.model_copy(
        update={
            "train": current.train.model_copy(
                update={"parameter_flags": flags, "native_tracking": []}, deep=True
            )
        },
        deep=True,
    )
    raw_manifest = canonical_adapter_manifest(legacy)
    digest = canonical_sha256(raw_manifest)
    revision = "215abfb217dbac7d5f1273282331b9b1866c0479"
    monkeypatch.setitem(
        adapters._VERSIONED_NATIVE_TRACKING_MIGRATIONS,
        digest,
        {
            "id": "test-openpi-native-wandb-v1",
            "adapter": "openpi",
            "source_revisions": {revision},
            "required_parameter_flags": {
                "native.config.wandb_enabled": ArgumentBinding(
                    flag="--wandb-enabled",
                    false_flag="--no-wandb-enabled",
                    style="boolean",
                )
            },
            "added_parameter_flags": {
                "native.config.project_name": ArgumentBinding(flag="--project-name")
            },
            "native_tracking": [
                NativeTrackingIntegration(
                    provider="wandb",
                    parameter_paths={
                        "enabled": "native.config.wandb_enabled",
                        "project": "native.config.project_name",
                    },
                    run_id_file=(
                        "{{tokens.run_dir}}/checkpoints/"
                        "{{native.config.config_name}}/{{identity.experiment}}/wandb_id.txt"
                    ),
                )
            ],
        },
    )
    spec = make_spec(
        source={
            "repository": "https://github.com/Physical-Intelligence/openpi.git",
            "revision": revision,
            "adapter": "openpi",
            "adapter_version": 104,
            "adapter_manifest": raw_manifest,
            "adapter_manifest_sha256": digest,
        },
        native={"config": {"config_name": "pi05_libero", "wandb_enabled": False}},
        tracking={
            "native_tracking": "preserve",
            "providers": [
                {
                    "provider": "wandb",
                    "enabled": True,
                    "entity": "team",
                    "project": "central-project",
                }
            ],
        },
    )
    pinned = ManifestAdapter(legacy, manifest_sha256=digest).resolve(spec)
    migrated, transformation = apply_pinned_adapter_plan_compatibility(spec, pinned)

    assert isinstance(migrated, AdapterPlan)
    assert transformation == {
        "kind": "versioned_adapter_native_tracking",
        "migration_id": "test-openpi-native-wandb-v1",
        "adapter": "openpi",
        "source_revision": revision,
        "source_manifest_sha256": digest,
        "providers": ["wandb"],
    }
    assert "--wandb-enabled" in migrated.argv
    assert "--no-wandb-enabled" not in migrated.argv
    project_index = migrated.argv.index("--project-name")
    assert migrated.argv[project_index + 1] == "central-project"
    assert migrated.native_tracking[0].run_id_file == (
        "{{SKYNET_RUN_DIR}}/checkpoints/pi05_libero/smoke/wandb_id.txt"
    )
    assert any("test-openpi-native-wandb-v1" in item for item in migrated.warnings)


def test_openpi_checkpoint_contract_is_atomic_and_adapter_owned():
    import pytest

    from skynet_app.adapters import (
        CommandTemplate,
        ManifestAdapter,
        builtin_adapter_manifests,
        get_adapter,
    )

    manifest = next(
        manifest for manifest in builtin_adapter_manifests() if manifest.slug == "openpi"
    )
    expected = {
        "checkpoint_globs": ["checkpoints/*/*/*"],
        "checkpoint_candidate_kind": "directory",
        "checkpoint_basename_regex": r"^[0-9]+$",
        "checkpoint_prune_globs": ["train_state"],
        "checkpoint_inference_required_globs": ["params", "assets"],
    }
    for field, value in expected.items():
        assert getattr(manifest.train, field) == value

    spec = make_spec(
        source={
            "repository": "https://github.com/Physical-Intelligence/openpi.git",
            "revision": COMMIT,
            "adapter": "openpi",
        },
        native={"config": {"config_name": "debug"}},
    )
    plans = [ManifestAdapter(manifest).resolve(spec), get_adapter("openpi").resolve(spec)]
    for plan in plans:
        for field, value in expected.items():
            assert getattr(plan, field) == value

    round_trip = type(manifest).model_validate(manifest.model_dump(mode="json"))
    for field, value in expected.items():
        assert getattr(round_trip.train, field) == value

    with pytest.raises(ValueError, match="relative POSIX"):
        CommandTemplate(checkpoint_globs=["/outside/checkpoint"])
    with pytest.raises(ValueError, match="must not escape"):
        CommandTemplate(checkpoint_prune_globs=["../outside"])
    with pytest.raises(ValueError, match="invalid checkpoint basename regex"):
        CommandTemplate(checkpoint_basename_regex="(")


def test_manifest_gpu_recommendations_are_ordered_and_only_apply_to_auto_mode():
    from skynet_app.adapters import (
        AdapterGPURecommendation,
        AdapterResourceDefaults,
        ManifestAdapter,
        builtin_adapter_manifests,
        resolve_gpu_count,
    )

    base = next(manifest for manifest in builtin_adapter_manifests() if manifest.slug == "generic")
    resources = AdapterResourceDefaults(
        gpu_recommendations=[
            AdapterGPURecommendation(
                id="first-match",
                enabled_when={"native.config.workload": ["large"]},
                minimum_gpus=3,
                recommended_gpus=4,
                maximum_gpus=6,
                gpu_type="l40s",
            ),
            AdapterGPURecommendation(
                id="later-match",
                enabled_when={"native.config.workload": ["large"]},
                minimum_gpus=2,
                recommended_gpus=2,
                maximum_gpus=8,
            ),
        ]
    )
    manifest = base.model_copy(
        update={
            "defaults": base.defaults.model_copy(
                update={"resources": resources},
                deep=True,
            )
        },
        deep=True,
    )
    auto_spec = make_spec(
        resources={
            "account": "rl2-lab",
            "partition": "rl2-lab",
            "gpu": {"mode": "auto", "profile": "recommended", "type": "any"},
        },
        native={
            "argv": ["python", "train.py"],
            "resume_argv": ["--resume", "{{SKYNET_RESUME_CHECKPOINT}}"],
            "config": {"workload": "large"},
        },
    )
    auto_plan = ManifestAdapter(manifest).resolve(auto_spec)

    assert auto_plan.capabilities.minimum_gpus == 3
    assert auto_plan.capabilities.recommended_gpus == 4
    assert auto_plan.capabilities.maximum_gpus == 6
    assert auto_plan.resolved_gpu_type == "l40s"
    assert resolve_gpu_count(auto_spec, auto_plan) == 4

    explicit_spec = make_spec(
        resources={
            "account": "rl2-lab",
            "partition": "rl2-lab",
            "gpu": {"mode": "explicit", "count": 1, "type": "a40"},
        },
        native={
            "argv": ["python", "train.py"],
            "resume_argv": ["--resume", "{{SKYNET_RESUME_CHECKPOINT}}"],
            "config": {"workload": "large"},
        },
    )
    explicit_plan = ManifestAdapter(manifest).resolve(explicit_spec)

    assert explicit_plan.capabilities.recommended_gpus == 1
    assert resolve_gpu_count(explicit_spec, explicit_plan) == 1


def test_openpi_pi05_libero_declares_conditional_gpu_profiles():
    from skynet_app.adapters import ManifestAdapter, builtin_adapter_manifests, resolve_gpu_count

    manifest = next(manifest for manifest in builtin_adapter_manifests() if manifest.slug == "openpi")
    recommendation = manifest.defaults.resources.gpu_recommendations[0]
    assert recommendation.id == "pi05-libero"
    assert recommendation.enabled_when == {"native.config.config_name": ["pi05_libero"]}
    assert recommendation.gpu_type == "l40s"
    assert (
        recommendation.minimum_gpus,
        recommendation.recommended_gpus,
        recommendation.maximum_gpus,
    ) == (2, 2, 8)

    def plan_for(config_name: str, gpu: dict):
        spec = make_spec(
            source={
                "repository": "https://github.com/Physical-Intelligence/openpi.git",
                "revision": COMMIT,
                "adapter": "openpi",
            },
            resources={
                "account": "rl2-lab",
                "partition": "rl2-lab",
                "gpu": {"type": "any", **gpu},
            },
            train={
                "batch": {
                    "declared_semantics": "global_before_accumulation",
                    "value": 32,
                    "gradient_accumulation_steps": 1,
                }
            },
            native={"config": {"config_name": config_name, "wandb_enabled": False}},
        )
        plan = ManifestAdapter(manifest).resolve(spec)
        return spec, plan

    auto_spec, auto_plan = plan_for(
        "pi05_libero", {"mode": "auto", "profile": "recommended"}
    )
    assert resolve_gpu_count(auto_spec, auto_plan) == 2
    assert auto_plan.capabilities.minimum_gpus == 2
    assert auto_plan.resolved_gpu_type == "l40s"
    auto_fsdp = auto_plan.argv.index("--fsdp-devices")
    assert auto_plan.argv[auto_fsdp + 1] == "2"
    from skynet_app.slurm import compile_sbatch

    compiled = compile_sbatch(auto_spec, auto_plan, run_id="openpi-auto-gpu")
    assert compiled.gpu_type == "l40s"
    assert "#SBATCH --gres=gpu:l40s:2" in compiled.script

    fallback_spec, fallback_plan = plan_for(
        "debug", {"mode": "auto", "profile": "recommended"}
    )
    assert resolve_gpu_count(fallback_spec, fallback_plan) == 1

    explicit_spec, explicit_plan = plan_for(
        "pi05_libero", {"mode": "explicit", "count": 1, "type": "l40s"}
    )
    assert resolve_gpu_count(explicit_spec, explicit_plan) == 1
    assert explicit_plan.capabilities.minimum_gpus == 1
    assert explicit_plan.resolved_gpu_type == "l40s"
    explicit_fsdp = explicit_plan.argv.index("--fsdp-devices")
    assert explicit_plan.argv[explicit_fsdp + 1] == "1"


def test_openpi_optional_ema_decay_maps_none_without_changing_upstream_default():
    from skynet_app.adapters import ManifestAdapter, builtin_adapter_manifests

    manifests = {manifest.slug: manifest for manifest in builtin_adapter_manifests()}
    openpi = manifests["openpi"]
    ema_field = next(
        field for field in openpi.train.input_fields
        if field.path == "native.config.ema_decay"
    )
    assert ema_field.choices == ["0.999", "None"]
    assert ema_field.default is None
    assert "None disables EMA" in ema_field.help
    assert all(
        field.path != "native.config.ema_decay"
        for field in manifests["generic"].train.input_fields
    )

    def plan_for(config: dict):
        return ManifestAdapter(openpi).resolve(
            make_spec(
                source={
                    "repository": "https://github.com/Physical-Intelligence/openpi.git",
                    "revision": COMMIT,
                    "adapter": "openpi",
                },
                native={"config": {"config_name": "debug", **config}},
            )
        )

    disabled = plan_for({"ema_decay": "None"})
    position = disabled.argv.index("--ema-decay")
    assert disabled.argv[position + 1] == "None"

    upstream_default = plan_for({})
    assert "--ema-decay" not in upstream_default.argv


def test_preview_retains_structured_openpi_blockers():
    from skynet_app.pipeline_api import PipelineService

    source = {
        "repository": "https://github.com/Physical-Intelligence/openpi.git",
        "revision": COMMIT,
        "adapter": "openpi",
    }
    spec = make_spec(
        source=source,
        train={"batch": {"value": 12, "gradient_accumulation_steps": 1}},
        native={"config": {}},
    )
    service = PipelineService.__new__(PipelineService)
    service.normalize_spec = lambda _payload: spec

    preview = service.preview({})

    assert preview["variant_count"] == 1
    assert len(preview["scripts"]) == 1
    assert preview["script"].startswith("# BLOCKED:")
    assert len(preview["blockers"]) == 1
    assert preview["blockers"][0]["variant"]
    assert "openpi requires native.config.config_name" in preview["blockers"][0]["reasons"]


def test_typed_adapter_input_fields_validate_and_align_required_values():
    import pytest

    from skynet_app.adapters import AdapterInputField, CommandTemplate

    command = CommandTemplate(
        input_fields=[
            AdapterInputField(
                path="native.config.config_name",
                label="Config",
                kind="string",
                required=True,
                choices=["debug", "production"],
                tutorial_value="debug",
            )
        ]
    )
    assert command.required_values == ["native.config.config_name"]

    with pytest.raises(ValueError, match="must match input field kind"):
        AdapterInputField(
            path="native.config.config_name",
            label="Config",
            kind="string",
            default=3,
        )
    with pytest.raises(ValueError, match="paths must be unique"):
        CommandTemplate(input_fields=[command.input_fields[0], command.input_fields[0]])


def test_builtin_manifests_declare_native_inputs_and_manifest_owned_tutorial_value():
    from skynet_app.adapters import builtin_adapter_manifests

    manifests = {manifest.slug: manifest for manifest in builtin_adapter_manifests()}
    paths = {
        slug: {field.path for field in manifest.train.input_fields}
        for slug, manifest in manifests.items()
    }

    openpi = next(
        field
        for field in manifests["openpi"].train.input_fields
        if field.path == "native.config.config_name"
    )
    assert openpi.required is True
    assert openpi.default is None
    assert openpi.choices == []
    assert openpi.choice_source is not None
    assert openpi.choice_source.kind == "python_static_registry"
    assert openpi.choice_source.entrypoint == "src/openpi/training/config.py"
    assert openpi.choice_source.registry == "_CONFIGS"
    assert openpi.choice_source.constructor == "TrainConfig"
    assert openpi.tutorial_value == "debug"
    assert "native.config.config_name" in manifests["openpi"].train.required_values
    assert {"native.argv", "native.resume_argv"} <= paths["generic"]
    assert {"native.argv", "native.resume_argv"} <= paths["dexverse"]
    assert {
        "native.config.training_config",
        "native.config.robomimic_revision",
    } <= paths["dexmimicgen"]
    assert {"native.config.mode", "native.config.override_scopes"} <= paths["get_zero"]
    assert {
        "native.config.dataset_path",
        "native.config.base_model_path",
        "native.config.embodiment_tag",
        "native.config.modality_config_path",
    } <= paths["groot"]
    assert "native.config.config_name" in paths["egoverse"]


def test_manifest_input_default_applies_without_overriding_user_value():
    from skynet_app.adapters import builtin_adapter_manifests
    from skynet_app.pipeline_api import PipelineService

    manifest = next(
        item for item in builtin_adapter_manifests() if item.slug == "egoverse"
    )
    missing: dict[str, object] = {"native": {"config": {}}}
    selected: dict[str, object] = {
        "native": {"config": {"config_name": "user-selected"}}
    }

    PipelineService._apply_manifest_input_defaults(missing, manifest)
    PipelineService._apply_manifest_input_defaults(selected, manifest)

    assert missing["native"]["config"]["config_name"] == "train_zarr_cartesian"
    assert selected["native"]["config"]["config_name"] == "user-selected"


def test_native_text_supports_deep_config_and_flat_dotted_overrides():
    import pytest

    from skynet_app.pipeline_api import _parse_native

    config, overrides, _, _ = _parse_native(
        ["config.optimizer.schedule.warmup=100", 'model.encoder.name="vit"']
    )

    assert config == {"optimizer": {"schedule": {"warmup": 100}}}
    assert overrides == {"model.encoder.name": "vit"}
    with pytest.raises(ValueError, match="native config path collision"):
        _parse_native(["config.optimizer=1", "config.optimizer.schedule=2"])
    with pytest.raises(ValueError, match="native config path collision"):
        _parse_native(["config.optimizer.schedule=2", "config.optimizer=1"])
    with pytest.raises(ValueError, match="invalid native config path"):
        _parse_native(["config.__proto__.polluted=true"])


def test_manifest_input_paths_are_native_only_and_prototype_safe():
    import pytest

    from skynet_app.adapters import AdapterInputField

    for path in (
        "train.batch.value",
        "native.config.__proto__.polluted",
        "native.overrides",
        "native.overrides.constructor.polluted",
    ):
        with pytest.raises(ValueError):
            AdapterInputField(path=path, label="Invalid", kind="json")

    field = AdapterInputField(
        path="native.overrides.model.encoder-name",
        label="Encoder override",
        kind="string",
    )
    assert field.path == "native.overrides.model.encoder-name"


def test_sensitive_input_is_presentation_only_and_cannot_embed_values():
    import pytest

    from skynet_app.adapters import AdapterInputField

    masked = AdapterInputField(
        path="native.config.access_token",
        label="Token",
        kind="string",
        sensitive=True,
    )
    assert masked.model_dump(mode="json")["sensitive"] is True
    schema = AdapterInputField.model_json_schema()
    assert schema["properties"]["sensitive"]["deprecated"] is True
    assert "does not provide secret storage" in schema["properties"]["sensitive"]["description"]

    for payload in (
        {"default": "secret"},
        {"tutorial_value": "secret"},
        {"choices": ["secret"]},
    ):
        with pytest.raises(ValueError, match="does not provide secret storage"):
            AdapterInputField(
                path="native.config.access_token",
                label="Token",
                kind="string",
                sensitive=True,
                **payload,
            )


def test_direct_canonical_normalization_enforces_required_kind_and_choices():
    import copy
    import pytest

    from skynet_app.adapters import AdapterInputField, CommandTemplate, builtin_adapter_manifests
    from skynet_app.pipeline_api import PipelineService

    base_manifest = next(
        item for item in builtin_adapter_manifests() if item.slug == "openpi"
    )
    manifest = base_manifest.model_copy(
        update={
            "train": CommandTemplate(
                input_fields=[
                    AdapterInputField(
                        path="native.config.worker_count",
                        label="Worker count",
                        kind="integer",
                        required=True,
                        choices=[1, 2],
                    )
                ]
            )
        },
        deep=True,
    )
    source = {
        "repository": "https://github.com/Physical-Intelligence/openpi.git",
        "revision": COMMIT,
        "adapter": "openpi",
    }
    payload = make_spec(
        source=source,
        train={"batch": {"value": 12, "gradient_accumulation_steps": 1}},
        native={"config": {}},
    ).model_dump(mode="json", by_alias=True)
    payload["source"]["adapter_manifest"] = manifest.model_dump(mode="json")
    payload["source"].pop("adapter_manifest_sha256", None)
    service = PipelineService.__new__(PipelineService)
    service._resolve_source_and_runtime = (
        lambda resolved_source, runtime, _manifest, _gateway: (resolved_source, runtime)
    )

    with pytest.raises(ValueError, match="native.config.worker_count: required"):
        service.normalize_spec(copy.deepcopy(payload))

    boolean_payload = copy.deepcopy(payload)
    boolean_payload["native"]["config"]["worker_count"] = True
    with pytest.raises(ValueError, match="native.config.worker_count: expected integer, got bool"):
        service.normalize_spec(boolean_payload)

    choice_payload = copy.deepcopy(payload)
    choice_payload["native"]["config"]["worker_count"] = 3
    with pytest.raises(ValueError, match="native.config.worker_count: value is not one"):
        service.normalize_spec(choice_payload)

    valid_payload = copy.deepcopy(payload)
    valid_payload["native"]["config"]["worker_count"] = 2
    normalized = service.normalize_spec(valid_payload)
    assert normalized.native.config["worker_count"] == 2


def test_number_input_rejects_boolean_for_direct_callers():
    import pytest

    from skynet_app.adapters import AdapterInputField, CommandTemplate, builtin_adapter_manifests
    from skynet_app.pipeline_api import PipelineService

    manifest = next(
        item for item in builtin_adapter_manifests() if item.slug == "openpi"
    ).model_copy(
        update={
            "train": CommandTemplate(
                input_fields=[
                    AdapterInputField(
                        path="native.config.temperature",
                        label="Temperature",
                        kind="number",
                        required=True,
                    )
                ]
            )
        },
        deep=True,
    )
    canonical = {"native": {"config": {"temperature": True}}}

    with pytest.raises(ValueError, match="native.config.temperature: expected number, got bool"):
        PipelineService._validate_manifest_input_fields(canonical, manifest)


def test_eval_catalog_and_canonical_result():
    assert any(item.suite == "libero_10" for item in get_evaluation_catalog())
    result = build_canonical_result(
        run_id="run-1",
        checkpoint=CheckpointReference(path="model.safetensors", sha256="b" * 64),
        evaluator=EvaluatorReference(adapter="libero", version="1"),
        environment=EnvironmentReference(suite="libero_10", version="1"),
        episodes=[
            EpisodeResult(task="task-a", seed=42, episode_index=0, success=True, reward=2.0),
            EpisodeResult(task="task-a", seed=42, episode_index=1, success=False, reward=0.0),
        ],
    )
    success = next(metric for metric in result.aggregate if metric.metric == "success_rate" and metric.task is None)
    assert success.mean == 0.5
    assert success.sample_count == 2


def test_openpi_selected_dataset_bridge_and_normalization_are_explicit():
    from skynet_app.adapters import ManifestAdapter, builtin_adapter_manifests
    manifest = next(item for item in builtin_adapter_manifests() if item.slug == "openpi")
    def selected(**config):
        return ManifestAdapter(manifest).resolve(make_spec(
            source={"repository": "https://github.com/Physical-Intelligence/openpi.git", "revision": COMMIT, "adapter": "openpi"},
            native={"config": {"config_name": "pi05_libero", **config}},
        ))
    original = selected()
    assert original.argv[1] == "scripts/train.py"
    assert any(step.id == "openpi-pi05-libero-norm-stats" for step in original.preparation_steps)
    missing = selected(dataset_path="/datasets/selected")
    assert any("Dataset normalization file is required" in message for message in missing.blockers)
    explicit = selected(dataset_path="/datasets/selected", dataset_norm_stats_path="/stats/selected/norm_stats.json")
    assert explicit.argv[1].endswith("/adapter-support/openpi-dataset.py")
    assert explicit.argv[explicit.argv.index("--dataset-root") + 1] == "/datasets/selected"
    assert explicit.argv[explicit.argv.index("--norm-stats") + 1] == "/stats/selected/norm_stats.json"
    assert not explicit.preparation_steps
    assert "adapter-support/openpi-dataset.py" in explicit.capsule_files
    assert not any("dataset" in message.lower() for message in explicit.blockers)


def test_old_openpi_manifest_cannot_silently_inherit_dataset_bridge():
    from skynet_app.adapters import ManifestAdapter, builtin_adapter_manifests
    manifest = next(item for item in builtin_adapter_manifests() if item.slug == "openpi").model_copy(deep=True)
    manifest.train.capsule_files.pop("adapter-support/openpi-dataset.py")
    plan = ManifestAdapter(manifest).resolve(make_spec(
        source={"repository":"https://github.com/Physical-Intelligence/openpi.git", "revision":COMMIT,"adapter":"openpi"},
        native={"config":{"config_name":"pi05_libero","dataset_path":"/selected","dataset_norm_stats_path":"/stats/norm_stats.json"}},
    ))
    assert any("unsupported by this pinned OpenPI adapter" in message for message in plan.blockers)
