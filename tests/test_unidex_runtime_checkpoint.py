"""Real CPU Lightning checkpoint/resume coverage; no UniDex weights or GPU."""

import json

import pytest

torch = pytest.importorskip("torch")
lightning = pytest.importorskip("pytorch_lightning")
pytest.importorskip("hydra")

from pytorch_lightning.callbacks import ModelCheckpoint

from skynet_app.adapters.unidex_data import UniDexMixtureSampler, make_training_dataloader
from skynet_app.adapters.unidex_runtime import (
    FORMAT,
    RUN_SCHEMA,
    make_training_wrapper,
    make_training_checkpoint,
    resume_initialization,
    training_strategy,
    validate_resume,
)


class ConstantSchedule:
    def __call__(self, step):
        return 1.0


class TinyRecordedDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, index):
        state = torch.tensor([[index / 10]], dtype=torch.float32)
        return {"state": state, "action": state * 0.3, "index": index}


class TinyNativePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.seen = []

    def forward(self, batch):
        self.seen.extend(batch["index"].tolist())
        loss = torch.nn.functional.mse_loss(self.linear(batch["state"]), batch["action"])
        return loss, {}

    def infer_action(self, batch):
        # The original UniDex method consumes this entry in place.
        batch.pop("action")
        return self.linear(batch["state"])


class TinyMLP(torch.nn.Sequential):
    pass


class TinyVisualBlock(torch.nn.Sequential):
    pass


class TinyDecoderContainer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinyMLP(torch.nn.Linear(2, 2))
        self.self_attn = torch.nn.Linear(2, 2)


def structured_policy():
    """Mirror the upstream aliases and forward-called versus custom-method units."""
    policy = torch.nn.Module()
    policy.embed_tokens = torch.nn.Embedding(4, 2)
    policy.joint = torch.nn.Module()
    policy.joint.mixtures = torch.nn.ModuleDict()
    for name in ("vlm", "proprio", "action"):
        mixture = torch.nn.Module()
        mixture.layers = torch.nn.ModuleList([TinyDecoderContainer()])
        policy.joint.mixtures[name] = mixture
    policy.joint_model = policy.joint
    policy.pointcloud_encoder = torch.nn.Module()
    policy.pointcloud_encoder.visual = torch.nn.Module()
    policy.pointcloud_encoder.visual.blocks = torch.nn.ModuleList([
        TinyVisualBlock(torch.nn.Linear(2, 2)),
    ])
    return policy


class SimulatedPreemption(RuntimeError):
    pass


class PreemptAfterFirstUpdate(lightning.Callback):
    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        if trainer.global_step == 1:
            raise SimulatedPreemption("Stop after the completed optimizer checkpoint")


def training_parts(*, max_steps=6, initialization=None):
    lightning.seed_everything(42, workers=True)
    policy = TinyNativePolicy()
    sampler = UniDexMixtureSampler(["wuji2"] * 10, seed=42)
    loader = make_training_dataloader(
        TinyRecordedDataset(), sampler, batch_size=2, num_workers=0, seed=42
    )
    identity = {"schema": RUN_SCHEMA, "dataset_format": FORMAT, "budget": {"max_steps": max_steps}}
    training = {
        "optimizer": {"_target_": "torch.optim.AdamW", "lr": 0.01, "eps": 1e-8},
        "scheduler": {"_target_": __name__ + ".ConstantSchedule"},
    }
    wrapper = make_training_wrapper(
        policy, training, identity,
        {"mode": "test"} if initialization is None else initialization, sampler,
    )
    return wrapper, loader, sampler, identity


def assert_exact(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_exact(a, b)
    else:
        assert left == right


def cpu_trainer(path, *, preempt=False, native_checkpoint=False, max_steps=6):
    checkpoint_options = (
        {"monitor": "val_loss", "mode": "min", "save_top_k": 1}
        if native_checkpoint else {"every_n_train_steps": 1, "save_top_k": -1}
    )
    checkpoint = ModelCheckpoint(
        dirpath=str(path / "checkpoints"), save_last=True, **checkpoint_options,
    )
    callbacks = [checkpoint]
    if preempt:
        callbacks.append(PreemptAfterFirstUpdate())
    return lightning.Trainer(
        accelerator="cpu", devices=1, max_steps=max_steps, max_epochs=-1,
        accumulate_grad_batches=2, precision="32-true", deterministic=True,
        gradient_clip_val=1.0, gradient_clip_algorithm="norm",
        logger=False, enable_progress_bar=False, enable_model_summary=False,
        num_sanity_val_steps=0, callbacks=callbacks, default_root_dir=str(path),
    ), checkpoint


def test_real_lightning_resume_preserves_completed_samples_and_optimizer(tmp_path):
    initialization = {"mode": "test", "checkpoint_sha256": "a" * 64}
    full, full_loader, _, identity = training_parts(initialization=initialization)
    trainer, _ = cpu_trainer(tmp_path / "full")
    trainer.fit(full, full_loader)
    assert trainer.global_step == 6

    interrupted, interrupted_loader, _, _ = training_parts(initialization=initialization)
    trainer, checkpoint = cpu_trainer(tmp_path / "interrupted", preempt=True)
    with pytest.raises(SimulatedPreemption):
        trainer.fit(interrupted, interrupted_loader)
    saved = torch.load(checkpoint.last_model_path, map_location="cpu", weights_only=False)
    validate_resume(saved, identity)
    assert saved["global_step"] == 1
    loader_state = saved["loops"]["fit_loop"]["state_dict"]["combined_loader"][0]
    assert loader_state["sampler"]["consumed_samples"] == 4
    del saved

    resumed_initialization = resume_initialization(
        checkpoint.last_model_path, identity, initialization_sha=initialization["checkpoint_sha256"],
    )
    assert resumed_initialization == initialization
    with pytest.raises(ValueError, match="Resume initialization differs"):
        resume_initialization(checkpoint.last_model_path, identity, initialization_sha="b" * 64)
    resumed, resumed_loader, resumed_sampler, _ = training_parts(initialization=resumed_initialization)
    trainer, _ = cpu_trainer(tmp_path / "resumed")
    trainer.fit(resumed, resumed_loader, ckpt_path=checkpoint.last_model_path)
    assert trainer.global_step == 6
    assert interrupted.policy.seen + resumed.policy.seen == full.policy.seen
    assert resumed_sampler.consumed_samples == 10
    for name, value in full.policy.state_dict().items():
        torch.testing.assert_close(resumed.policy.state_dict()[name], value, rtol=0, atol=0)
    rows = [json.loads(line) for line in (tmp_path / "resumed" / "logs.json.txt").read_text().splitlines()]
    updates = [row for row in rows if "epoch" not in row]
    assert [row["global_step"] for row in updates] == [2, 3, 4, 5, 6]
    assert all(torch.isfinite(torch.tensor(row["train_loss"])) for row in updates)


def test_training_checkpoint_resume_does_not_save_partial_accumulation(tmp_path):
    """The first resumed microbatch must not rewrite its restored global step."""
    from pathlib import Path

    def train(path, wrapper, loader, *, resume=None, preempt=False):
        checkpoint = make_training_checkpoint(path, save_every_steps=1, keep_last=10)
        callbacks = [checkpoint, *([PreemptAfterFirstUpdate()] if preempt else [])]
        trainer = lightning.Trainer(
            accelerator="cpu", devices=1, max_steps=6, max_epochs=-1,
            accumulate_grad_batches=2, precision="32-true", deterministic=True,
            gradient_clip_val=1.0, gradient_clip_algorithm="norm",
            logger=False, enable_progress_bar=False, enable_model_summary=False,
            num_sanity_val_steps=0, callbacks=callbacks, default_root_dir=str(path),
        )
        if preempt:
            with pytest.raises(SimulatedPreemption):
                trainer.fit(wrapper, loader)
        else:
            trainer.fit(wrapper, loader, ckpt_path=resume)
        return trainer, checkpoint

    full, full_loader, full_sampler, identity = training_parts()
    _, full_checkpoint = train(tmp_path / "full", full, full_loader)
    assert [path.name for path in sorted((tmp_path / "full/checkpoints").glob("step-*.ckpt"))] == [
        f"step-{step:08d}.ckpt" for step in range(1, 7)
    ]
    interrupted, interrupted_loader, _, _ = training_parts()
    resumed_path = tmp_path / "resumed"
    _, checkpoint = train(resumed_path, interrupted, interrupted_loader, preempt=True)
    boundary = Path(checkpoint.last_model_path).resolve()
    boundary_bytes = boundary.read_bytes()
    saved = torch.load(boundary, map_location="cpu", weights_only=False)
    validate_resume(saved, identity)
    assert saved["global_step"] == 1
    assert saved["loops"]["fit_loop"]["state_dict"]["combined_loader"][0]["sampler"]["consumed_samples"] == 4

    resumed, resumed_loader, resumed_sampler, _ = training_parts()
    trainer, final_checkpoint = train(resumed_path, resumed, resumed_loader, resume=str(boundary))
    assert trainer.global_step == 6
    assert boundary.read_bytes() == boundary_bytes
    assert not list(boundary.parent.glob("step-*-v*.ckpt"))
    checkpoints = sorted(boundary.parent.glob("step-*.ckpt"))
    assert [path.name for path in checkpoints] == [f"step-{step:08d}.ckpt" for step in range(1, 7)]
    assert interrupted.policy.seen + resumed.policy.seen == full.policy.seen
    assert resumed_sampler.state_dict() == full_sampler.state_dict()

    expected = torch.load(full_checkpoint.last_model_path, map_location="cpu", weights_only=False)
    actual = torch.load(final_checkpoint.last_model_path, map_location="cpu", weights_only=False)

    for key in ("state_dict", "optimizer_states", "lr_schedulers", "global_step"):
        assert_exact(expected[key], actual[key])
    assert_exact(expected["loops"]["fit_loop"]["state_dict"]["combined_loader"],
                 actual["loops"]["fit_loop"]["state_dict"]["combined_loader"])


@pytest.mark.parametrize("old_interval", [1, 2])
def test_checkpoint_cadence_change_preserves_full_resume_state(tmp_path, old_interval):
    """New revision output/callback identity must not reset any training state."""
    from pathlib import Path

    class PreemptAtBoundary(lightning.Callback):
        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            if trainer.global_step == old_interval:
                raise SimulatedPreemption("Saved old-cadence optimizer boundary")

    def train(path, wrapper, loader, *, interval, resume=None, preempt=False):
        path.mkdir(parents=True)
        checkpoint = make_training_checkpoint(path, save_every_steps=interval, keep_last=3)
        callbacks = [checkpoint, *([PreemptAtBoundary()] if preempt else [])]
        trainer = lightning.Trainer(
            accelerator="cpu", devices=1, max_steps=12, max_epochs=-1,
            accumulate_grad_batches=2, precision="32-true", deterministic=True,
            gradient_clip_val=1.0, gradient_clip_algorithm="norm",
            logger=False, enable_progress_bar=False, enable_model_summary=False,
            num_sanity_val_steps=0, callbacks=callbacks, default_root_dir=str(path),
        )
        if preempt:
            with pytest.raises(SimulatedPreemption):
                trainer.fit(wrapper, loader)
        else:
            trainer.fit(wrapper, loader, ckpt_path=resume)
        return trainer, checkpoint

    full, full_loader, full_sampler, identity = training_parts(max_steps=12)
    _, full_checkpoint = train(tmp_path / "full", full, full_loader, interval=4)
    interrupted, interrupted_loader, _, _ = training_parts(max_steps=12)
    old_output = tmp_path / "old-revision"
    _, old_checkpoint = train(
        old_output, interrupted, interrupted_loader, interval=old_interval, preempt=True,
    )
    source = Path(old_checkpoint.last_model_path).resolve()
    source_bytes = source.read_bytes()
    saved = torch.load(source, map_location="cpu", weights_only=False)
    validate_resume(saved, identity)
    assert saved["global_step"] == old_interval
    assert old_checkpoint.state_key in saved["callbacks"]
    old_files = sorted(path.name for path in source.parent.iterdir())

    initialization = resume_initialization(source, identity)
    resumed, resumed_loader, resumed_sampler, _ = training_parts(
        max_steps=12, initialization=initialization,
    )
    new_output = tmp_path / "new-revision"
    trainer, new_checkpoint = train(
        new_output, resumed, resumed_loader, interval=4, resume=str(source),
    )
    assert new_checkpoint.state_key != old_checkpoint.state_key
    assert trainer.global_step == 12
    assert source.read_bytes() == source_bytes
    assert sorted(path.name for path in source.parent.iterdir()) == old_files
    retained = sorted((new_output / "checkpoints").glob("step-*.ckpt"))
    assert [path.name for path in retained] == [
        f"step-{step:08d}.ckpt" for step in (4, 8, 12)
    ]
    for path, step in zip(retained, (4, 8, 12)):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert checkpoint["global_step"] == step
        assert checkpoint["lr_schedulers"][0]["last_epoch"] == step
        assert all(int(state["step"]) == step
                   for state in checkpoint["optimizer_states"][0]["state"].values())
    assert interrupted.policy.seen + resumed.policy.seen == full.policy.seen
    assert resumed_sampler.state_dict() == full_sampler.state_dict()

    expected = torch.load(full_checkpoint.last_model_path, map_location="cpu", weights_only=False)
    actual = torch.load(new_checkpoint.last_model_path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "optimizer_states", "lr_schedulers", "global_step", "skynet"):
        assert_exact(expected[key], actual[key])
    assert_exact(expected["loops"]["fit_loop"]["state_dict"]["combined_loader"],
                 actual["loops"]["fit_loop"]["state_dict"]["combined_loader"])
    updates = [json.loads(line) for line in (new_output / "logs.json.txt").read_text().splitlines()]
    assert [row["global_step"] for row in updates if "epoch" not in row] == list(
        range(old_interval + 1, 13)
    )


def test_real_validation_and_epoch_checkpoint_include_resume_state(tmp_path):
    wrapper, loader, sampler, identity = training_parts()
    validation = torch.utils.data.DataLoader(TinyRecordedDataset(), batch_size=2)
    trainer, checkpoint = cpu_trainer(tmp_path, native_checkpoint=True)
    trainer.fit(wrapper, loader, validation)
    assert trainer.global_step == 6
    assert sampler.consumed_samples == 10
    assert checkpoint.best_model_path and checkpoint.last_model_path
    saved = torch.load(checkpoint.last_model_path, map_location="cpu", weights_only=False)
    validate_resume(saved, identity)
    rows = [json.loads(line) for line in (tmp_path / "logs.json.txt").read_text().splitlines()]
    epochs = [row for row in rows if "epoch" in row]
    assert [row["epoch"] for row in epochs] == [0, 1]
    assert all(torch.isfinite(torch.tensor(row["val_loss"])) for row in epochs)


@pytest.mark.parametrize("max_steps, retained_steps", [
    (4, [3, 4]),  # A partial second epoch with a previous validation checkpoint.
    (6, [3, 6]),  # Finishing exactly at the checkpoint/epoch boundary.
    (7, [6, 7]),  # Retention must also include the final partial epoch.
])
def test_training_checkpoint_saves_final_update_and_retains_latest_steps(
    tmp_path, max_steps, retained_steps,
):
    from pathlib import Path

    from skynet_app.adapters.unidex_runtime import make_training_checkpoint

    wrapper, loader, sampler, identity = training_parts(max_steps=max_steps)
    validation = torch.utils.data.DataLoader(TinyRecordedDataset(), batch_size=2)
    checkpoint = make_training_checkpoint(tmp_path, save_every_steps=3, keep_last=2)
    trainer = lightning.Trainer(
        accelerator="cpu", devices=1, max_steps=max_steps, max_epochs=-1,
        accumulate_grad_batches=2, precision="32-true", deterministic=True,
        gradient_clip_val=1.0, gradient_clip_algorithm="norm",
        logger=False, enable_progress_bar=False, enable_model_summary=False,
        num_sanity_val_steps=0, callbacks=[checkpoint], default_root_dir=str(tmp_path),
    )
    trainer.fit(wrapper, loader, validation)
    assert trainer.global_step == max_steps

    latest = Path(checkpoint.last_model_path)
    assert latest.is_symlink()
    assert latest.resolve() == Path(checkpoint.best_model_path).resolve()
    retained = sorted(latest.parent.glob("step-*.ckpt"))
    assert [int(path.stem.removeprefix("step-")) for path in retained] == retained_steps
    for path, expected_step in zip(retained, retained_steps):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        validate_resume(saved, identity)
        assert saved["global_step"] == expected_step
        assert saved["lr_schedulers"][0]["last_epoch"] == expected_step
        assert all(int(state["step"]) == expected_step
                   for state in saved["optimizer_states"][0]["state"].values())

    saved = torch.load(latest, map_location="cpu", weights_only=False)
    assert saved["global_step"] == max_steps
    for name, value in wrapper.policy.state_dict().items():
        torch.testing.assert_close(saved["state_dict"]["policy." + name], value, rtol=0, atol=0)
    loader_state = saved["loops"]["fit_loop"]["state_dict"]["combined_loader"][0]
    assert loader_state["sampler"] == sampler.state_dict()
    assert sampler.consumed_samples == (10 if max_steps % 3 == 0 else 4)
    rows = [json.loads(line) for line in (tmp_path / "logs.json.txt").read_text().splitlines()]
    final_epoch = [row for row in rows if "epoch" in row][-1]
    assert final_epoch["global_step"] == max_steps
    if max_steps % 3:
        assert final_epoch["val_loss"] is None
        assert "val_loss" not in final_epoch["metrics"]
    else:
        assert torch.isfinite(torch.tensor(final_epoch["val_loss"]))


def test_fsdp_strategy_preserves_full_checkpoint_and_original_parameters():
    from pytorch_lightning.strategies import FSDPStrategy

    policy = structured_policy()
    strategy = training_strategy("fsdp", 4, policy)
    assert isinstance(strategy, FSDPStrategy)
    assert strategy._state_dict_type == "full"
    assert strategy.kwargs["use_orig_params"] is True
    wrap = strategy.kwargs["auto_wrap_policy"]
    selected = {
        module for module in policy.modules()
        if wrap(module=module, recurse=False, nonwrapped_numel=0)
    }
    expected = {policy.embed_tokens, *policy.pointcloud_encoder.visual.blocks}
    expected.update(layer.mlp for mixture in policy.joint_model.mixtures.values()
                    for layer in mixture.layers)
    assert selected == expected
    assert policy.joint not in selected
    assert all(layer not in selected and layer.self_attn not in selected
               for mixture in policy.joint_model.mixtures.values() for layer in mixture.layers)
    assert training_strategy("ddp", 1) == "auto"
    assert training_strategy("ddp", 4) == "ddp_find_unused_parameters_true"


@pytest.mark.parametrize("name, count, error", [
    ("fsdp", 1, "at least two GPUs"),
    ("fsdp", 0, "at least two GPUs"),
    ("fsdp", 4, "initialized native policy"),
    ("unsupported", 4, "Unknown distributed training strategy"),
])
def test_invalid_distributed_strategy_fails_before_allocating_gpu(name, count, error):
    with pytest.raises(ValueError, match=error):
        training_strategy(name, count)


@pytest.mark.parametrize("algorithm", [None, "norm"])
def test_gradient_clipping_override_preserves_single_device_l2_update(tmp_path, algorithm):
    policy = TinyNativePolicy()
    policy.linear = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        policy.linear.weight.zero_()
    dataset = [{"state": torch.tensor([[2.0]]), "action": torch.tensor([[10.0]]), "index": 0}]
    sampler = UniDexMixtureSampler(["wuji2"], seed=42)
    loader = make_training_dataloader(dataset, sampler, batch_size=1, num_workers=0, seed=42)
    wrapper = make_training_wrapper(policy, {
        "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.1},
        "scheduler": {"_target_": __name__ + ".ConstantSchedule"},
    }, {}, {}, sampler)
    observed_algorithms = []
    actual_clipping = wrapper.configure_gradient_clipping

    def observe_clipping(optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        observed_algorithms.append(gradient_clip_algorithm)
        return actual_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)

    wrapper.configure_gradient_clipping = observe_clipping
    clipping_options = {} if algorithm is None else {"gradient_clip_algorithm": algorithm}
    trainer = lightning.Trainer(
        accelerator="cpu", devices=1, max_steps=1, max_epochs=-1,
        gradient_clip_val=1.0, precision="32-true", **clipping_options,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, default_root_dir=str(tmp_path),
    )
    trainer.fit(wrapper, loader)
    # d((2w-10)^2)/dw at w=0 is -40. Global L2 clipping makes it -1,
    # so SGD(lr=.1) must produce w=.1 rather than the unclipped w=4.
    assert policy.linear.weight.item() == pytest.approx(0.1, abs=1e-7)
    assert observed_algorithms == [algorithm]


@pytest.mark.parametrize("algorithm", [None, "norm"])
def test_real_fsdp_root_clipping_accepts_lightning_default_algorithm(tmp_path, algorithm):
    """Actual CPU FSDP dispatch/numerics, not a distributed reduction test."""
    import torch.distributed as distributed
    from pytorch_lightning.plugins.precision import FSDPPrecision
    from torch.distributed.fsdp import FullyShardedDataParallel

    if not distributed.is_available() or not distributed.is_gloo_available():
        pytest.skip("Requires the CPU Gloo distributed backend")
    distributed.init_process_group(
        "gloo", init_method=(tmp_path / "gloo-rendezvous").as_uri(), rank=0, world_size=1,
    )
    try:
        policy = TinyNativePolicy()
        policy.linear = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            policy.linear.weight.zero_()
        wrapper = make_training_wrapper(policy, {
            "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.1},
            "scheduler": {"_target_": __name__ + ".ConstantSchedule"},
        }, {}, {})
        root = FullyShardedDataParallel(policy, use_orig_params=True, device_id=torch.device("cpu"))
        trainer = lightning.Trainer(
            accelerator="cpu", devices=1, plugins=[FSDPPrecision("32-true")],
            gradient_clip_val=1.0, logger=False, enable_checkpointing=False,
            enable_progress_bar=False, default_root_dir=str(tmp_path),
        )
        trainer.strategy.model = root
        wrapper.trainer = trainer
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
        loss, _ = root({"state": torch.tensor([[[2.0]]]), "action": torch.tensor([[[10.0]]]),
                        "index": torch.tensor([0])})
        loss.backward()
        # The old None-only fallback reached the real FSDPPrecision and raised
        # unsupported norm clipping here, before the optimizer could step.
        wrapper.configure_gradient_clipping(optimizer, 1.0, algorithm)
        optimizer.step()
        assert policy.linear.weight.item() == pytest.approx(0.1, abs=1e-7)
    finally:
        distributed.destroy_process_group()
