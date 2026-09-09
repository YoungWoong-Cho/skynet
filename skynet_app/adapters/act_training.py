"""Pinned XPolicyLab ACT with registered splits and aligned image/action samples."""

import argparse
import json
from contextlib import nullcontext
import os
from pathlib import Path

from xpolicy_runtime import repository, validate_manifest, normalization, write_json

CAMERAS = ["cam_head", "cam_left_wrist", "cam_right_wrist"]


def read_sample(root, episode, step, stats, chunk):
    import h5py
    import numpy as np

    with h5py.File(Path(root) / "dataset" / f"episode_{episode}.hdf5", "r") as f:
        qpos = f["observations/qpos"][step]
        images = np.stack([f["observations/images/" + c][step] for c in CAMERAS])
        # Every observation precedes its matching command. Never shift backwards.
        actions = f["action"][step : step + chunk]
    count = len(actions)
    padded = np.zeros((chunk, len(qpos)), dtype=np.float32)
    padded[:count] = (actions - stats["action_mean"]) / stats["action_std"]
    qpos = (qpos - stats["state_mean"]) / stats["state_std"]
    if (
        images.shape != (3, 480, 640, 3)
        or images.dtype != np.uint8
        or not np.isfinite(qpos).all()
        or not np.isfinite(padded).all()
    ):
        raise ValueError(
            "ACT requires three RGB cameras and finite aligned joint commands"
        )
    return qpos, images, padded, np.arange(chunk) >= count


def build_policy(repository_path, revision, settings, dimension):
    repository(repository_path, revision, "ACT")
    os.environ["ACT_ACTION_DIM"] = str(dimension)
    from detr.main import get_args_parser
    from detr.act_policy import ACTPolicy

    args = get_args_parser().parse_args(
        [
            "--ckpt_dir",
            ".",
            "--policy_class",
            "ACT",
            "--bench_name",
            "Skynet",
            "--task_name",
            "recorded",
            "--seed",
            str(settings["seed"]),
            "--num_epochs",
            str(settings["epochs"]),
        ]
    )
    overrides = dict(
        lr=settings["learning_rate"],
        lr_backbone=settings["backbone_learning_rate"],
        weight_decay=settings["weight_decay"],
        chunk_size=settings["action_steps"],
        kl_weight=settings["kl_weight"],
        hidden_dim=settings["hidden_dim"],
        dim_feedforward=settings["feedforward_dim"],
        enc_layers=4,
        dec_layers=7,
        nheads=8,
        backbone="resnet18",
        camera_names=CAMERAS,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return ACTPolicy(overrides, args)


class RecordedACTDataset:
    def __init__(self, root, manifest, stats, split, chunk):
        self.root, self.stats, self.chunk = Path(root), stats, chunk
        self.samples = [(i, t) for i in manifest["split"][split]
            for t in range(manifest["episodes"][i]["steps"])]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import torch
        episode, step = self.samples[index]
        qpos, images, padded, mask = read_sample(self.root, episode, step, self.stats, self.chunk)
        image = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).float() / 255
        return torch.from_numpy(qpos), image, torch.from_numpy(padded), torch.from_numpy(mask)


def arguments():
    parser = argparse.ArgumentParser()
    for key in ["repository", "revision", "dataset", "manifest-sha", "output"]:
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--training-preset", default="xpolicylab-act/v1")
    for key, value in dict(
        batch_size=16,
        gpu_count=1,
        epochs=6000,
        seed=42,
        action_steps=50,
        hidden_dim=512,
        feedforward_dim=3200,
        gradient_accumulation=1,
        num_workers=0,
        early_stopping_patience=20,
    ).items():
        parser.add_argument("--" + key.replace("_", "-"), type=int, default=value)
    for key, value in dict(
        learning_rate=1e-5,
        backbone_learning_rate=1e-5,
        weight_decay=1e-4,
        kl_weight=10.0,
        gradient_clip=1.0,
    ).items():
        parser.add_argument("--" + key.replace("_", "-"), type=float, default=value)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--batch-semantics",
        default="per_device",
        choices=[
            "per_device",
            "global_before_accumulation",
            "global_effective",
            "repository_native",
        ],
    )
    args = parser.parse_args()
    for key in [
        "batch_size",
        "gpu_count",
        "epochs",
        "action_steps",
        "hidden_dim",
        "feedforward_dim",
        "gradient_accumulation",
    ]:
        if getattr(args, key) < 1:
            parser.error(key + " must be positive")
    if (
        args.hidden_dim % 8
        or args.num_workers < 0
        or args.early_stopping_patience < 0
        or args.kl_weight < 0
        or args.gradient_clip <= 0
        or args.learning_rate <= 0
        or args.backbone_learning_rate <= 0
        or args.weight_decay < 0
    ):
        parser.error("Invalid ACT settings")
    return args


def main():
    args = arguments()
    from training_parallel import launch_distributed, TrainingContext
    if not args.verify_only and launch_distributed(args.gpu_count):
        return
    context = TrainingContext(args.gpu_count) if not args.verify_only else None
    primary = context is None or context.primary
    import numpy as np
    import torch
    from training_parallel import PolicyLoss, training_batch_size
    from artifacts import verify, digest

    root, output = Path(args.dataset), Path(args.output)
    if primary:
        manifest = validate_manifest(verify(root, args.manifest_sha))
    if context is not None and context.world_size > 1:
        context.dist.barrier()
    if not primary:
        manifest = validate_manifest(json.loads((root / "manifest.json").read_text()))
    if (
        manifest.get("contract") != "skynet.act-rgb-joints/v1"
        or manifest["format"] != "xpolicylab-act-hdf5/v1"
    ):
        raise ValueError("Select a prepared ACT RGB dataset")
    stats = normalization(root, manifest)
    datasets = {
        s: RecordedACTDataset(root, manifest, stats, s, args.action_steps)
        for s in ["train", "validation"]
    }
    # Validate every episode without reading all image pixels a second time.
    import h5py

    dim = len(manifest["policy_to_source_indices"])
    for i, ep in enumerate(manifest["episodes"]):
        with h5py.File(root / "dataset" / f"episode_{i}.hdf5", "r") as f:
            if f["action"].shape != (ep["steps"], dim) or f[
                "observations/qpos"
            ].shape != (ep["steps"], dim):
                raise ValueError(
                    "ACT episode length or joint dimensions differ from the manifest"
                )
            for camera in CAMERAS:
                if f["observations/images/" + camera].shape != (
                    ep["steps"],
                    480,
                    640,
                    3,
                ):
                    raise ValueError(
                        "ACT camera shape differs from the conversion contract"
                    )
    receipt = dict(
        schema="skynet.act-loader-validation/v1",
        manifest_sha256=args.manifest_sha,
        split=manifest["split"],
        train_sequences=len(datasets["train"]),
        validation_sequences=len(datasets["validation"]),
        action_shape=[args.action_steps, dim],
        normalization="training_episodes_only",
        observation_mode="rgb",
    )
    output.mkdir(parents=True, exist_ok=True)
    if primary:
        write_json(output / "dataset-validation.json", receipt)
        print(json.dumps(receipt), flush=True)
    if args.verify_only:
        return
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    policy = build_policy(args.repository, args.revision, vars(args), dim)
    optimizer = policy.configure_optimizers()
    compute_loss = context.wrap(PolicyLoss(policy))
    validation_loss = PolicyLoss(policy)
    torch.manual_seed(args.seed + context.rank)
    batch_size = training_batch_size(
        args.batch_size,
        args.gpu_count,
        args.gradient_accumulation,
        args.batch_semantics,
    )
    loaders = {s: context.loader(d, batch_size, args.num_workers, shuffle=(s == "train"), seed=args.seed)
        for s, d in datasets.items()}
    if primary:
        write_json(
            output / "applied-settings.json",
            dict(
                schema="skynet.applied-settings/v1",
                settings=vars(args),
                effective_batch_size=batch_size * args.gradient_accumulation,
                manifest_sha256=args.manifest_sha,
                repository_revision=args.revision,
            ),
        )
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    best, stale, global_step, reason = float("inf"), 0, 0, "max_epochs"
    for epoch in range(args.epochs):
        means = {}
        for split, loader in loaders.items():
            training = split == "train"
            policy.train(training)
            loader.batch_sampler.set_epoch(epoch)
            loss_sum = examples = window_examples = window_expected = 0
            optimizer.zero_grad(set_to_none=True)
            with (
                torch.random.fork_rng(devices=[context.local_rank])
                if not training
                else nullcontext()
            ):
                if not training:
                    torch.manual_seed(args.seed + context.rank + 1)
                with torch.set_grad_enabled(training):
                    for batch_index, batch in enumerate(loader):
                        inputs = [x.to(context.device, non_blocking=True) for x in batch]
                        loss = (compute_loss if training else validation_loss)(inputs)[0]["loss"]
                        context.check_loss(loss)
                        count = loader.batch_sampler.valid_count(batch_index)
                        loss_sum += loss.item() * count
                        examples += count
                        if training:
                            context.backward(loss, count)
                            window_examples += count
                            window_expected += loader.batch_sampler.global_count(batch_index)
                            if (
                                batch_index + 1
                            ) % args.gradient_accumulation == 0 or batch_index + 1 == len(
                                loader
                            ):
                                context.normalize_gradients(policy.parameters(), window_examples, window_expected)
                                window_examples = window_expected = 0
                                torch.nn.utils.clip_grad_norm_(
                                    policy.parameters(), args.gradient_clip, error_if_nonfinite=True
                                )
                                optimizer.step()
                                optimizer.zero_grad(set_to_none=True)
                                global_step += 1
            means[split] = context.mean(loss_sum, examples, len(loader.dataset))
        improved = means["validation"] < best
        if improved:
            best, stale = means["validation"], 0
        else:
            stale += 1
        stopping = bool(
            args.early_stopping_patience and stale >= args.early_stopping_patience
        )
        record = dict(
            epoch=epoch,
            global_step=global_step,
            train_loss=means["train"],
            val_loss=means["validation"],
            best_val_loss=best,
            lr=optimizer.param_groups[0]["lr"],
            early_stopping=stopping,
        )
        if primary:
            with (output / "logs.json.txt").open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            payload = dict(
                schema="skynet.act-checkpoint/v1",
                settings=vars(args),
                model=policy.state_dict(),
                normalization={k: v.tolist() for k, v in stats.items()},
                dimension=dim,
                epoch=epoch + 1,
                global_step=global_step,
                manifest_sha256=args.manifest_sha,
            )
            for name in ["latest.ckpt", *(["best.ckpt"] if improved else [])]:
                temporary = checkpoints / (name + ".tmp")
                torch.save(payload, temporary)
                temporary.replace(checkpoints / name)
        if stopping:
            reason = "early_stopping"
            break
    if not primary:
        context.close()
        return
    checkpoint = checkpoints / "best.ckpt"
    write_json(
        output / "training-result.json",
        dict(
            checkpoint=str(checkpoint),
            checkpoint_sha256=digest(checkpoint),
            global_step=global_step,
            epochs=epoch + 1,
            stop_reason=reason,
            manifest_sha256=args.manifest_sha,
        ),
    )

    context.close()


if __name__ == "__main__":
    main()
