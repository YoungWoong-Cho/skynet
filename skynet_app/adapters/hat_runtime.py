"""Official HAT policy/optimizer with Skynet's shared data and training services.

The upstream network, image transform and L1 + KL + 2*EEF loss are called
directly. Skynet supplies finite episode windows, held-out validation, progress
and checkpoint receipts; no simulator performance is inferred from fit losses.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REVISION = "2d9d73cc5a3859094ef705f35b8f2faecfc2bc4f"
DINO_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
# Official model configs at REVISION. hat_linear is HAT's DINOv2 ViT-S/14 setup;
# act_resnet is the same repository's ImageNet ResNet18 setup with identical
# transformer, CVAE and loss settings. Checkpoints written before this choice
# existed carry no model_config and were all trained with hat_linear.
MODEL_CONFIGS = {"hat_linear": "hdt/configs/models/hat_linear.yaml",
                 "act_resnet": "hdt/configs/models/act_resnet.yaml"}
DEFAULT_MODEL_CONFIG = "hat_linear"
ORIGINAL_MODELS = {
    "hat_linear": "Official hat_linear DINOv2/CVAE, native image transform, optimizer and L1+KL+2*EEF loss",
    "act_resnet": "Official act_resnet ResNet18/CVAE, native image transform, optimizer and L1+KL+2*EEF loss",
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_checkpoint_score(path, step, validation):
    path = Path(path)
    write_json(path.with_name(path.name + ".skynet-checkpoint.json"), dict(
        schema="skynet.checkpoint-score/v1", global_step=step,
        selection=dict(score=float(validation["val_loss"]), mode="min", metric="validation/loss")))


def repository(path):
    root = Path(path).resolve()
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION or subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("HAT requires the clean pinned official source revision")
    return root


def build_policy(root, chunk, learning_rate, model_config=DEFAULT_MODEL_CONFIG):
    if model_config not in MODEL_CONFIGS:
        raise ValueError(f"Unknown official HAT model config {model_config!r}")
    import torch
    import yaml
    from hat_data import CAMERAS
    root = repository(root)
    config = yaml.safe_load((root / MODEL_CONFIGS[model_config]).read_text())
    sys.path[:0] = [str(root / "hdt"), str(root / "hdt/detr"), str(root)]
    from policy import ACTPolicy
    settings = {**config["common"], **config["model"], "camera_names": list(CAMERAS),
                "lr": learning_rate, "chunk_size": chunk, "num_queries": chunk}
    settings["lr_backbone"] = float(settings["lr_backbone"])
    # The upstream literal branch reference is resolved to an immutable official
    # DINOv2 revision. Model factory and pretrained weights remain upstream.
    hub_load = torch.hub.load
    def pinned_hub(repo, *args, **kwargs):
        if repo != "facebookresearch/dinov2":
            raise ValueError("Unexpected HAT backbone repository")
        return hub_load(repo + ":" + DINO_REVISION, *args, **kwargs)
    torch.hub.load = pinned_hub
    try:
        policy = ACTPolicy(settings)
    finally:
        torch.hub.load = hub_load
    return policy, settings


def arguments():
    parser = argparse.ArgumentParser()
    for name in ("repository", "output"):
        parser.add_argument("--" + name, required=True)
    for name, default in (("batch-size",16),("max-steps",3000),("num-workers",2),("seed",42),
                          ("action-steps",50),("validation-every",200)):
        parser.add_argument("--" + name, type=int, default=default)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--dataset")
    inputs.add_argument("--data-spec")
    parser.add_argument("--manifest-sha")
    parser.add_argument("--unique-source-frames", type=int)
    parser.add_argument("--data-selection-seed", type=int, default=20260920)
    parser.add_argument("--window-policy", choices=("pad", "complete"), default="pad")
    parser.add_argument("--mixing-policy", choices=("window_proportional", "hand_balanced"), default="window_proportional")
    parser.add_argument("--checkpoint-selection", choices=("best", "latest"), default="best")
    parser.add_argument("--validation-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--model-config", choices=sorted(MODEL_CONFIGS), default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--control-hz", type=float)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.max_steps, args.action_steps, args.validation_every) < 1 or args.num_workers < 0 or args.learning_rate <= 0:
        parser.error("Invalid HAT training settings")
    if args.dataset and not args.manifest_sha:
        parser.error("A single dataset requires its manifest checksum")
    if args.data_spec and args.manifest_sha:
        parser.error("The frozen data spec owns every input checksum")
    if not args.validation_enabled and args.checkpoint_selection != "latest":
        parser.error("Validation-free training requires the final checkpoint selector")
    return args


def main():
    args = arguments()
    import numpy as np
    import torch
    from hat_data import prepare_data, prepare_collection, HATDataset
    from recording_dataset import close_handles
    from training_parallel import TrainingContext
    root = repository(args.repository)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.data_spec:
        from dataset_inputs import resolve_data_selections
        selections = resolve_data_selections(json.loads(Path(args.data_spec).read_text()))
        datasets, stats, sampling, selections = prepare_collection(selections,
            control_hz=args.control_hz, action_steps=args.action_steps, window_policy=args.window_policy,
            unique_source_frames=args.unique_source_frames, data_selection_seed=args.data_selection_seed)
        splits = [dict(position=item["position"], split=item["metadata"]["split"]) for item in selections]
        episodes = sum(len(item["metadata"]["episodes"]) for item in selections)
    else:
        recordings, values, stats, sampling = prepare_data(args.dataset, args.manifest_sha,
            control_hz=args.control_hz, action_steps=1 if args.verify_only else args.action_steps,
            window_policy=args.window_policy)
        selections = [dict(position=0, version_id=None, path=args.dataset,
                           manifest_sha256=args.manifest_sha, metadata=recordings.manifest)]
        datasets = {split:HATDataset(recordings, values, stats, split, 1 if args.verify_only else args.action_steps, sampling)
                    for split in ("train", "validation")}
        splits, episodes = recordings.manifest["split"], len(recordings)
    training_inputs = [{key:item.get(key) for key in ("position", "version_id", "manifest_sha256")} for item in selections]
    receipt = dict(schema="skynet.hat-loader-validation/v1", manifest_sha256=args.manifest_sha,
        training_inputs=training_inputs, episodes=episodes, split=splits, train_sequences=len(datasets["train"]),
        validation_sequences=len(datasets["validation"]), state_shape=[128],
        action_shape=[1 if args.verify_only else args.action_steps,128],
        normalization="selected_source_frames_only" if args.unique_source_frames else "training_episodes_only",
        observation_mode="rgb", recording_sampling=sampling)
    for split, dataset in datasets.items():
        if len(dataset):
            sample = dataset[0]
            if not all(torch.isfinite(value).all() for value in sample):
                raise ValueError("Nonfinite HAT sample")
    write_json(output / "dataset-validation.json", receipt)
    print(json.dumps(receipt), flush=True)
    close_handles()
    if args.verify_only:
        return
    context = TrainingContext(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    policy, settings = build_policy(root, args.action_steps, args.learning_rate, args.model_config)
    policy.to(context.device)
    optimizer = policy.configure_optimizers()
    from torchvision.transforms import v2
    augment = v2.ColorJitter(brightness=.4, contrast=.4, saturation=.4)
    from hat_data import HandMixtureSampler, make_training_dataloader
    sampler = HandMixtureSampler(datasets["train"].hands, args.mixing_policy,
        seed=args.seed, rank=0, replicas=1, batch_size=args.batch_size)
    loaders = {s:context.loader(d,args.batch_size,args.num_workers,shuffle=False,seed=args.seed)
               for s,d in datasets.items() if len(d) and s != "train" and args.validation_enabled}
    loaders["train"] = make_training_dataloader(datasets["train"],sampler,
        batch_size=args.batch_size,num_workers=args.num_workers,seed=args.seed)
    write_json(output / "sampling.json", sampling)
    write_json(output / "mixture.json", dict(identity=sampler.identity(), epoch=sampler.epoch_plan(1)))
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    write_json(output / "applied-settings.json", dict(schema="skynet.applied-settings/v1", settings=vars(args),
        native_model_config=settings, repository_revision=REVISION, dinov2_revision=DINO_REVISION,
        recording_sampling=sampling, manifest_sha256=args.manifest_sha, training_inputs=training_inputs,
        original_model=ORIGINAL_MODELS[args.model_config],
        skynet_changes="Verified right-hand geometry in HAT slots; camera-OpenGL wrists, metric local tips, absent fingers and unused slots zero; three scene views; source-only statistics; shared finite windows/subsets/hand mixture; fixed episode split and optimizer-step budget"))
    stats_json = {k:v.tolist() for k,v in stats.items()}
    np.savez(output / "normalization.npz", **stats)
    def batch_to_gpu(batch):
        return [x.to(context.device,non_blocking=True) for x in batch]
    def validate():
        if not args.validation_enabled:
            return {}
        split = "validation" if "validation" in loaders else "train"
        policy.eval()
        losses, count, wrist_error, tip_error = 0., 0, 0., 0.
        with torch.random.fork_rng(devices=[context.local_rank]), torch.inference_mode():
            torch.manual_seed(args.seed + 1)
            for batch in loaders[split]:
                image, state, action, padding = batch_to_gpu(batch)
                loss = policy(image,state,action,padding)["loss"]
                context.check_loss(loss)
                prediction = policy(image,state)
                if prediction.shape != action.shape or not torch.isfinite(prediction).all():
                    raise ValueError("HAT prior inference produced invalid action chunks")
                losses += float(loss) * len(state)
                count += len(state)
                scale = torch.as_tensor(stats["action_std"] + 1e-6, device=prediction.device)
                delta = (prediction[:,0]-action[:,0]).abs() * scale
                wrist_error += float(delta[:,30:33].mean()) * len(state)
                tip_error += float(delta[:,43:58].mean()) * len(state)
        return dict(val_loss=losses/count, prior_wrist_mae_m=wrist_error/count, prior_tip_mae_m=tip_error/count)
    def save(name, step, validation):
        path = checkpoint_dir / name
        temporary = path.with_suffix(".tmp")
        torch.save(dict(schema="skynet.hat-checkpoint/v1", model=policy.state_dict(),
            settings=vars(args), native_model_config=settings, normalization=stats_json,
            repository_revision=REVISION, dinov2_revision=DINO_REVISION,
            manifest_sha256=args.manifest_sha, training_inputs=training_inputs if args.data_spec else None,
            recording_sampling=sampling, global_step=step, validation=validation), temporary)
        temporary.replace(path)
        if validation:
            write_checkpoint_score(path, step, validation)
        else:
            write_json(path.with_name(path.name + ".skynet-checkpoint.json"),
                       dict(schema="skynet.checkpoint-score/v1", global_step=step))
    initial = validate()
    best = initial.get("val_loss", float("inf"))
    if args.checkpoint_selection == "best":
        save("best.ckpt",0,initial)
    write_json(output / "initial-validation.json",initial)
    print(json.dumps({"global_step":0, **initial}),flush=True)
    step, epoch, running_loss, running_count = 0, 0, 0., 0
    started = time.monotonic()
    last_validation = initial
    while step < args.max_steps:
        sampler.set_epoch(epoch)
        policy.train()
        for batch in loaders["train"]:
            image,state,action,padding = batch_to_gpu(batch)
            image = augment(image)
            state = state * (torch.rand((len(state),1),device=state.device) >= .1)
            optimizer.zero_grad(set_to_none=True)
            loss = policy(image,state,action,padding)["loss"]
            context.check_loss(loss)
            loss.backward()
            # As in upstream: no gradient clipping or mixed precision override.
            optimizer.step()
            sampler.mark_consumed(len(state))
            step += 1
            running_loss += float(loss.detach())*len(state)
            running_count += len(state)
            if step % 20 == 0 or step % args.validation_every == 0 or step == args.max_steps:
                record = dict(global_step=step,train_loss=running_loss/running_count,lr=args.learning_rate,
                              elapsed_seconds=time.monotonic()-started)
                if step % args.validation_every == 0 or step == args.max_steps:
                    last_validation = validate()
                    record.update(last_validation)
                    if args.checkpoint_selection == "best" and last_validation["val_loss"] < best:
                        best = last_validation["val_loss"]
                        save("best.ckpt",step,last_validation)
                    save("latest.ckpt",step,last_validation)
                    policy.train()
                with (output / "logs.json.txt").open("a") as stream:
                    stream.write(json.dumps(record,allow_nan=False)+"\n")
                print(json.dumps(record),flush=True)
                running_loss,running_count=0.,0
            if step >= args.max_steps:
                break
        epoch += 1
    checkpoint = checkpoint_dir / ("best.ckpt" if args.checkpoint_selection == "best" else "latest.ckpt")
    # Reload the retained checkpoint and check prior inference independently.
    saved = torch.load(checkpoint,map_location=context.device,weights_only=False)
    policy.load_state_dict(saved["model"],strict=True)
    final = validate()
    sample = batch_to_gpu(next(iter(context.loader(datasets["train"], 1, 0, shuffle=False, seed=args.seed))))
    policy.eval()
    with torch.inference_mode():
        prediction = policy(sample[0], sample[1])
        if tuple(prediction.shape) != (1, args.action_steps, 128) or not torch.isfinite(prediction).all():
            raise ValueError("Reloaded HAT checkpoint has invalid prior inference")
    if args.checkpoint_selection == "latest":
        final_path = checkpoint_dir / f"step-{step:08d}.ckpt"
        checkpoint.replace(final_path)
        checkpoint.with_name(checkpoint.name + ".skynet-checkpoint.json").replace(
            final_path.with_name(final_path.name + ".skynet-checkpoint.json"))
        checkpoint = final_path
    write_json(output / "training-result.json",dict(checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        global_step=step, best_step=saved["global_step"], epochs=epoch, stop_reason="max_steps",
        initial_validation=initial,best_validation=final,manifest_sha256=args.manifest_sha, training_inputs=training_inputs,
        checkpoint_selection=args.checkpoint_selection, selected_checkpoint_step=saved["global_step"],
        prior_inference_shape=[args.action_steps,128],prior_inference_finite=True,
        recorded_task_success="not_evaluated"))
    context.close()


if __name__ == "__main__":
    main()
