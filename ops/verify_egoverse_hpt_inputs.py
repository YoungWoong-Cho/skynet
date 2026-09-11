"""Exercise recorded-joint routing through the pinned native HPT on one GPU.

Uses only normalization statistics from an existing checkpoint. Both models start
with fresh weights; the defective checkpoint's policy weights are never reused.
Run with the EgoVerse environment, --source-dir, --support-dir, --dataset,
--manifest-sha, --normalization-checkpoint, and --result.
"""

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile


def verify(args):
    sys.path[:0] = [args.support_dir, args.source_dir]
    import hydra
    import torch
    from omegaconf import open_dict
    from torch.utils.data._utils.collate import default_collate
    from egoverse_runtime import (
        build_config, parser, register_joint_domain, validate_manifest,
    )

    register_joint_domain()
    from egomimic.trainHydra import _build_model_config_tree
    from egomimic.pl_utils.pl_model import ModelWrapper
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    if not torch.cuda.is_available():
        raise RuntimeError("Run this verification inside a one-GPU allocation")
    saved = torch.load(args.normalization_checkpoint, map_location="cpu", weights_only=False)
    if saved["skynet"]["manifest_sha256"] != args.manifest_sha:
        raise ValueError("Normalization statistics belong to a different dataset")
    norm_state = saved["hyper_parameters"]["norm_stats_state"]
    del saved
    manifest = validate_manifest(args.dataset, args.manifest_sha, "hpt_joints")
    runtime_args = parser().parse_args([
        "--repository", args.source_dir, "--dataset", args.dataset,
        "--manifest-sha", args.manifest_sha, "--output", str(Path(args.result).parent),
        "--model", "hpt_joints", "--algorithm", "hpt", "--batch-size", "2",
        "--num-workers", "0", "--epochs", "1", "--validation-every", "1",
        "--train-batches", "2", "--validation-batches", "1", "--reject-outliers", "false",
    ])
    cfg = build_config(runtime_args, manifest)
    stats = MultiDataset.from_state(norm_state)
    dataset = hydra.utils.instantiate(cfg.data.valid_datasets.skynet_joints)
    dataset.set_norm_stats_from(stats)
    sample = default_collate([dataset[0], dataset[1]])
    sample = {key: value.cuda() if isinstance(value, torch.Tensor) else value
              for key, value in sample.items()}

    def exercise(config, state_key, *, roundtrip=False):
        torch.manual_seed(42)
        wrapper = ModelWrapper(
            config_tree=_build_model_config_tree(config), norm_stats_state=norm_state,
        ).cuda()
        model = wrapper.model
        policy = model.nets["policy"]
        stem = policy.stems["skynet_joints_" + state_key]
        observed = {"joint_stem_calls": 0, "trunk_shapes": [], "joint_token_shapes": []}

        def joint_hook(module, inputs, output):
            observed["joint_stem_calls"] += 1
            observed["joint_token_shapes"].append(list(output.shape))

        def trunk_hook(module, inputs):
            observed["trunk_shapes"].append(list(inputs[0].shape))

        handles = [stem.cross_attention.register_forward_hook(joint_hook),
                   policy.trunk["trunk"].register_forward_pre_hook(trunk_hook)]
        wrapper.train()
        processed = model.process_batch_for_training({"skynet_joints": copy.deepcopy(sample)})
        predictions = model.forward_training(processed)
        loss = model.compute_losses(predictions, processed)["action_loss"]
        loss.backward()
        gradient = stem.net[0].weight.grad
        observed["loss"] = float(loss.detach())
        observed["joint_gradient_norm"] = float(gradient.norm()) if gradient is not None else 0.0
        observed["finite_joint_gradient"] = bool(gradient is not None and torch.isfinite(gradient).all())
        assert torch.isfinite(loss), "Native training loss is not finite"
        wrapper.eval()

        def predict(values, target=model):
            torch.manual_seed(123)
            with torch.no_grad():
                batch = target.process_batch_for_training({"skynet_joints": copy.deepcopy(values)})
                return target.forward_eval(batch)["skynet_joints_actions_joints"].detach()

        baseline = predict(sample)
        repeated = predict(sample)
        changed = copy.deepcopy(sample)
        changed["joint_positions"] = changed["joint_positions"].clone()
        changed["joint_positions"][..., 0] += 0.5
        perturbed = predict(changed)
        observed["repeat_prediction_max_delta"] = float((baseline - repeated).abs().max())
        observed["joint_only_prediction_max_delta"] = float((baseline - perturbed).abs().max())
        observed["prediction_shape"] = list(baseline.shape)
        assert torch.isfinite(baseline).all() and torch.isfinite(perturbed).all()
        assert torch.equal(baseline, repeated), "Control prediction is nondeterministic"
        if roundtrip:
            for handle in handles:
                handle.remove()
            with tempfile.TemporaryDirectory() as folder:
                checkpoint = Path(folder) / "roundtrip.ckpt"
                torch.save({"hyper_parameters": dict(wrapper.hparams),
                            "state_dict": wrapper.state_dict()}, checkpoint)
                restored = torch.load(checkpoint, map_location="cpu", weights_only=False)
                reloaded = ModelWrapper(**restored["hyper_parameters"])
                reloaded.load_state_dict(restored["state_dict"], strict=True)
                reloaded.cuda().eval()
                restored_prediction = predict(sample, reloaded.model)
                observed["checkpoint_prediction_max_delta"] = float((baseline - restored_prediction).abs().max())
                assert torch.equal(baseline, restored_prediction), "Checkpoint changed native prediction"
                del reloaded, restored
        else:
            for handle in handles:
                handle.remove()
        del wrapper, model, policy, stem, processed, predictions, loss, gradient
        torch.cuda.empty_cache()
        return observed

    # Negative control: recreate the exact old key mismatch, after config checks.
    old = copy.deepcopy(cfg)
    old_stems = old.model.robomimic_model.stem_specs.skynet_joints
    with open_dict(old_stems):
        old_stems.joint_positions = old_stems.pop("state_joint_positions")
    defective = exercise(old, "joint_positions")
    assert defective["joint_stem_calls"] == 0
    assert defective["joint_gradient_norm"] == 0
    assert defective["joint_only_prediction_max_delta"] == 0

    corrected = exercise(cfg, "state_joint_positions", roundtrip=True)
    assert corrected["joint_stem_calls"] > 0
    assert corrected["finite_joint_gradient"] and corrected["joint_gradient_norm"] > 0
    assert corrected["joint_only_prediction_max_delta"] > 1e-6
    assert all(shape[1:] == [128, 256] for shape in corrected["trunk_shapes"])
    assert all(shape[1:] == [16, 256] for shape in corrected["joint_token_shapes"])
    return {"passed": True, "dataset_manifest_sha256": args.manifest_sha,
            "defective_control": defective, "corrected": corrected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "support-dir", "dataset", "manifest-sha",
                 "normalization-checkpoint", "result"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    try:
        result = verify(args)
    except Exception as error:
        result = {"passed": False, "error": f"{type(error).__name__}: {error}"}
        Path(args.result).write_text(json.dumps(result, indent=2))
        raise
    Path(args.result).write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
