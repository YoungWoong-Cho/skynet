"""Evaluate an exact native EgoVerse checkpoint on held-out demonstrations."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def evaluate(context, repository):
    import torch
    import hydra
    import numpy as np
    from omegaconf import OmegaConf
    from artifacts import digest, verify
    from egoverse_runtime import register_joint_domain, JOINT_CONTRACT

    checkpoint = context["checkpoint"]
    if digest(checkpoint["path"]) != checkpoint["sha256"]:
        raise ValueError(
            "Checkpoint checksum differs from the selected training result"
        )
    sys.path.insert(0, repository)
    register_joint_domain()
    from egomimic.pl_utils.pl_model import ModelWrapper
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    saved = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
    receipt = saved.get("skynet")
    if not receipt:
        raise ValueError(
            "Checkpoint has no dataset/configuration receipt; use an EgoVerse model adapter checkpoint"
        )
    actual_revision = subprocess.check_output(
        ["git", "-C", repository, "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_revision != receipt["revision"]:
        raise ValueError("Evaluation must use the checkpoint's exact EgoVerse revision")
    config = context["policy"]["native_config"]
    if receipt["manifest_sha256"] != config["dataset_manifest_sha256"]:
        raise ValueError(
            "Evaluation dataset differs from the checkpoint's training dataset"
        )
    manifest = verify(Path(config["dataset_path"]), receipt["manifest_sha256"])
    recorded = manifest["contract"] == JOINT_CONTRACT
    model = ModelWrapper(**saved["hyper_parameters"])
    model.load_state_dict(saved["state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    model.model.device = device
    stats = MultiDataset.from_state(saved["hyper_parameters"]["norm_stats_state"])
    cfg = OmegaConf.create(receipt["config"])
    sources = {}
    for domain, dataset_cfg in cfg.data.valid_datasets.items():
        dataset = hydra.utils.instantiate(dataset_cfg)
        dataset.set_norm_stats_from(stats)
        for name, leaf in dataset.datasets.items():
            path = str(Path(leaf.episode_path).resolve())
            if path in sources:
                raise ValueError(
                    "Duplicate validation episode in native data configuration"
                )
            sources[path] = (domain, dataset, dataset._global_indices_by_dataset[name])
    evaluator = None if recorded else hydra.utils.instantiate(cfg.evaluator)
    if evaluator is not None:
        evaluator.model = model.model
    # Evaluate original episode boundaries, with no resampling between episodes.
    episodes = [manifest["episodes"][i] for i in manifest["split"]["validation"]]
    requested = context["episodes_per_task"]
    if requested > len(episodes):
        raise ValueError(
            f"Only {len(episodes)} held-out episodes exist; choose at most that many"
        )
    results = []
    output = Path(context["result_path"]).parent
    from torch.utils.data._utils.collate import default_collate

    for seed in context["seeds"]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        for index, episode in enumerate(episodes[:requested]):
            import cv2
            import imageio.v2 as imageio

            video = output / "videos" / f"held-out-{seed}-{index}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            squared, elements, observed_metrics = 0.0, 0, {}
            domain, dataset, indices = sources[
                str((Path(config["dataset_path"]) / episode["path"]).resolve())
            ]
            if len(indices) != episode["steps"]:
                raise ValueError(
                    "Native episode length differs from the saved manifest"
                )
            with imageio.get_writer(
                str(video), fps=1 / manifest["capture"]["step_dt"]
            ) as writer:
                for frame in range(episode["steps"]):
                    sample = dataset[indices[frame]]
                    sample = default_collate([sample])
                    sample = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in sample.items()
                    }
                    with torch.no_grad():
                        batch = model.model.process_batch_for_training({domain: sample})
                        if not recorded:
                            metrics, images = evaluator.compute_metrics_and_viz(batch)
                            for key, value in metrics.items():
                                observed_metrics.setdefault(key, []).append(
                                    float(value)
                                )
                            writer.append_data(
                                np.asarray(
                                    next(iter(images.values()))[0], dtype=np.uint8
                                )
                            )
                            continue
                        preds = model.model.forward_eval(batch)
                    if hasattr(model.model, "data_schematic"):
                        actual = preds["actions_joints"]
                        expected = stats.unnormalize(batch, 100)["actions_joints"]
                    else:
                        actual = preds["skynet_joints_actions_joints"]
                        expected = stats.unnormalize(batch[100], 100)["actions_joints"]
                    # Score real future frames only, excluding native end padding.
                    length = min(actual.shape[1], episode["steps"] - frame)
                    error = (actual[:, :length] - expected[:, :length]).float()
                    squared += float(error.square().sum())
                    elements += error.numel()
                    rgb = sample["scene_front"][0].detach().cpu().numpy()
                    rgb = np.ascontiguousarray(
                        np.moveaxis(rgb, 0, -1) * 255, dtype=np.uint8
                    )
                    cv2.putText(
                        rgb,
                        f"Held-out prediction: {frame + 1}/{episode['steps']}",
                        (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
                    cv2.putText(
                        rgb,
                        f"Joint MSE: {squared / elements:.5f}",
                        (5, 38),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
                    writer.append_data(rgb)
            row = dict(
                task="held_out",
                seed=seed,
                episode_index=index,
                success=None,
                status="SUCCEEDED",
                episode_length=episode["steps"],
                metrics=(
                    {"joint_mse": squared / elements}
                    if recorded
                    else {k: statistics.fmean(v) for k, v in observed_metrics.items()}
                ),
                video_path=str(video),
            )
            results.append(row)
            with Path(context["progress_path"]).open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(
                            kind="episode_observed",
                            episode=row,
                            total=requested * len(context["seeds"]),
                            recorded_at=datetime.now(timezone.utc).isoformat(),
                        )
                    )
                    + "\n"
                )
    return result(context, results)


def result(context, episodes):
    aggregate = []
    for key in sorted({k for ep in episodes for k in ep["metrics"]}):
        values = [ep["metrics"][key] for ep in episodes if key in ep["metrics"]]
        aggregate.append(
            dict(
                metric=key,
                unit="scalar",
                mean=statistics.fmean(values),
                std=statistics.pstdev(values),
                sample_count=len(values),
                task="held_out",
            )
        )
    return dict(
        schema_version=1,
        run_id=context["run_id"],
        checkpoint={k: context["checkpoint"][k] for k in ("path", "sha256")},
        evaluator=context["evaluator"],
        environment=dict(
            suite=context["suite"]["name"], version=context["suite"]["version"]
        ),
        episodes=episodes,
        aggregate=aggregate,
        artifacts=[ep["video_path"] for ep in episodes if ep.get("video_path")],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True)
    parser.add_argument("--source-dir", required=True)
    args = parser.parse_args()
    context = json.loads(Path(args.context).read_text())
    Path(context["progress_path"]).parent.mkdir(parents=True, exist_ok=True)
    write_json(context["result_path"], evaluate(context, args.source_dir))


if __name__ == "__main__":
    main()
