"""Policy-side evaluator for recorded ACT; Isaac stays in its own runtime."""

import argparse
import json
import os
from pathlib import Path

from policy_simulator import run_simulator, validate_simulation
from recording_dataset import digest, verify_dataset
from xpolicy_runtime import validate_manifest, checkpoint_sampling
from recording_time import resolve_sampling


class RecordedPolicy:
    def __init__(self, context, source_dir, manifest):
        import numpy as np
        import torch

        self.torch, self.np = torch, np
        self.manifest = validate_manifest(manifest)
        self.kind = context["policy"]["adapter"]
        checkpoint = context["checkpoint"]
        if digest(checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError("Checkpoint changed since it was registered")
        config = context["policy"]["native_config"]
        revision = context["policy"]["source"]["revision"]
        if self.kind == "xpolicylab-act-native":
            from act_native_evaluation import load_native_act
            self.model, self.stats, self.scenes, self.action_steps = load_native_act(context, source_dir, manifest)
            sampling = resolve_sampling(manifest, config.get("control_hz"),
                action_steps=config.get("action_steps", 50), window_policy="pad", require_validation=True)
            self.mode = "rgb"
        elif self.kind == "xpolicylab-act":
            from skynet_act_training import build_policy

            payload = torch.load(
                checkpoint["path"], map_location="cpu", weights_only=False
            )
            if (
                payload.get("schema") != "skynet.act-checkpoint/v1"
                or payload["manifest_sha256"] != config["dataset_manifest_sha256"]
            ):
                raise ValueError("ACT checkpoint does not match the selected dataset")
            sampling = checkpoint_sampling(manifest, config, payload.get("recording_sampling"))
            self.model = build_policy(
                source_dir, revision, payload["settings"], payload["dimension"]
            )
            self.model.load_state_dict(payload["model"])
            self.stats = {
                k: np.asarray(v, dtype=np.float32)
                for k, v in payload["normalization"].items()
            }
            self.mode, self.action_steps = "rgb", payload["settings"]["action_steps"]
            self.scenes = ["scene_front", "scene_left", "scene_right"]
        else:
            raise ValueError("Unsupported recorded-data policy")
        self.control_hz = sampling["control_hz"]
        if self.action_steps != sampling["action_steps"]:
            raise ValueError("ACT checkpoint action chunk differs from its sampling receipt")
        self.model.cuda().eval()

    def reset(self, seed):
        self.torch.manual_seed(seed)
        self.np.random.seed(seed)
        if hasattr(self.model, "reset"):
            self.model.reset()

    def step(self, observation, predict):
        import cv2

        torch, np = self.torch, self.np
        state = np.asarray(observation["state"], dtype=np.float32)
        dim = len(self.manifest["policy_to_source_indices"])
        if state.shape != (dim,) or not np.isfinite(state).all():
            raise ValueError(
                "Simulator joint observations violate the dataset contract"
            )
        if not predict:
            return None
        with torch.inference_mode():
            frames = np.stack(
                [
                    cv2.resize(observation["images"][scene], (640, 480))
                    for scene in self.scenes
                ]
            )
            image = (
                torch.as_tensor(frames, device="cuda")
                .permute(0, 3, 1, 2)
                .float()[None]
                / 255
            )
            qpos = torch.as_tensor(
                (state - self.stats["state_mean"]) / self.stats["state_std"],
                device="cuda",
            )[None]
            result = (
                self.model(qpos, image)[0].cpu().numpy() * self.stats["action_std"]
                + self.stats["action_mean"]
            )
        if result.shape != (self.action_steps, dim) or not np.isfinite(result).all():
            raise ValueError("Policy produced invalid action chunks")
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--context", required=True)
    p.add_argument("--source-dir", required=True)
    args = p.parse_args()
    context = json.loads(Path(args.context).read_text())
    config = context["policy"]["native_config"]
    manifest = validate_manifest(
        verify_dataset(config["dataset_path"], config["dataset_manifest_sha256"])
    )
    validate_simulation(context, manifest)
    print(
        json.dumps(
            dict(
                event="worker_device",
                worker=context.get("worker_index"),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                slurm_step_id=os.environ.get("SLURM_STEP_ID"),
                slurm_step_gpus=os.environ.get("SLURM_STEP_GPUS"),
            )
        ),
        flush=True,
    )
    policy = RecordedPolicy(context, args.source_dir, manifest)
    run_simulator(context, args.context, policy)


if __name__ == "__main__":
    main()
