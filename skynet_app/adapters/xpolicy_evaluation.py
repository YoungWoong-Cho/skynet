"""Policy-side evaluator for recorded DP/ACT; Isaac stays in its own runtime."""

import argparse
from collections import deque
import json
import os
from pathlib import Path

from policy_simulator import run_simulator, simulator_environment, validate_simulation
from artifacts import digest, verify
from xpolicy_runtime import repository, validate_manifest


class RecordedPolicy:
    def __init__(self, context, source_dir, manifest):
        import numpy as np
        import torch

        self.torch, self.np = torch, np
        self.manifest = manifest
        self.kind = context["policy"]["adapter"]
        checkpoint = context["checkpoint"]
        if digest(checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError("Checkpoint changed since it was registered")
        config = context["policy"]["native_config"]
        revision = context["policy"]["source"]["revision"]
        if self.kind == "xpolicylab-dp":
            repository(source_dir, revision, "DP")
            import dill
            from hydra.utils import instantiate
            import skynet_dp_training  # Targets frozen in the training checkpoint.

            payload = torch.load(
                checkpoint["path"],
                map_location="cpu",
                pickle_module=dill,
                weights_only=False,
            )
            cfg = payload["cfg"]
            if str(cfg.task.dataset.manifest_path) != str(
                Path(config["dataset_path"]) / "manifest.json"
            ):
                raise ValueError("DP checkpoint belongs to another dataset")
            self.model = instantiate(cfg.policy)
            self.model.load_state_dict(
                payload["state_dicts"]["ema_model" if cfg.training.use_ema else "model"]
            )
            self.mode = cfg.task.dataset.observation_mode
            self.history = deque(maxlen=int(cfg.n_obs_steps))
            self.action_steps = int(cfg.n_action_steps)
        elif self.kind == "xpolicylab-act-native":
            from act_native_evaluation import load_native_act
            self.model, self.stats, self.scenes, self.action_steps = load_native_act(context, source_dir, manifest)
            self.history = deque(maxlen=1)
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
            self.model = build_policy(
                source_dir, revision, payload["settings"], payload["dimension"]
            )
            self.model.load_state_dict(payload["model"])
            self.stats = {
                k: np.asarray(v, dtype=np.float32)
                for k, v in payload["normalization"].items()
            }
            self.history = deque(maxlen=1)
            self.mode, self.action_steps = "rgb", payload["settings"]["action_steps"]
            self.scenes = ["scene_front", "scene_left", "scene_right"]
        else:
            raise ValueError("Unsupported recorded-data policy")
        self.model.cuda().eval()

    def reset(self, seed):
        self.history.clear()
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
        self.history.append(observation)
        while len(self.history) < self.history.maxlen:
            self.history.appendleft(observation)
        if not predict:
            return None
        with torch.inference_mode():
            if self.kind == "xpolicylab-dp":
                obs = {
                    "agent_pos": torch.as_tensor(
                        np.stack([v["state"] for v in self.history]), device="cuda"
                    ).float()[None]
                }
                if self.mode == "rgb":
                    for key, scene in [
                        ("head_cam", "scene_front"),
                        ("left_cam", "scene_left"),
                        ("right_cam", "scene_right"),
                    ]:
                        frames = np.stack(
                            [
                                cv2.resize(
                                    v["images"][scene],
                                    (320, 240),
                                    interpolation=cv2.INTER_AREA,
                                )
                                for v in self.history
                            ]
                        )
                        obs[key] = (
                            torch.as_tensor(frames, device="cuda")
                            .permute(0, 3, 1, 2)
                            .float()[None]
                            / 255
                        )
                result = self.model.predict_action(obs)["action"][0].cpu().numpy()
            else:
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
        verify(config["dataset_path"], config["dataset_manifest_sha256"])
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
