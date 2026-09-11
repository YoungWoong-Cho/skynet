"""Native EgoVerse ACT/HPT inference on live recorded-joint observations."""

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys

from artifacts import digest, verify
from policy_simulator import run_simulator, validate_simulation


class RecordedPolicy:
    mode = "rgb"

    def __init__(self, context, repository, *, device="cuda"):
        import torch
        from egoverse_runtime import JOINT_CONTRACT, register_joint_domain, validate_hpt_joint_inputs

        checkpoint = context["checkpoint"]
        if digest(checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError("Checkpoint checksum differs from the selected training result")
        saved = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
        receipt = saved.get("skynet") or {}
        self.kind = receipt.get("model")
        if self.kind not in {"act", "hpt_joints"}:
            raise ValueError("Simulator evaluation requires native ACT or HPT recorded joints")
        revision = subprocess.check_output(["git", "-C", repository, "rev-parse", "HEAD"], text=True).strip()
        if revision != receipt.get("revision") or revision != context["policy"]["source"]["revision"]:
            raise ValueError("Evaluation must use the checkpoint's exact EgoVerse revision")
        config = context["policy"]["native_config"]
        if receipt.get("manifest_sha256") != config["dataset_manifest_sha256"]:
            raise ValueError("Evaluation dataset differs from the checkpoint's training dataset")
        self.manifest = verify(config["dataset_path"], config["dataset_manifest_sha256"])
        if self.manifest["contract"] != JOINT_CONTRACT:
            raise ValueError("Simulator evaluation requires the recorded-joint dataset contract")
        if self.kind == "hpt_joints":
            validate_hpt_joint_inputs(receipt["config"]["model"], checkpoint=True)
        sys.path.insert(0, repository)
        register_joint_domain()
        from egomimic.pl_utils.pl_model import ModelWrapper
        from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

        self.device = torch.device(device)
        wrapper = ModelWrapper(**saved["hyper_parameters"])
        wrapper.load_state_dict(saved["state_dict"], strict=True)
        wrapper.to(self.device).eval()
        self.model = wrapper.model
        self.model.device = self.device
        self.stats = MultiDataset.from_state(saved["hyper_parameters"]["norm_stats_state"])
        self.dimension = len(self.manifest["policy_to_source_indices"])
        self.horizon = int(self.stats.key_shape("actions_joints", 100)[0])
        if self.horizon < 1 or self.dimension < 1:
            raise ValueError("Checkpoint action shape is invalid")

    def reset(self, seed):
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if hasattr(self.model, "reset"):
            self.model.reset()

    def sample(self, observation):
        """Match native Zarr decoding and checkpoint normalization exactly once."""
        import numpy as np
        import torch
        from egoverse_runtime import CAMERAS

        state = np.asarray(observation["state"], dtype=np.float32)
        if state.shape != (self.dimension,) or not np.isfinite(state).all():
            raise ValueError("Simulator joints violate the training dataset contract")
        # Native preprocessing needs action shape, but rollout has no action labels.
        sample = {
            "joint_positions": torch.from_numpy(state.copy()),
            "actions_joints": torch.zeros(self.horizon, self.dimension),
        }
        for name in CAMERAS:
            image = np.asarray(observation["images"][name])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Simulator camera must supply an RGB uint8 image")
            sample[name] = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255
        sample = self.stats.normalize(sample, 100)
        return {key: value[None].to(self.device) for key, value in sample.items()}

    def step(self, observation, predict):
        import numpy as np
        import torch

        if not predict:
            return None
        with torch.inference_mode():
            batch = self.model.process_batch_for_training({"skynet_joints": self.sample(observation)})
            if self.kind == "act":
                actions = self.model.forward_eval(batch)["actions_joints"]
            else:
                model = self.model
                data = model._robomimic_to_hpt_data(
                    batch[100], model.camera_keys[100], model.proprio_keys[100],
                    model.lang_keys[100], model.ac_keys[100],
                    model.auxiliary_ac_keys.get("skynet_joints", []),
                )
                # This is the same native forward path used by forward_eval,
                # without its supervised loss on unavailable target actions.
                predictions = model.nets["policy"].forward("skynet_joints", data)
                actions = self.stats.unnormalize({"actions_joints": predictions["skynet_joints"]}, 100)["actions_joints"]
        result = actions[0].float().cpu().numpy()
        if result.shape != (self.horizon, self.dimension) or not np.isfinite(result).all():
            raise ValueError("Native policy produced an invalid action chunk")
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True)
    parser.add_argument("--source-dir", required=True)
    args = parser.parse_args()
    context = json.loads(Path(args.context).read_text())
    policy = RecordedPolicy(context, args.source_dir)
    validate_simulation(context, policy.manifest)
    run_simulator(context, args.context, policy)


if __name__ == "__main__":
    main()
