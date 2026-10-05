"""Skynet recorded RGB/joints input for the unmodified official DP policy.

Observation history ends at t. The official policy slices predicted actions
starting at n_obs_steps-1, so the training sequence begins at t-history+1.
Endpoint replication matches upstream DP sampling. No validation frames enter
the normalizer and no RGB or joint arrays are exported to another data store.
"""
import numpy as np

try:
    from .recording_dataset import RecordingDataset, verify_dataset
    from .recording_time import resolve_sampling
    from .policy_contract import recorded_contract, contract_issues
except ImportError:
    from recording_dataset import RecordingDataset, verify_dataset
    from recording_time import resolve_sampling
    from policy_contract import recorded_contract, contract_issues

CONTRACTS = ("skynet.hat-rgb-fingertips/v1", "skynet.act-rgb-joints/v1")
CAMERAS = ("scene_front", "scene_left", "scene_right")
IMAGE_SIZE = (320, 240)  # Same source RGB resize as the Skynet HAT fit loader.


def horizon_for(action_steps, observation_steps):
    if type(action_steps) is not int or type(observation_steps) is not int or min(action_steps, observation_steps) < 1:
        raise ValueError("Positive integer action/observation steps are required")
    # Three levels of the upstream convolutional U-Net require multiples of 4.
    return ((action_steps + observation_steps - 1 + 3) // 4) * 4


def window_indices(step, length, observation_steps, horizon):
    if not 0 <= step < length:
        raise ValueError("Observation index outside episode")
    start = step - observation_steps + 1
    obs = np.clip(np.arange(start, step + 1), 0, length - 1)
    action = np.clip(np.arange(start, start + horizon), 0, length - 1)
    return obs, action


def validate_joint_manifest(manifest):
    if manifest.get("contract") not in CONTRACTS:
        raise ValueError("DP requires a registered RGB recording dataset")
    contract = recorded_contract(manifest)
    issues = contract_issues(contract)
    if issues:
        raise ValueError("; ".join(x["message"] for x in issues))
    dimension = len(contract["joint_names"])
    for episode in manifest["episodes"]:
        capture = episode.get("capture", manifest["capture"])
        order = episode.get("policy_to_source_indices", manifest["policy_to_source_indices"])
        actual = recorded_contract({**manifest, "capture": capture, "policy_to_source_indices": order})
        if actual != contract:
            raise ValueError("DP joint policy requires one consistent hand/camera contract")
        for name in ("state", "action"):
            if episode["streams"][name]["shape"] != [episode["steps"], dimension]:
                raise ValueError("Recorded joint dimensions do not match the policy")
        for name in CAMERAS:
            stream = episode["streams"].get(name, {})
            camera = contract["cameras"][name]
            if stream.get("dtype") != "uint8" or stream.get("shape") != [episode["steps"], camera["height"], camera["width"], 3]:
                raise ValueError("DP requires aligned uint8 RGB camera streams")
    return dimension


class RecordedDPDataset:
    def __init__(self, recordings, split, action_steps=50, observation_steps=2):
        self.recordings = recordings
        self.observation_steps = observation_steps
        self.horizon = horizon_for(action_steps, observation_steps)
        self.samples = [(i, t) for i in recordings.manifest["split"][split]
                        for t in range(recordings.episode_steps(i))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import torch
        episode, step = self.samples[index]
        reader = self.recordings.episode(episode)
        obs_indices, action_indices = window_indices(step, reader.steps, self.observation_steps, self.horizon)
        obs = {"agent_pos": torch.from_numpy(reader.joint("state", obs_indices).astype(np.float32))}
        for name in CAMERAS:
            obs[name] = torch.from_numpy(reader.rgb(name, obs_indices, size=IMAGE_SIZE, chw=True).copy()).float() / 255
        action = torch.from_numpy(reader.joint("action", action_indices).astype(np.float32))
        if not all(torch.isfinite(v).all() for v in [*obs.values(), action]):
            raise ValueError("DP sample contains nonfinite values")
        return {"obs": obs, "action": action}


def training_normalizer(recordings):
    from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
    data = {}
    for source, target in (("state", "agent_pos"), ("action", "action")):
        values = np.concatenate([recordings.episode(i).joint(source)
                                 for i in recordings.manifest["split"]["train"]]).astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite training normalization values")
        data[target] = values
    normalizer = LinearNormalizer()
    normalizer.fit(data, last_n_dims=1, mode="limits")
    for name in CAMERAS:
        normalizer[name] = SingleFieldLinearNormalizer.create_identity()
    return normalizer


def prepare_data(path, sha, *, control_hz=None, action_steps=50, observation_steps=2, verify_files=True):
    manifest = verify_dataset(path, sha, verify_files=verify_files)
    dimension = validate_joint_manifest(manifest)
    sampling = resolve_sampling(manifest, control_hz, action_steps=action_steps, window_policy="pad", require_validation=True)
    recordings = RecordingDataset(path, manifest=manifest, control_hz=sampling["control_hz"])
    datasets = {s: RecordedDPDataset(recordings, s, action_steps, observation_steps)
                for s in ("train", "validation")}
    return recordings, datasets, sampling, dimension
