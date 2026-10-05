"""Skynet recording bridge to the official HAT 128-dimensional slot layout.

Original: wrist position/row-6D and palm-relative palm + five fingertips.
Skynet: verified right-hand assets, fixed scene cameras, camera-OpenGL wrist poses from
the existing verified recording codec. Head, left-hand and H1 joint slots are
zero; no H1 joint layout or human demonstration timing is assumed.
"""
import numpy as np

try:
    from .recording_dataset import RecordingDataset, verify_dataset
    from .recording_time import resolve_sampling
    from ops.datasets.action_codecs.unidex import load_codec
    from ops.datasets.action_codecs.geometry import inverse_transform
except ImportError:
    from recording_dataset import RecordingDataset, verify_dataset
    from recording_time import resolve_sampling
    from action_codecs.unidex import load_codec
    from action_codecs.geometry import inverse_transform

CONTRACT = "skynet.hat-rgb-fingertips/v1"
CAMERAS = ("scene_front", "scene_left", "scene_right")


def fingertip_slots(codec):
    """Derive HAT finger identities from the existing verified anatomical map.

    This uses static URDF ancestry, never a hand-name alias or demonstration.
    Missing fingers keep zero slots; no policy architecture or loss is added.
    """
    spec = codec.spec
    if spec.data["side"] != "right" or spec.mapping_status != "VERIFIED":
        raise ValueError("HAT requires a verified right-hand anatomical mapping")
    slots = {row["joint"]: (0 if row["slot"] == 26 else 4 if row["slot"] == 25 else int(row["slot"]) // 5)
             for row in spec.data["mapping"]}
    parents = {joint.child: joint for joint in spec.joints.values()}
    result = []
    for tip in spec.tip_links:
        link, groups = tip, set()
        while link in parents:
            joint = parents[link]
            if joint.name in slots:
                groups.add(slots[joint.name])
            link = joint.parent
        if len(groups) != 1 or not groups <= set(range(5)):
            raise ValueError("HAT fingertip has no unambiguous anatomical finger identity")
        result.append(groups.pop())
    if not result or len(set(result)) != len(result):
        raise ValueError("HAT fingertip identities must be unique")
    return np.asarray(result, dtype=int)


def encode_human_state(codec, native_q, absolute_faas):
    """Reuse verified root/camera wrist geometry; compute actual metric tips by FK."""
    q = np.asarray(native_q)
    wrist = np.asarray(absolute_faas)
    if q.shape != (codec.action_dim,) or wrist.shape != (82,) or not np.isfinite(wrist).all():
        raise ValueError("HAT requires aligned native joints and verified absolute wrist poses")
    poses = codec.spec.forward_kinematics(q)
    local_from_root = inverse_transform(codec.wrist_pose(q))
    tips = [(local_from_root @ poses[name])[:3, 3] for name in codec.spec.tip_links]
    slots = fingertip_slots(codec)
    result = np.zeros(128, dtype=np.float32)
    result[30:39] = wrist[:9]
    result[43:58].reshape(5, 3)[slots] = np.asarray(tips)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite HAT geometry")
    return result


def prepare_data(root, sha, *, control_hz=None, action_steps=50, verify_files=True,
                 window_policy="pad", calculate_stats=True):
    manifest = verify_dataset(root, sha, verify_files=verify_files)
    if manifest.get("contract") != CONTRACT:
        raise ValueError("Select the HAT RGB/fingertip recording preparation")
    sampling = resolve_sampling(manifest, control_hz, action_steps=action_steps, window_policy=window_policy)
    recordings = RecordingDataset(root, manifest=manifest, control_hz=sampling["control_hz"])
    values = {}
    for index, episode in enumerate(manifest["episodes"]):
        codec = load_codec(episode["capture"]["robot"], episode["capture"])
        fingertip_slots(codec)
        rep = episode.get("action_representation", {})
        if (rep.get("id") != "skynet.unidex-faas/v1" or rep.get("state_alignment") != "pre_action"
                or rep.get("codec_sha256") != codec.digest or rep.get("frame") != "camera_opengl"
                or rep.get("action_semantics") != "controller_targets"):
            raise ValueError("HAT requires checksum-pinned absolute controller-target geometry")
        order = episode.get("policy_to_source_indices", manifest.get("policy_to_source_indices"))
        if [episode["capture"]["action_joint_names"][i] for i in order] != codec.action_names:
            raise ValueError("HAT native joint order differs from its verified hand asset")
        reader = recordings.episode(index)
        for camera in CAMERAS:
            stream = episode["streams"].get(camera, {})
            if stream.get("dtype") != "uint8" or stream.get("shape") != [episode["steps"], 256, 256, 3]:
                raise ValueError("HAT requires the three aligned 256x256 RGB scene streams")
            reader.rgb(camera, 0)
        native = {"state": reader.joint("state"), "action": codec.targets(reader.joint("action"))}
        for kind in ("state", "action"):
            wrist = reader.read("faas_" + kind + "_absolute")
            if wrist.shape != (reader.steps, 82):
                raise ValueError("HAT wrist and native joint frames differ")
            values[index, kind] = np.stack([encode_human_state(codec, q, w) for q, w in zip(native[kind], wrist)])
    stats = {}
    if not calculate_stats:
        return recordings, values, stats, sampling
    for kind in ("state", "action"):
        training = np.concatenate([values[i, kind] for i in manifest["split"]["train"]]).astype(np.float64)
        if len(training) < 2:
            raise ValueError("HAT normalization needs at least two training frames")
        stats[kind + "_mean"] = training.mean(0).astype(np.float32)
        stats[kind + "_std"] = np.maximum(training.std(0, ddof=1), .01).astype(np.float32)
    return recordings, values, stats, sampling


class HATDataset:
    def __init__(self, recordings, values, stats, split, chunk, sampling=None):
        self.recordings, self.values, self.stats, self.chunk = recordings, values, stats, chunk
        self.samples = [(i, t) for i in recordings.manifest["split"][split]
                        for t in range(recordings.episode_steps(i) if sampling is None or sampling["window_policy"] == "pad"
                                       else max(0, recordings.episode_steps(i) - chunk + 1))]
        if sampling is not None and split == "train":
            try:
                from .unidex_subset import subset_windows
            except ImportError:
                from unidex_subset import subset_windows
            selected = subset_windows(recordings.manifest, sampling)
            if selected is not None:
                self.samples = [(i, t // sampling["episodes"][i]["stride"]) for i, t in selected]
        self.hands = [recordings.manifest["episodes"][i].get("hand_id") or
                      recordings.manifest["episodes"][i]["capture"]["robot"] for i, _ in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import torch
        episode, step = self.samples[index]
        reader = self.recordings.episode(episode)
        images = np.stack([reader.rgb(camera, step, size=(320, 240), chw=True) for camera in CAMERAS])
        state = self.values[episode, "state"][step]
        actions = self.values[episode, "action"][step:step + self.chunk]
        padded = np.zeros((self.chunk, 128), np.float32)
        padded[:len(actions)] = actions
        # Original HAT z-score with 0.01 std floor and 1e-6 denominator.
        state = (state - self.stats["state_mean"]) / (self.stats["state_std"] + 1e-6)
        padded = (padded - self.stats["action_mean"]) / (self.stats["action_std"] + 1e-6)
        return (torch.from_numpy(images.copy()).float() / 255, torch.from_numpy(state),
                torch.from_numpy(padded), torch.arange(self.chunk) >= len(actions))


def prepare_collection(selections, *, control_hz=None, action_steps=30,
                       unique_source_frames=None, data_selection_seed=20260920, window_policy="complete"):
    """Reuse the source-frame budget and reader for mixed, frozen hand inputs."""
    from torch.utils.data import ConcatDataset
    try:
        from .recording_time import resolve_collection_sampling
        from .unidex_subset import apply_frame_budget
    except ImportError:
        from recording_time import resolve_collection_sampling
        from unidex_subset import apply_frame_budget
    verified = [{**item, "metadata": verify_dataset(item["path"], item["manifest_sha256"])} for item in selections]
    sampling = resolve_collection_sampling(verified, control_hz, action_steps,
                                          window_policy=window_policy, require_validation=True)
    sampling = apply_frame_budget(verified, sampling, unique_source_frames, selection_seed=data_selection_seed)
    parts, training_values = [], {"state": [], "action": []}
    for item, row in zip(verified, sampling["datasets"]):
        recordings, values, _, _ = prepare_data(item["path"], item["manifest_sha256"],
            control_hz=sampling["control_hz"], action_steps=action_steps,
            window_policy=window_policy, calculate_stats=False)
        plan = row["sampling"]
        indices = {i: set(range(recordings.episode_steps(i))) for i in recordings.manifest["split"]["train"]}
        if "training_subset" in plan:
            indices = {}
            for segment in plan["training_subset"]["segments"]:
                i, stride = segment["episode_index"], segment["stride"]
                start = segment["start"] // stride
                indices[i] = set(range(start, start + segment["frames"]))
        for kind in training_values:
            training_values[kind].extend(values[i, kind][sorted(steps)] for i, steps in indices.items() if steps)
        parts.append((recordings, values, plan))
    stats = {}
    for kind, arrays in training_values.items():
        values = np.concatenate(arrays).astype(np.float64)
        if len(values) < 2:
            raise ValueError("HAT normalization needs two selected source frames")
        stats[kind + "_mean"] = values.mean(0).astype(np.float32)
        stats[kind + "_std"] = np.maximum(values.std(0, ddof=1), .01).astype(np.float32)
    datasets = {}
    for split in ("train", "validation"):
        children = [HATDataset(recordings, values, stats, split, action_steps, plan)
                    for recordings, values, plan in parts]
        children = [child for child in children if len(child)]
        if not children:
            raise ValueError("HAT collection has no " + split + " windows")
        datasets[split] = ConcatDataset(children)
        datasets[split].hands = [hand for child in children for hand in child.hands]
    return datasets, stats, sampling, verified
