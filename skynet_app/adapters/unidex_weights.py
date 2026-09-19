"""Explicit, fail-closed weight identity for native UniDex initialization."""
from pathlib import Path, PurePosixPath
import hashlib
import json


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_weight_provenance(base_path, pointcloud_path, provenance_path):
    base, pointcloud = Path(base_path).resolve(), Path(pointcloud_path).resolve()
    receipt = json.loads(Path(provenance_path).read_text())
    if receipt.get("schema") != "skynet.unidex-weights/v1":
        raise ValueError("Expected an explicit skynet.unidex-weights/v1 provenance file")
    base_receipt, pc_receipt = receipt.get("base", {}), receipt.get("pointcloud", {})
    if not all(isinstance(row.get("origin"), str) and row["origin"].strip() for row in (base_receipt, pc_receipt)):
        raise ValueError("Declare the origin/revision of both PaliGemma and Uni3D pretrained weights")
    files = base_receipt.get("files", {})
    actual = {str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()} if base.is_dir() else set()
    if not files or actual != set(files) or not any(p.endswith(".safetensors") for p in files):
        raise ValueError("PaliGemma provenance must cover every local weight/tokenizer file")
    for name, checksum in files.items():
        path = (base / name).resolve()
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or digest(path) != checksum:
            raise ValueError("PaliGemma weight/tokenizer checksum mismatch: " + name)
    if not pointcloud.is_file() or digest(pointcloud) != pc_receipt.get("sha256"):
        raise ValueError("Uni3D checkpoint checksum differs from its provenance")
    return receipt


def _tensor_state(value):
    if not isinstance(value, dict):
        raise ValueError("Expected a tensor state dictionary")
    for key in ("state_dict", "module", "model"):
        if key in value and isinstance(value[key], dict):
            value = value[key]
            break
    return value


def initialize_pretrained(policy, base_path, pointcloud_path):
    """Load native modules with complete key/shape coverage, never random fallback.

    UniDex's official helper starts from existing random parameters and uses
    non-strict loads. Here missing pretrained tensors are an explicit failure.
    The architecture and native flow-matching objective are unchanged.
    """
    import torch
    from safetensors import safe_open

    tensors = {}
    for path in sorted(Path(base_path).glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as stream:
            for key in stream.keys():
                if key.startswith("language_model.model."):
                    if key in tensors:
                        raise ValueError("Duplicate PaliGemma tensor: " + key)
                    tensors[key] = stream.get_tensor(key)
    embedding = tensors.pop("language_model.model.embed_tokens.weight", None)
    if embedding is None or embedding.shape != policy.embed_tokens.weight.shape:
        raise ValueError("PaliGemma embedding tensor is missing or has the wrong shape")
    policy.embed_tokens.load_state_dict({"weight": embedding}, strict=True)
    native_vlm = policy.joint_model.mixtures["vlm"]
    vlm = {k.removeprefix("language_model.model."): v for k, v in tensors.items()}
    expected = native_vlm.state_dict()
    missing = [key for key, value in expected.items() if key not in vlm or vlm[key].shape != value.shape]
    if missing:
        raise ValueError("PaliGemma tensors do not cover native VLM: " + ", ".join(missing[:8]))
    native_vlm.load_state_dict({key: vlm[key] for key in expected}, strict=True)
    pointcloud = _tensor_state(torch.load(pointcloud_path, map_location="cpu", weights_only=False))
    # Distributed checkpoints may prefix every key; partial guessing is refused.
    if pointcloud and all(key.startswith("module.") for key in pointcloud):
        pointcloud = {key[7:]: value for key, value in pointcloud.items()}
    expected_pc = policy.pointcloud_encoder.state_dict()
    if set(pointcloud) != set(expected_pc) or any(pointcloud[key].shape != value.shape for key, value in expected_pc.items()):
        raise ValueError("Uni3D checkpoint keys/shapes do not match the pinned large encoder")
    policy.pointcloud_encoder.load_state_dict(pointcloud, strict=True)
    return {"paligemma_tensors": len(expected) + 1, "uni3d_tensors": len(expected_pc)}


def load_policy_state(policy, checkpoint_path):
    import torch
    state = _tensor_state(torch.load(checkpoint_path, map_location="cpu", weights_only=False))
    if state and all(key.startswith("policy.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    policy.load_state_dict(state, strict=True)
    return {"policy_tensors": len(state)}
