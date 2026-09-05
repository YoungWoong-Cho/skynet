"""Read-only CPU tracing of the saved state-BC policy. Runs in its Torch runtime.

No simulator or GPU is started. Keep this module standalone for transmission over
SSH; the web server deliberately does not depend on PyTorch or h5py.
"""

import hashlib
import json
from pathlib import Path
import sys

SCHEMA = "skynet.tensor-trace/v1"
POLICY = "skynet-state-bc/v1"


def trace(root, frame, expected):
    import h5py
    import torch

    torch.set_num_threads(1)
    root = Path(root)
    for name, limit in (("state-bc.pt", 16 << 20), ("dataset.hdf5", 256 << 20)):
        path = root / name
        size = path.stat().st_size
        if size > limit:
            raise ValueError(
                f"Tensor inspection supports {name} up to {limit // (1 << 20)} MB"
            )
        if size != expected[name]["size_bytes"]:
            raise ValueError(f"{name} size changed since verification")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected[name]["sha256"]:
            raise ValueError(f"{name} checksum changed since verification")

    saved = torch.load(root / "state-bc.pt", map_location="cpu", weights_only=True)
    if saved.get("schema") != POLICY or saved.get("input_dim") != 35:
        raise ValueError(
            "Unsupported checkpoint: this inspector requires the 35-input state-BC v1 policy"
        )
    if saved["metadata"]["dataset_sha256"] != expected["dataset.hdf5"]["sha256"]:
        raise ValueError("Checkpoint was trained on a different dataset")

    def check_tensor(value, shape):
        if not isinstance(value, torch.Tensor) or list(value.shape) != shape:
            raise ValueError(
                f"Checkpoint tensor has an unsupported shape; expected {shape}"
            )
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError("Only finite float32 tensors are supported")
        return value

    for key, width in (
        ("x_mean", 35),
        ("x_std", 35),
        ("y_mean", 28),
        ("y_std", 28),
        ("action_min", 28),
        ("action_max", 28),
    ):
        check_tensor(saved[key], [width])
    if (saved["x_std"] <= 0).any() or (saved["y_std"] <= 0).any():
        raise ValueError("Checkpoint normalization scales must be positive")
    if (saved["action_min"] > saved["action_max"]).any():
        raise ValueError("Checkpoint action bounds are reversed")
    shapes = {
        "0.weight": [128, 35],
        "0.bias": [128],
        "2.weight": [128, 128],
        "2.bias": [128],
        "4.weight": [28, 128],
        "4.bias": [28],
    }
    if set(saved["state_dict"]) != set(shapes):
        raise ValueError("Unsupported layer structure in checkpoint")
    for key, shape in shapes.items():
        check_tensor(saved["state_dict"][key], shape)
    net = torch.nn.Sequential(
        torch.nn.Linear(35, 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 28),
    )
    net.load_state_dict(saved["state_dict"], strict=True)
    net.eval()
    with h5py.File(root / "dataset.hdf5", "r") as dataset:
        obs = dataset["data/demo_0/obs/state"]
        count = len(obs)
        if type(frame) is not int or not 0 <= frame < count:
            raise ValueError(f"Frame must be between 0 and {count - 1}")
        x = torch.as_tensor(obs[frame]).unsqueeze(0)
        target = torch.as_tensor(dataset["data/demo_0/actions"][frame]).unsqueeze(0)
        source_index = int(dataset["data/demo_0/source_indices"][frame])
    check_tensor(x, [1, 35])
    check_tensor(target, [1, 28])
    steps = []

    def add(identifier, title, explanation, formula, tensor, parameters=None):
        check_tensor(tensor, [1, tensor.shape[1]])
        steps.append(
            {
                "id": identifier,
                "title": title,
                "explanation": explanation,
                "formula": formula,
                "shape": list(tensor.shape),
                "dtype": "float32",
                "values": tensor[0].tolist(),
                "parameters": parameters or [],
                "stats": {
                    "min": tensor.min().item(),
                    "max": tensor.max().item(),
                    "mean": tensor.mean().item(),
                },
            }
        )

    def parameter(name, key):
        return {
            "name": name,
            "shape": list(saved[key].shape),
            "values": saved[key].tolist(),
        }

    with torch.inference_mode():
        add(
            "input",
            "Input observation",
            "One saved simulator observation: 28 robot joint positions followed by 7 task-state values. Feature indices follow the dataset; joint names were not saved.",
            "x = concatenate(proprio, state)",
            x,
        )
        current = (x - saved["x_mean"]) / saved["x_std"]
        add(
            "normalize",
            "Normalize input",
            "Use the mean and standard deviation saved during training to put the input features on a comparable scale.",
            "z = (x − input_mean) / input_std",
            current,
            [parameter("input_mean", "x_mean"), parameter("input_std", "x_std")],
        )
        for index, layer in enumerate(net):
            current = layer(current)
            if isinstance(layer, torch.nn.Linear):
                title = {
                    0: "Linear layer 1",
                    2: "Linear layer 2",
                    4: "Action prediction",
                }[index]
                add(
                    f"layer-{index}",
                    title,
                    "Each output feature is a weighted sum of the previous tensor plus a learned bias."
                    + (
                        " These predicted actions are still normalized."
                        if index == 4
                        else ""
                    ),
                    "y = x @ weightᵀ + bias",
                    current,
                    [
                        {
                            "name": "weight",
                            "shape": list(layer.weight.shape),
                            "count": layer.weight.numel(),
                        },
                        {
                            "name": "bias",
                            "shape": list(layer.bias.shape),
                            "count": layer.bias.numel(),
                        },
                    ],
                )
            else:
                add(
                    f"layer-{index}",
                    f"Tanh activation {1 if index == 1 else 2}",
                    "Apply the hyperbolic tangent to each value. Outputs are bounded between −1 and 1; the tensor shape stays the same.",
                    "y = tanh(x)",
                    current,
                )
        raw = current * saved["y_std"] + saved["y_mean"]
        add(
            "denormalize",
            "Restore action scale",
            "Convert the predicted actions from normalized values back to the action scale used by the dataset.",
            "action = prediction × action_std + action_mean",
            raw,
            [parameter("action_std", "y_std"), parameter("action_mean", "y_mean")],
        )
        action = raw.clamp(saved["action_min"], saved["action_max"])
        changed = (action[0] != raw[0]).nonzero().flatten().tolist()
        add(
            "clamp",
            "Final bounded action",
            "Restrict each action to the range seen during training, matching the evaluation code. This preview does not move the robot or step the simulator.",
            "output = clamp(action, training_min, training_max)",
            action,
            [
                parameter("training_min", "action_min"),
                parameter("training_max", "action_max"),
            ],
        )
        # Verify the explicit layer walk agrees with a normal complete forward pass.
        reference = (
            net((x - saved["x_mean"]) / saved["x_std"]) * saved["y_std"]
            + saved["y_mean"]
        ).clamp(saved["action_min"], saved["action_max"])
        torch.testing.assert_close(action, reference, rtol=0, atol=0)
    return {
        "schema": SCHEMA,
        "policy": POLICY,
        "device": "cpu",
        "frame": frame,
        "frame_count": count,
        "source_index": source_index,
        "steps": steps,
        "checkpoint_sha256": expected["state-bc.pt"]["sha256"],
        "dataset_sha256": expected["dataset.hdf5"]["sha256"],
        "changed_action_indices": changed,
        "demonstrated_action": target[0].tolist(),
        "action_mae": (action - target).abs().mean().item(),
        "parameter_count": sum(p.numel() for p in net.parameters()),
        "provenance": "Saved dataset frame replayed through the saved checkpoint on CPU, in float32. This is not a recorded evaluation rollout; CPU and GPU rounding may differ.",
    }


if __name__ == "__main__":
    try:
        payload = trace(sys.argv[1], int(sys.argv[2]), json.loads(sys.argv[3]))
        print(json.dumps(payload, allow_nan=False))
    except Exception as error:
        print(json.dumps({"error": f"Tensor trace failed: {error}"}))
        sys.exit(1)
