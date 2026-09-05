"""Optional numerical tests: run in a Torch/h5py environment, not the web venv."""

import hashlib

import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

from skynet_app.capture_processing.tensor_trace_runner import trace  # noqa: E402


def artifact(path):
    return {
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


@pytest.fixture
def model(tmp_path):
    torch.manual_seed(31)
    x = torch.linspace(-3, 3, 70, dtype=torch.float32).reshape(2, 35)
    targets = torch.zeros(2, 28, dtype=torch.float32)
    with h5py.File(tmp_path / "dataset.hdf5", "w") as file:
        file["data/demo_0/obs/state"] = x.numpy()
        file["data/demo_0/actions"] = targets.numpy()
        file["data/demo_0/source_indices"] = [12, 24]
    net = torch.nn.Sequential(
        torch.nn.Linear(35, 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 28),
    ).eval()
    saved = {
        "schema": "skynet-state-bc/v1",
        "input_dim": 35,
        "state_dict": net.state_dict(),
        "x_mean": torch.linspace(-0.5, 0.5, 35),
        "x_std": torch.full((35,), 0.7),
        "y_mean": torch.linspace(-0.1, 0.1, 28),
        "y_std": torch.full((28,), 2.0),
        "action_min": torch.full((28,), -0.1),
        "action_max": torch.full((28,), 0.1),
        "metadata": {"dataset_sha256": artifact(tmp_path / "dataset.hdf5")["sha256"]},
    }
    torch.save(saved, tmp_path / "state-bc.pt")
    expected = {
        name: artifact(tmp_path / name) for name in ("state-bc.pt", "dataset.hdf5")
    }
    return tmp_path, expected, x, net, saved


def test_every_layer_matches_the_saved_model_and_frames_change(model):
    root, expected, x, net, saved = model
    first = trace(root, 0, expected)
    last = trace(root, 1, expected)
    for frame, result in enumerate((first, last)):
        with torch.inference_mode():
            values = [x[frame : frame + 1]]
            values.append((values[0] - saved["x_mean"]) / saved["x_std"])
            for layer in net:
                values.append(layer(values[-1]))
            values.append(values[-1] * saved["y_std"] + saved["y_mean"])
            values.append(values[-1].clamp(saved["action_min"], saved["action_max"]))
        for step, value in zip(result["steps"], values, strict=True):
            torch.testing.assert_close(
                torch.tensor(step["values"]), value[0], atol=0, rtol=0
            )
        assert result["changed_action_indices"]
        assert max(abs(v) for v in result["steps"][-1]["values"]) <= 0.100001
        assert result["source_index"] == [12, 24][frame]
    assert first["steps"][0]["values"] != last["steps"][0]["values"]


def test_changed_artifacts_and_out_of_range_fail(model):
    root, expected, *_ = model
    with pytest.raises(ValueError, match="Frame must be"):
        trace(root, 2, expected)
    path = root / "state-bc.pt"
    content = bytearray(path.read_bytes())
    content[200] ^= 1
    path.write_bytes(content)
    with pytest.raises(ValueError, match="checksum changed"):
        trace(root, 0, expected)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda saved: saved.update(schema="unknown"), "Unsupported checkpoint"),
        (lambda saved: saved["x_std"].fill_(0), "positive"),
        (lambda saved: saved["x_mean"].fill_(float("nan")), "finite float32"),
        (lambda saved: saved["state_dict"].pop("0.bias"), "layer structure"),
        (lambda saved: saved["action_min"].fill_(10), "bounds are reversed"),
    ],
)
def test_invalid_checkpoints_fail_without_a_fallback(model, mutation, message):
    root, expected, _, _, saved = model
    mutation(saved)
    torch.save(saved, root / "state-bc.pt")
    expected["state-bc.pt"] = artifact(root / "state-bc.pt")
    with pytest.raises(ValueError, match=message):
        trace(root, 0, expected)
