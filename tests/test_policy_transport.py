"""The policy and simulator have independently pinned NumPy versions."""

import io
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "skynet_app/adapters"))
    import policy_transport

    return policy_transport


def test_numeric_messages_do_not_reference_numpy_pickle_classes(transport):
    class BuiltinsOnly(pickle.Unpickler):
        def find_class(self, module, name):
            raise AssertionError(f"Unexpected pickled class: {module}.{name}")

    image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)[:, ::-1]
    message = {"command": "step", "observation": {"images": {"scene_front": image},
               "state": np.arange(28, dtype=np.float32)}, "predict": True,
               "result": np.arange(2800, dtype=np.float32).reshape(100, 28), "error": None}
    encoded = pickle.dumps(transport.encode(message), protocol=4)
    decoded = transport.decode(BuiltinsOnly(io.BytesIO(encoded)).load())
    np.testing.assert_array_equal(decoded["result"], message["result"])
    np.testing.assert_array_equal(decoded["observation"]["images"]["scene_front"], image)
    assert decoded["result"].dtype == np.float32
    assert decoded["predict"] is True and decoded["error"] is None


def test_transport_rejects_object_arrays(transport):
    with pytest.raises(ValueError):
        transport.encode(np.array([object()], dtype=object))


def test_zero_exit_without_result_acknowledgement_is_failure(transport, tmp_path):
    from policy_simulator import run_checked_simulator

    marker = tmp_path / "complete"
    with pytest.raises(RuntimeError, match="before writing"):
        run_checked_simulator([sys.executable, "-c", "pass"], completion_path=marker)
    run_checked_simulator(
        [sys.executable, "-c", "import pathlib,sys; pathlib.Path(sys.argv[1]).touch()", str(marker)],
        completion_path=marker,
    )
    with pytest.raises(subprocess.CalledProcessError):
        run_checked_simulator([sys.executable, "-c", "raise SystemExit(1)"], completion_path=marker)
