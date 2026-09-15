import importlib.util
import os
from pathlib import Path
import time



spec = importlib.util.spec_from_file_location(
    "native_session", Path(__file__).resolve().parents[1] / "ops/xr/native_session.py"
)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_failure_uses_this_sessions_redirected_sdk_log_only(tmp_path):
    import tempfile
    started = time.time() - 1
    # Older runtime versions redirect SDK output into /tmp.
    with tempfile.NamedTemporaryFile(prefix="cxr_server.", suffix=".log", dir="/tmp") as sdk:
        sdk.write(b"(E)<Signaling> Failed to start signaling server: 'Net Exception: Address already in use: 0.0.0.0:48010'\n")
        sdk.flush()
        (tmp_path / "cloudxr.log").write_text(f"Logging to {sdk.name}\n")
        reason = worker.cloudxr_failure(tmp_path, started, "CloudXR exited before readiness")
        assert "0.0.0.0:48010 is already in use" in reason
        os.utime(sdk.name, (started - 20, started - 20))
        assert worker.cloudxr_failure(tmp_path, started, "Failed") == "Failed; inspect the CloudXR log"


def test_failure_reads_bounded_session_log_and_ignores_symlink(tmp_path):
    started = time.time() - 1
    log = tmp_path / "cxr_server.current.log"
    log.write_text("(E)<GPU> stale failure\n" + "x\n" * 100000 + "(E)<GPU> Failed to initialize encoder\n")
    assert worker.cloudxr_failure(tmp_path, started, "Failed") == "Failed: Failed to initialize encoder"
    log.unlink()
    unrelated = tmp_path / "another-service.log"
    unrelated.write_text("(E)<Signaling> Address already in use: 0.0.0.0:48010\n")
    log.symlink_to(unrelated)
    assert worker.cloudxr_failure(tmp_path, started, "Failed") == "Failed; inspect the CloudXR log"


def test_gpu_exhaustion_is_reported_instead_of_secondary_joint_failure(tmp_path):
    log = tmp_path / "simulation.log"
    log.write_text(
        "[Error] [carb.graphics-vulkan.plugin] Out of GPU memory allocating resource\n"
        "Exception: Failed to get DOF velocities from backend\n"
    )
    result = worker.simulation_failure(log, "Simulator ended unexpectedly")
    assert "GPU memory is exhausted" in result
    assert "DOF velocities" not in result
