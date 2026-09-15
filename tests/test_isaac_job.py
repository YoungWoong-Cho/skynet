import pytest

from skynet_app.cluster_runtime import WORK_ROOT
from skynet_app.isaac_job import compile_isaac_job


def profile(**changes):
    return dict(account="overcap", partition="overcap", runtime=WORK_ROOT + "/runtime", repository=WORK_ROOT + "/repo", **changes)


def test_simulator_job_preserves_cluster_gpu_aliases_and_quoted_arguments():
    script = compile_isaac_job(profile(gpu_type="rtx_6000"), WORK_ROOT + "/jobs/runs/test", "test", ["python", "worker.py", "argument with spaces"], checks=["verify_source"], after=["verify_result"])
    assert "#SBATCH --gres=gpu:rtx_6000:1" in script
    assert "#SBATCH --account=overcap" in script
    assert "export OMNI_KIT_ACCEPT_EULA=YES" in script
    assert "python worker.py 'argument with spaces'" in script
    assert script.index("verify_source") < script.index("python worker.py") < script.index("verify_result")
    assert "#SBATCH --gres=gpu:1" in compile_isaac_job(profile(gpu_type="any"), WORK_ROOT + "/jobs/runs/test", "test", ["true"])


def test_simulator_job_rejects_unconfigured_gpu_and_invalid_paths():
    with pytest.raises(ValueError, match="GPU type is not configured"):
        compile_isaac_job(profile(gpu_type="made-up-gpu"), WORK_ROOT + "/jobs/runs/test", "test", ["true"])
    for target in ("/tmp/outside", WORK_ROOT + "/../outside", WORK_ROOT + "/space name"):
        with pytest.raises(ValueError):
            compile_isaac_job(profile(gpu_type="any"), target, "test", ["true"])
