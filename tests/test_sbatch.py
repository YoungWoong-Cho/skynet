"""Every generated job script shares one sbatch header and takes its shape from the cluster profile."""
import re
from pathlib import Path

from test_runtime_readiness import PROFILE, SUITE
from test_slurm import make_spec

from skynet_app import data_imports, isaac_job, runtime_readiness, slurm
from skynet_app.adapters import resolve_adapter_plan
from skynet_app.cluster_config import CLUSTER, QueueProfile
from skynet_app.sbatch import SHEBANG, cpu_thread_exports, sbatch_header, shell_prelude

QUEUE = QueueProfile(partition="p", account="a", qos="q", max_time_seconds=3600)
JOBS = CLUSTER.defaults.background_jobs


def test_header_orders_directives_and_defaults_error_to_output():
    assert sbatch_header(job_name="j", queue=QUEUE, cpus=2, memory_gb=8, time_limit="00:30:00", output="/o.log") == [
        "#SBATCH --job-name=j", "#SBATCH --account=a", "#SBATCH --partition=p", "#SBATCH --qos=q",
        "#SBATCH --cpus-per-task=2", "#SBATCH --mem=8G", "#SBATCH --time=00:30:00",
        "#SBATCH --output=/o.log", "#SBATCH --error=/o.log",
    ]
    full = sbatch_header(job_name="j", queue=QUEUE, cpus=8, memory_gb=48, time_limit="01:00:00", output="/o", error="/e",
                         gres="gpu:a:1", single_task=True, chdir="/w", extra=["#SBATCH --requeue"])
    assert full[4:6] == ["#SBATCH --nodes=1", "#SBATCH --ntasks=1"], "single-task lines follow the queue directives"
    assert full[6:] == ["#SBATCH --cpus-per-task=8", "#SBATCH --mem=48G", "#SBATCH --gres=gpu:a:1", "#SBATCH --time=01:00:00",
                        "#SBATCH --chdir=/w", "#SBATCH --output=/o", "#SBATCH --error=/e", "#SBATCH --requeue"]
    assert shell_prelude() == ["set -euo pipefail", "umask 077"]
    assert shell_prelude(umask=None, errtrace=True) == ["set -Eeuo pipefail"]
    assert cpu_thread_exports(3) == "export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3"


def test_builders_start_with_the_shared_shebang_and_configured_shapes():
    spec = make_spec()
    training = slurm.compile_sbatch(spec, resolve_adapter_plan(spec), run_id="header").script
    assert training.startswith(SHEBANG + "\n") and "#SBATCH --export=NIL" in training
    root = CLUSTER.paths.work_root
    default = CLUSTER.queue(CLUSTER.defaults.queue_policy)
    profile = {"partition": default.partition, "runtime": f"{root}/env", "repository": f"{root}/repo", "gpu_type": "any"}
    replay = isaac_job.compile_isaac_job(profile, f"{root}/jobs/probe", "probe", ["true"])
    assert replay.startswith(SHEBANG + "\n")
    assert f"#SBATCH --mem={JOBS.isaac_replay.memory_gb}G\n" in replay and f"#SBATCH --time={JOBS.isaac_replay.time_limit}\n" in replay
    assert f"#SBATCH --cpus-per-task={CLUSTER.defaults.cpus_per_gpu}\n" in replay
    video = isaac_job.compile_isaac_job(profile, f"{root}/jobs/probe", "probe", ["true"], resources=JOBS.review_video)
    assert f"#SBATCH --time={JOBS.review_video.time_limit}\n" in video
    readiness = runtime_readiness.render_readiness_sbatch(PROFILE, SUITE)
    assert readiness.startswith(SHEBANG + "\n") and "#SBATCH --export=ALL" in readiness
    from test_data_imports import RESOURCE, import_request
    request = import_request()
    assert (request.cpus, request.memory_gb) == (JOBS.data_import.cpus_per_task, JOBS.data_import.memory_gb)
    imported = data_imports.build_huggingface_import_job("a" * 36, RESOURCE, request.model_dump()).script
    assert imported.startswith(SHEBANG + "\n") and "\numask 0027\n" in imported


def test_no_builder_spells_its_own_resource_literals():
    # Memory, time and CPU directives come from a request or the cluster profile, never a literal.
    pattern = re.compile(r"#SBATCH --(?:mem=\d|time=\d|cpus-per-task=\d)")
    offenders = sorted(path.name for path in Path("skynet_app").rglob("*.py") if pattern.search(path.read_text()))
    assert offenders == []
