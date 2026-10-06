"""Every Skynet job takes its account, partition and QOS from the configured queue."""

import pytest

from skynet_app import collection, data_imports, isaac_job, observation_preparation, policy_exports_cluster, recording_preflight, runtime_readiness, slurm
from skynet_app.cluster_config import CLUSTER, QueueProfile


def test_queue_directives_name_the_qos_only_when_configured():
    queue = QueueProfile(partition="p", account="a", max_time_seconds=3600)
    assert queue.sbatch_directives() == ["#SBATCH --account=a", "#SBATCH --partition=p"]
    assert QueueProfile(partition="p", account="a", qos="short", max_time_seconds=3600).sbatch_directives()[-1] == "#SBATCH --qos=short"


def test_every_job_builder_uses_the_queue_directives(monkeypatch):
    # A QOS on the configured queues must reach each generated header through the one helper.
    queues = {name: queue.model_copy(update={"qos": f"qos-{name}"}) for name, queue in CLUSTER.queues.items()}
    configured = CLUSTER.model_copy(update={"queues": queues})
    for module in (collection, data_imports, isaac_job, observation_preparation, policy_exports_cluster, recording_preflight, runtime_readiness, slurm):
        if hasattr(module, "CLUSTER"):
            monkeypatch.setattr(module, "CLUSTER", configured)
    monkeypatch.setattr("skynet_app.cluster_config.CLUSTER", configured)
    default = configured.queue(configured.defaults.queue_policy)
    importing = configured.queue(configured.defaults.import_queue_policy)
    from test_data_imports import RESOURCE, import_request
    job = data_imports.build_huggingface_import_job("a" * 36, RESOURCE, import_request().model_dump())
    assert f"#SBATCH --qos={importing.qos}\n" in job.script
    assert f"#SBATCH --partition={importing.partition}\n#SBATCH --qos={importing.qos}\n#SBATCH --nodes=1" in job.script
    from test_slurm import make_spec
    from skynet_app.adapters import resolve_adapter_plan
    spec = make_spec()
    compiled = slurm.compile_sbatch(spec, resolve_adapter_plan(spec), run_id="qos")
    assert f"#SBATCH --qos={configured.queue_for_partition(spec.resources.partition).qos}" in compiled.script
    root = configured.paths.work_root
    profile = {"account": default.account, "partition": default.partition, "runtime": f"{root}/env", "repository": f"{root}/repo", "gpu_type": "any"}
    assert f"#SBATCH --qos={default.qos}" in isaac_job.compile_isaac_job(profile, f"{root}/jobs/probe", "probe", ["true"])
