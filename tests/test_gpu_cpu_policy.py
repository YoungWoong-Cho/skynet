import json
from pathlib import Path
import pytest
from skynet_app.cluster_config import CLUSTER, cpus_for_gpus
from skynet_app.collection import CollectionResources
from skynet_app.experiments import ResourceSpec
from skynet_app.adapters import resolve_adapter_plan
from skynet_app.slurm import compile_sbatch
from test_slurm import make_spec

@pytest.mark.parametrize('gpus', [1, 3, 8])
def test_policy_scales_total_cpus_for_every_gpu_type(gpus):
    for gpu_type in CLUSTER.gpu_aliases:
        resources = ResourceSpec.model_validate({'gpu': {'mode':'explicit','count':gpus,'type':gpu_type}, 'cpus_per_task':12*gpus})
        resolved = resources.with_gpu_cpu_policy(gpus)
        assert resolved.cpus_per_task == 8*gpus
        assert resources.cpus_per_task == 12*gpus
        assert resolved.memory_gb == resources.memory_gb
    assert CollectionResources(gpu_count=gpus, cpu_count=12*gpus).cpu_count == 8*gpus


def test_old_pinned_spec_compiles_with_new_cpu_policy_without_mutation():
    spec = make_spec(); spec.resources.cpus_per_task = 48
    compiled = compile_sbatch(spec, resolve_adapter_plan(spec), run_id='cpu-policy')
    assert '#SBATCH --cpus-per-task=32' in compiled.script
    assert spec.resources.cpus_per_task == 48
    assert '#SBATCH --gres=gpu:l40s:4' in compiled.script


def test_browser_and_runtime_defaults_match_cluster_policy():
    root=Path(__file__).resolve().parents[1]
    assert CLUSTER.defaults.cpus_per_gpu == CLUSTER.defaults.cpus_per_task == 8
    assert 'const CPUS_PER_GPU = 8;' in (root/'static/app.js').read_text()
    for profile in CLUSTER.runtime_profiles.values():
        smoke=profile.verification.compute_smoke
        if smoke: assert smoke.resources.cpus_per_task == cpus_for_gpus(smoke.resources.gpu_count)
