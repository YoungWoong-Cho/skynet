"""Exercise native launch isolation and failure propagation without GPU jobs."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from skynet_app.adapters import ManifestAdapter
from skynet_app.adapters.xpolicy_native_manifest import catalog, manifests
from test_experiments import make_spec


@pytest.fixture
def runner(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "ops/datasets"))
    spec = importlib.util.spec_from_file_location("native_runner", root / "skynet_app/adapters/xpolicy_native.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_catalog_has_real_training_entries_and_no_inference_stubs():
    records = catalog()
    assert len(records["policies"]) == 38
    assert set(records["without_training_entrypoint"]) == {
        "Dexora_1B", "InternVLA_A1_5", "Meituan_Robotics_0", "MolmoAct2", "OpenDM", "Spatial_Forcing",
    }
    assert len({r["slug"] for r in records["policies"]}) == 38
    for manifest in manifests():
        assert manifest.train.capsule_files["adapter-support/xpolicy_native.py"]
        assert not manifest.evaluations
        assert not manifest.capabilities.supports_resume
        assert not manifest.train.checkpoint_globs
        spec = make_spec()
        spec.source.adapter = manifest.slug
        spec.runtime.backend = "existing"
        spec.native.argv = []
        spec.native.resume_argv = []
        spec.native.config = {"dataset_path": "/cluster/native", "dataset_manifest_sha256": "a" * 64}
        spec.train.checkpoint.auto_resume = manifest.defaults.checkpoint.auto_resume
        spec.resources.gpu.count = manifest.capabilities.minimum_gpus
        spec.intent.explicit_parameters = []
        plan = ManifestAdapter(manifest).resolve(spec)
        assert not plan.blockers, (manifest.slug, plan.blockers)
        assert "{{native.config." not in " ".join(plan.argv)
        if manifest.capabilities.maximum_gpus == 1:
            spec.resources.gpu.count = 2
            assert ManifestAdapter(manifest).resolve(spec).blockers


def test_native_device_mapping_preserves_slurm_ids(runner):
    assert runner.gpu_ids(2, {"CUDA_VISIBLE_DEVICES": "3,5"}) == "3,5"
    assert runner.gpu_ids(1, {"CUDA_VISIBLE_DEVICES": "GPU-abc-def"}) == "GPU-abc-def"
    for value in ("", "0", "0,0", "0,1,2", "0;touch /tmp/wrong,1"):
        with pytest.raises(ValueError):
            runner.gpu_ids(2, {"CUDA_VISIBLE_DEVICES": value})


def test_policy_specific_argument_contracts(runner):
    records = {r["policy"]: r for r in catalog()["policies"]}
    launch = dict(bench_name="RoboDojo", task_name="cube", env_cfg_type="arx_x5", action_type="joint")
    assert runner.native_arguments(records["ACT"], launch, 7, "3") == ["RoboDojo", "cube", "arx_x5", "joint", "7", "3"]
    assert runner.native_arguments(records["Mem_0"], launch, 7, "3")[-1] == "execution"
    assert runner.native_arguments(records["RISE"], launch, 7, "3")[-1] == "all"
    event = dict(data_mix="robotwin_mem8", memory_ablation_mode="pure_image_keyframe_memory", keyframe_memory_policy="teacher")
    assert runner.native_arguments(records["EventVLA"], event, None, "0") == list(event.values())
    assert runner.native_arguments(records["Hy_Embodied_05_VLA"], {}, None, "0") == []
    for name in ("OLA_SEM", "OpenWAM", "X_WAM", "Xiaomi_Robotics_1"):
        with pytest.raises(ValueError):
            runner.native_arguments(records[name], launch, 7, "0")
    with pytest.raises(ValueError, match="additional"):
        runner.native_arguments(records["ACT"], {**launch, "extra_args": ["--ignored"]}, 7, "0")


@pytest.fixture
def native_run(tmp_path, monkeypatch, runner):
    repository = tmp_path / "source"
    policy = repository / "policy/ACT"
    policy.mkdir(parents=True)
    (repository / "utils").mkdir()
    (repository / "utils/module.py").write_text("# tracked source\n")
    (repository / "__init__.py").write_text("# package root\n")
    (repository / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (policy / "train.sh").write_text('printf "%s\\n" "$@" > arguments.txt\nprintf "%s" "$CUDA_VISIBLE_DEVICES" > devices.txt\nprintf changed > data/input.txt\n')
    for args in (["init", "-q"], ["add", "."], ["-c", "user.name=Test", "-c", "user.email=test@example.test", "commit", "-qm", "fixture"]):
        subprocess.run(["git", "-C", str(repository), *args], check=True, capture_output=True)
    revision = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    record = {**next(r for r in catalog()["policies"] if r["policy"] == "ACT"),
              "train_sha256": runner.digest(policy / "train.sh")}
    monkeypatch.setattr(runner, "read_catalog", lambda: {"revision": revision, "policies": [record]})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    dataset = tmp_path / "dataset"
    name = "XPolicyLab/policy/ACT/data/input.txt"
    file = dataset / name
    file.parent.mkdir(parents=True)
    file.write_text("original")
    manifest = {
        "format": "xpolicylab-native-act/v1", "contract": "skynet.xpolicylab-native/v1",
        "policy": "ACT", "source_revision": revision,
        "files": {name: {"sha256": runner.digest(file), "size_bytes": file.stat().st_size}},
        "launch": {"bench_name": "RoboDojo", "task_name": "cube", "env_cfg_type": "arx_x5",
                   "action_type": "joint", "required_paths": [name]},
    }
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    args = argparse.Namespace(repository=str(repository), revision=revision, policy="ACT", dataset=str(dataset),
                              manifest_sha=runner.digest(dataset / "manifest.json"), output=str(tmp_path / "run/artifacts"),
                              gpu_count=1, seed=42)
    return args, dataset, repository, record


def test_real_subprocess_isolated_source_inputs_and_outputs(runner, native_run):
    args, dataset, source, _ = native_run
    # A local source modification must not be imported into the pinned execution.
    (source / "policy/ACT/train.sh").write_text("exit 99\n")
    assert runner.run(args) == 0
    policy = Path(args.output) / "native-workspace/XPolicyLab/policy/ACT"
    assert (policy / "devices.txt").read_text() == "3"
    assert (policy / "arguments.txt").read_text().splitlines() == ["RoboDojo", "cube", "arx_x5", "joint", "42", "3"]
    assert (policy / "data/input.txt").read_text() == "changed"
    assert (dataset / "XPolicyLab/policy/ACT/data/input.txt").read_text() == "original"
    assert not (source / "policy/ACT/arguments.txt").exists()
    assert (policy.parents[1] / "__init__.py").is_file()
    assert json.loads((Path(args.output) / "native-launch.json").read_text())["exit_code"] == 0
    with pytest.raises(FileExistsError):
        runner.run(args)


def test_bad_data_fails_before_creating_workspace(runner, native_run):
    args, dataset, _, _ = native_run
    (dataset / "XPolicyLab/policy/ACT/data/input.txt").write_text("tampered")
    with pytest.raises(ValueError, match="verification"):
        runner.run(args)
    assert not Path(args.output).exists()


def test_child_failure_is_not_marked_success(runner, native_run, monkeypatch):
    args, _, source, record = native_run
    script = source / "policy/ACT/train.sh"
    script.write_text("false | cat\nprintf incorrect-success > succeeded.txt\n")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Test", "-c", "user.email=test@example.test", "commit", "-qm", "failure"], check=True)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    record["train_sha256"] = runner.digest(script)
    args.revision = revision
    manifest_path = Path(args.dataset) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_revision"] = revision
    manifest_path.write_text(json.dumps(manifest))
    args.manifest_sha = runner.digest(manifest_path)
    monkeypatch.setattr(runner, "read_catalog", lambda: {"revision": revision, "policies": [record]})
    assert runner.run(args) != 0
    assert json.loads((Path(args.output) / "native-launch.json").read_text())["status"] == "failed"
    assert not list(Path(args.output).rglob("succeeded.txt"))


def test_environment_and_input_paths_cannot_replace_runtime_or_code(runner, tmp_path):
    for key in ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS", "SHELLOPTS", "WANDB_API_KEY"):
        with pytest.raises(ValueError):
            runner.environment_for({"environment": {key: "wrong"}}, tmp_path, tmp_path, 1, "ACT")
    for path in ("../outside", "/outside"):
        with pytest.raises(ValueError):
            runner.inside(tmp_path, path)
    target = tmp_path / "workspace/train.sh"
    target.parent.mkdir()
    target.write_text("original")
    with pytest.raises(ValueError, match="pinned code"):
        runner.copy_inputs(tmp_path, {"files": {"train.sh": {}}}, target.parent)


def test_infrastructure_patch_rejects_changed_source(runner, tmp_path):
    target = tmp_path / "train.sh"
    target.write_text("num_processes=8\n")
    record = {"infrastructure_patches": [{"path": "train.sh", "sha256": runner.digest(target),
        "replacements": [["num_processes=8", 'num_processes="${SKYNET_NATIVE_GPU_COUNT}"']]}]}
    receipts = runner.adapt_infrastructure(record, tmp_path)
    assert len(receipts) == 1
    assert receipts[0]["source_sha256"] != receipts[0]["runtime_sha256"]
    with pytest.raises(ValueError, match="audited revision"):
        runner.adapt_infrastructure(record, tmp_path)


def test_process_and_network_settings_follow_one_node_allocation(runner, tmp_path):
    env = runner.environment_for({"environment": {"NUM_GPUS": "16", "NODE_COUNT": "2"}}, tmp_path, tmp_path, 3, "H_RDT")
    assert env["NUM_GPUS"] == env["NPROC_PER_NODE"] == env["PROC_PER_NODE"] == "3"
    assert env["NODE_COUNT"] == env["NUM_MACHINES"] == "1"
    assert env["MASTER_ADDR"] == "127.0.0.1"
    assert 0 < int(env["MASTER_PORT"]) < 65536
