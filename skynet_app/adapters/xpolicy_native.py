"""Portable launcher for pinned XPolicyLab native training entrypoints.

Runs upstream code in a private workspace. It does not translate observations,
invent hyperparameters, or treat a successful launch as a validated policy.
"""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import signal
import uuid
import subprocess
import sys
import tarfile
import tempfile

from artifacts import verify, digest, relative_file


def read_catalog():
    return json.loads(Path(__file__).with_name("xpolicy_native_catalog.json").read_text())


def inside(root, name):
    path = root / relative_file(name)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Path escapes the native workspace: " + name)
    return path


def gpu_ids(count, environment):
    # Native scripts overwrite CUDA_VISIBLE_DEVICES with their sixth argument.
    # Preserve Slurm's actual device IDs, rather than inventing 0..count-1.
    visible = [s.strip() for s in environment.get("CUDA_VISIBLE_DEVICES", "").split(",") if s.strip()]
    if len(visible) != count or len(set(visible)) != count:
        raise ValueError("CUDA_VISIBLE_DEVICES does not match the allocated GPU count")
    if any(not re.fullmatch(r"[A-Za-z0-9_./-]+", item) for item in visible):
        raise ValueError("Invalid CUDA device identifier")
    return ",".join(visible)


def native_arguments(record, launch, seed, devices):
    kind = record["entry_kind"]
    extra = launch.get("extra_args", [])
    if not isinstance(extra, list) or any(not isinstance(s, str) or "\0" in s for s in extra):
        raise ValueError("extra_args must be a list of arguments")
    if kind == "eventvla":
        names = ["data_mix", "memory_ablation_mode", "keyframe_memory_policy"]
        args = [str(launch.get(k, "")) for k in names]
        if not all(args) or args[2] not in {"teacher", "predict"}:
            raise ValueError("EventVLA requires data_mix, memory_ablation_mode and teacher/predict")
        return args + extra
    if kind == "hy-vla":
        return extra
    values = [launch.get(k, "") for k in ("bench_name", "task_name", "env_cfg_type", "action_type")]
    if any(not isinstance(v, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", v) for v in values):
        raise ValueError("Native inputs require bench_name, task_name, env_cfg_type and action_type")
    if values[3] not in record["action_types"]:
        raise ValueError(record["policy"] + " does not support this action representation")
    if record["policy"] == "OLA_SEM" and (values[0], values[2]) != ("RoboTwin", "aloha_agilex"):
        raise ValueError("OLA_SEM requires RoboTwin and aloha_agilex")
    if seed is None:
        raise ValueError("Native training seed is missing")
    if record["policy"] == "Mem_0":
        stage = launch.get("stage", "execution")
        if stage not in {"execution", "planning", "both"}:
            raise ValueError("Mem_0 stage must be execution, planning or both")
        extra = [stage] + extra
    elif record["policy"] == "RISE":
        stage = launch.get("stage", "all")
        if stage not in {"advantage", "policy", "all"}:
            raise ValueError("RISE stage must be advantage, policy or all")
        extra = [stage] + extra
    elif extra and record["policy"] not in {"H_RDT", "Xiaomi_Robotics_1"}:
        raise ValueError("This upstream launcher has no additional argument contract")
    return values + [str(seed), devices] + extra


def export_source(repository, revision, policy, destination):
    actual = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise ValueError("XPolicyLab source differs from the pinned revision")
    destination.mkdir(parents=True)
    tree = subprocess.check_output(["git", "-C", str(repository), "ls-tree", revision], text=True)
    root_files = [line.split("\t", 1)[1] for line in tree.splitlines() if line.split()[1] == "blob"]
    # Archive tracked files, not local data, checkpoints, uncommitted edits or .git.
    with tempfile.TemporaryFile() as archive:
        subprocess.run(["git", "-C", str(repository), "archive", "--format=tar", revision,
                        "policy/" + policy, "utils", *root_files], stdout=archive, check=True)
        archive.seek(0)
        with tarfile.open(fileobj=archive) as tar:
            for member in tar:
                target = inside(destination, member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
                    target.chmod(member.mode & 0o777)
                else:
                    raise ValueError("Native source archive contains a link or special file: " + member.name)


def copy_inputs(dataset, manifest, workspace):
    for name in manifest["files"]:
        target = inside(workspace, name)
        if target.exists():
            # Configuration is an explicit input; upstream executable code is pinned.
            if target.suffix not in {".yaml", ".yml", ".json"}:
                raise ValueError("Native inputs cannot overwrite pinned code: " + name)
        if ".git" in PurePosixPath(name).parts or ".venv" in PurePosixPath(name).parts:
            raise ValueError("Do not embed a Git repository or Python environment in native inputs")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dataset / name, target)
        target.chmod(target.stat().st_mode | 0o200)


def expand(value, workspace, output):
    if not isinstance(value, str) or "\0" in value:
        raise ValueError("Native environment values must be strings")
    return value.replace("${WORKSPACE}", str(workspace)).replace("${OUTPUT}", str(output))


def environment_for(launch, workspace, output, count, policy):
    environment = os.environ.copy()
    settings = launch.get("environment", {})
    if not isinstance(settings, dict):
        raise ValueError("Native environment must be an object")
    protected = {"CUDA_VISIBLE_DEVICES", "PATH", "PYTHONPATH", "HOME", "LD_PRELOAD",
                 "BASH_ENV", "ENV", "SHELLOPTS", "SLURM_JOB_ID", "SLURM_JOB_GPUS", "RANK", "WORLD_SIZE"}
    for key, value in settings.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) or key in protected or key.startswith("SLURM_"):
            raise ValueError("Native configuration cannot override allocation/runtime variable: " + key)
        if any(word in key for word in ("TOKEN", "PASSWORD", "API_KEY", "WEBHOOK")):
            raise ValueError("Credentials belong in Skynet connections, not a dataset manifest")
        environment[key] = expand(value, workspace, output)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = str(listener.getsockname()[1])
    environment.update({
        "PATH": str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", ""),
        "PYTHONPATH": str(workspace) + os.pathsep + environment.get("PYTHONPATH", ""),
        "PYTHONUNBUFFERED": "1", "GIT_TERMINAL_PROMPT": "0",
        "SHELLOPTS": "errexit:pipefail",
        "NUM_GPUS": str(count), "NPROC_PER_NODE": str(count),
        "PROC_PER_NODE": str(count), "SKYNET_NATIVE_GPU_COUNT": str(count),
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": port,
        "LDA_NUM_PROCESSES": str(count), "DREAMZERO_NUM_GPUS": str(count),
        "NUM_MACHINES": "1", "NUM_NODES": "1", "NODE_COUNT": "1", "NNODES": "1", "NODE_RANK": "0",
        "CHIEF_IP": "127.0.0.1", "INDEX": "0",
        "EVENTVLA_RUN_ROOT_DIR": str(output / "checkpoints"),
        "EXP_ROOT": str(output / "checkpoints"),
        "OPENPI_LOCAL_CACHE_ROOT": str(output / "cache/openpi"),
        "SMOVLA_BASHRC": "/dev/null",
    })
    environment.setdefault("NCCL_SOCKET_IFNAME", "lo")
    environment.setdefault("NCCL_IB_DISABLE", "1")
    if policy in {"SmolVLA", "Being_H05", "InternVLA_A1"}:
        prefix = environment.get("CONDA_PREFIX")
        if not prefix:
            raise ValueError(policy + " requires an installed conda Runtime")
        if policy != "InternVLA_A1":
            environment["SMOVLA_CONDA_ENV" if policy == "SmolVLA" else "BEINGH_CONDA_ENV"] = prefix
    return environment


def adapt_infrastructure(record, directory):
    """Hash-checked, auditable site changes; never rewrite model hyperparameters."""
    changed = []
    for patch in record.get("infrastructure_patches", []):
        path = inside(directory, patch["path"])
        if digest(path) != patch["sha256"]:
            raise ValueError("Infrastructure patch source differs from its audited revision: " + patch["path"])
        text = path.read_text()
        for before, after in patch["replacements"]:
            if text.count(before) != 1:
                raise ValueError("Infrastructure patch must match exactly once: " + patch["path"])
            text = text.replace(before, after)
        path.write_text(text)
        changed.append({"path": patch["path"], "source_sha256": patch["sha256"], "runtime_sha256": digest(path)})
    return changed


def preflight(record, launch, workspace, environment, arguments):
    required = launch.get("required_paths")
    if not isinstance(required, list) or not required or any(not isinstance(p, str) for p in required):
        raise ValueError("Native inputs must declare required_paths for data, configs and pretrained weights")
    for path in required:
        if not inside(workspace, path).exists():
            raise ValueError("Required native input is missing: " + path)
    policy = record["policy"]
    directory = workspace / "XPolicyLab/policy" / policy
    if digest(directory / "train.sh") != record["train_sha256"]:
        raise ValueError("Native training entrypoint does not match the audited source")
    if policy == "Hy_Embodied_05_VLA":
        source = Path(environment.get("HY_VLA_ROOT", directory / "Hy-Embodied-0.5-VLA"))
        if not (source / "scripts/train_robodojo_umi.sh").is_file():
            raise ValueError("Hy-VLA's external training source is missing; install its required Runtime/source first")
    if policy == "TinyVLA":
        tag = "-".join(arguments[:5])
        pretrained = directory / "checkpoints" / tag / "pretrained_vlm"
        if not (pretrained / "config.json").is_file() or not any(
            list(pretrained.glob("*.bin")) + list(pretrained.glob("*.safetensors"))
        ):
            raise ValueError("TinyVLA pretrained_vlm config/weights are missing; interactive downloads are disabled")
    # Two original scripts activate a .venv relative to their nested source tree.
    # Link only the selected, already-installed runtime, never copy an environment.
    nested = {"GR00T_N17": "gr00t_n17", "GalaxeaVLA": "GalaxeaVLA"}.get(policy)
    if nested:
        activation = Path(sys.prefix) / "bin/activate"
        if not activation.is_file():
            raise ValueError(policy + " requires a virtualenv Runtime with bin/activate")
        folder = directory / nested
        if not folder.is_dir():
            raise ValueError("Native framework source is missing: " + str(folder))
        (folder / ".venv").symlink_to(Path(sys.prefix), target_is_directory=True)


def configure_act_lifecycle(args, directory, output, environment):
    from act_native_checkpoint import instrument
    patch = instrument(directory)
    epochs = getattr(args, "epochs", 6000)
    if not 1 <= epochs <= 100000:
        raise ValueError("ACT epochs must be between 1 and 100000")
    stop_request = output / ("act-stop-" + uuid.uuid4().hex)
    environment.update(
        SKYNET_ACT_OUTPUT=str(output), SKYNET_ACT_STOP_REQUEST=str(stop_request),
        SKYNET_ACT_MANIFEST_SHA=args.manifest_sha, SKYNET_ACT_REVISION=args.revision,
        SKYNET_ACT_EPOCHS=str(epochs), SKYNET_ACT_RESUME=getattr(args, "resume", None) or "",
        PYTHONPATH=str(Path(__file__).parent) + os.pathsep + environment["PYTHONPATH"],
    )
    return patch, stop_request


def execute_act(command, directory, environment, stop_request):
    # Keep data workers alive long enough to finish and atomically save this epoch.
    # The Slurm runner signals us; its signal must not kill the inner bash first.
    previous = {}
    def stop(signum, _frame):
        stop_request.write_text(str(signum))
    try:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            previous[sig] = signal.signal(sig, stop)
        with tempfile.TemporaryDirectory(prefix="sk-act-", dir="/tmp") as temporary:
            environment.update(TMPDIR=temporary, TMP=temporary, TEMP=temporary)
            return subprocess.run(command, cwd=directory, env=environment,
                                  stdin=subprocess.DEVNULL, start_new_session=True)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        stop_request.unlink(missing_ok=True)


def run(args):
    records = read_catalog()
    if args.revision != records["revision"]:
        raise ValueError("Requested source revision differs from the audited catalog")
    record = next((p for p in records["policies"] if p["policy"] == args.policy), None)
    if record is None:
        raise ValueError("No audited native training entrypoint for this policy")
    if not record["minimum_gpus"] <= args.gpu_count <= record["maximum_gpus"]:
        raise ValueError("GPU allocation is outside this native launcher's supported range")
    verify_only = getattr(args, "verify_only", False)
    if getattr(args, "resume", None) and (args.policy != "ACT" or verify_only):
        raise ValueError("Native checkpoint resume is supported for ACT training only")
    devices = "0" if verify_only else gpu_ids(args.gpu_count, os.environ)
    dataset, output = Path(args.dataset).resolve(), Path(args.output).resolve()
    manifest = verify(dataset, args.manifest_sha)
    recorded_act = args.policy == "ACT" and manifest.get("format") == "xpolicylab-act-hdf5/v1"
    if recorded_act:
        from act_native_data import validate_recorded_act
        dimensions = validate_recorded_act(dataset, manifest)
    else:
        if manifest.get("format") != f"xpolicylab-native-{args.policy.lower()}/v1" or manifest.get("policy") != args.policy:
            raise ValueError("Dataset was not prepared for the selected native policy")
        if manifest.get("source_revision") != args.revision or manifest.get("contract") != "skynet.xpolicylab-native/v1":
            raise ValueError("Dataset native source/data contract does not match")
    if verify_only and not recorded_act:
        raise ValueError("Loader verification is supported for recorded ACT inputs only")
    output.mkdir(parents=True, exist_ok=True)
    workspace = output / "native-workspace"
    # Every ACT attempt exports fresh pinned code; recovery state is separate.
    if args.policy == "ACT" and not verify_only:
        workspace = workspace / ("attempt-" + uuid.uuid4().hex)
        workspace.mkdir(parents=True)
    else:
        workspace.mkdir()
    export_source(Path(args.repository), args.revision, args.policy, workspace / "XPolicyLab")
    if recorded_act:
        from act_native_data import prepare_recorded_act
        launch = prepare_recorded_act(dataset, manifest, workspace, dimensions)
    else:
        copy_inputs(dataset, manifest, workspace)
        launch = manifest.get("launch", {})
    arguments = native_arguments(record, launch, args.seed, devices)
    environment = environment_for(launch, workspace, output, args.gpu_count, args.policy)
    preflight(record, launch, workspace, environment, arguments)
    directory = workspace / "XPolicyLab/policy" / args.policy
    if verify_only:
        from act_native_data import verify_loader
        receipt = verify_loader(directory, args.manifest_sha, args.seed)
        (output / "loader-validation.json").write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt), flush=True)
        return 0
    patches = adapt_infrastructure(record, directory)
    stop_request = None
    if args.policy == "ACT":
        patch, stop_request = configure_act_lifecycle(args, directory, output, environment)
        patches.append(patch)
    command = ["bash", "-e", "-o", "pipefail", "train.sh", *arguments]
    receipt = {"policy": args.policy, "source_revision": args.revision,
               "dataset_manifest_sha256": args.manifest_sha,
               "argv": command, "cwd": str(directory), "gpu_count": args.gpu_count,
               "infrastructure_patches": patches, "status": "prepared"}
    receipt_path = output / "native-launch.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    # EOF prevents upstream prompts from hanging an unattended Slurm job.
    completed = (execute_act(command, directory, environment, stop_request)
                 if args.policy == "ACT" else
                 subprocess.run(command, cwd=directory, env=environment, stdin=subprocess.DEVNULL))
    receipt.update(status="completed" if completed.returncode == 0 else "failed", exit_code=completed.returncode)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return completed.returncode


def main():
    parser = argparse.ArgumentParser()
    for flag in ("repository", "revision", "policy", "dataset", "manifest-sha", "output"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int, default=6000)
    parser.add_argument("--resume")
    args = parser.parse_args()
    try:
        return run(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print("XPolicyLab native preparation failed: " + str(error), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
