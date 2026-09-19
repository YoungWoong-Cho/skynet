"""Install an isolated, pinned UniDex runtime. Never downloads model weights.

--loader-only enables CPU Convert verification. Full setup also compiles native
PyTorch3D against the pinned Torch/CUDA stack; run it where the CUDA toolkit is
available. Training readiness remains a separate GPU/weight validation step.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

UNIDEX_REVISION = "97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d"
PYTORCH3D_REVISION = "33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba"
MARKER = ".skynet-unidex-runtime.json"


def run(argv, **kwargs):
    return subprocess.run([str(value) for value in argv], check=True, **kwargs)


def checkout(path, url, revision):
    path = Path(path).absolute()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "init", path])
        run(["git", "-C", path, "remote", "add", "origin", url])
        run(["git", "-C", path, "fetch", "--depth", "1", "origin", revision])
        run(["git", "-C", path, "checkout", "--detach", "FETCH_HEAD"])
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision or subprocess.run(["git", "-C", str(path), "diff", "--quiet", "HEAD", "--"]).returncode:
        raise ValueError("Existing source differs from the pinned clean checkout: " + str(path))
    return path


def install_cuda_toolkit(path, lock):
    """Install only NVIDIA's pinned compiler/headers/runtime in our own prefix."""
    marker = ".skynet-cuda-toolkit.json"
    if path.exists():
        if not (path / marker).is_file() or json.loads((path / marker).read_text()) != lock:
            raise ValueError("Existing CUDA compiler prefix is not the exact owned toolkit")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="unidex-cuda-", dir=path.parent) as folder:
        staged = Path(folder) / "toolkit"
        staged.mkdir()
        for name, component in lock["components"].items():
            archive = Path(folder) / (name + ".tar.xz")
            with urllib.request.urlopen(component["url"], timeout=120) as source, archive.open("wb") as out:
                shutil.copyfileobj(source, out)
            if archive.stat().st_size != component["size_bytes"] or hashlib.sha256(archive.read_bytes()).hexdigest() != component["sha256"]:
                raise ValueError("NVIDIA CUDA compiler archive integrity verification failed")
            with tarfile.open(archive) as source:
                members = []
                for member in source.getmembers():
                    parts = Path(member.name).parts
                    if len(parts) == 1:
                        continue
                    if Path(member.name).is_absolute() or ".." in parts:
                        raise ValueError("Unsafe CUDA archive member")
                    member.name = str(Path(*parts[1:]))
                    members.append(member)
                source.extractall(staged, members=members, filter="data")
        (staged / marker).write_text(json.dumps(lock, indent=2))
        staged.rename(path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--python", required=True, help="Existing Python 3.11 executable; it is never modified")
    parser.add_argument("--loader-only", action="store_true")
    parser.add_argument("--install-cuda-toolkit", action="store_true", help="Install pinned NVIDIA CUDA 12.6 compiler into the isolated environment; never modify a system toolkit or driver")
    args = parser.parse_args()
    environment = Path(args.environment).absolute()
    uv = shutil.which("uv")
    if uv is None:
        raise ValueError("Install uv outside the runtime before invoking this bootstrap")
    if environment.exists() and not (environment / MARKER).is_file():
        raise ValueError("Refusing to modify an existing environment without the UniDex ownership marker")
    version = subprocess.check_output([args.python, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], text=True).strip()
    if version != "3.11":
        raise ValueError("The locked native runtime requires Python 3.11")
    source = checkout(args.source, "https://github.com/unidex-ai/UniDex", UNIDEX_REVISION)
    environment.parent.mkdir(parents=True, exist_ok=True)
    if not environment.exists():
        run([uv, "venv", "--python", args.python, environment])
        (environment / MARKER).write_text(json.dumps({"schema": "skynet.unidex-runtime/v1", "status": "INSTALLING"}))
    root = Path(__file__).resolve().parent
    lock = root / ("unidex-loader-requirements.lock" if args.loader_only else "unidex-native-requirements.lock")
    python = environment / "bin/python"
    run([uv, "pip", "install", "--python", python, "--require-hashes", "-r", lock])
    if not args.loader_only:
        if args.install_cuda_toolkit:
            toolkit = install_cuda_toolkit(environment / "cuda-12.6.3", json.loads((root / "unidex-cuda-toolkit.json").read_text()))
            os.environ["CUDA_HOME"] = str(toolkit)
            os.environ["PATH"] = str(toolkit / "bin") + os.pathsep + os.environ["PATH"]
        nvcc = shutil.which("nvcc")
        cuda_home = os.environ.get("CUDA_HOME")
        if not nvcc and not (cuda_home and (Path(cuda_home) / "bin/nvcc").is_file()):
            raise ValueError("Full UniDex runtime requires a CUDA toolkit for native PyTorch3D. Loader-only setup remains available.")
        p3d = checkout(source.parent / ("pytorch3d-" + PYTORCH3D_REVISION),
                       "https://github.com/facebookresearch/pytorch3d", PYTORCH3D_REVISION)
        # Torch's pinned NVIDIA wheels already provide cuBLAS/cuSPARSE headers;
        # make those visible to nvcc without downloading another full toolkit.
        headers = subprocess.check_output([python, "-c", "import glob,sysconfig,os; print(os.pathsep.join(sorted(glob.glob(sysconfig.get_paths()['purelib']+'/nvidia/*/include'))))"], text=True).strip()
        env = {**os.environ, "FORCE_CUDA": "1", "MAX_JOBS": os.environ.get("MAX_JOBS", "4"),
               "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0;8.6;8.9;9.0"),
               "CPATH": os.pathsep.join(filter(None, [headers, os.environ.get("CPATH")]))}
        run([uv, "pip", "install", "--python", python, "--no-deps", "--no-build-isolation", p3d], env=env)
        run([python, "-c", "import sys; sys.path.insert(0,sys.argv[1]); import torch,pytorch3d.ops,src.unidex.unidex,src.pointcloud_encoder.uni3d; print('Native UniDex imports passed; GPU model/weights validation remains required')", source])
    else:
        run([python, "-c", "import numpy,h5py,yaml,hydra,omegaconf; print('UniDex CPU loader dependencies passed')"])
    receipt = {"schema": "skynet.unidex-runtime/v1", "status": "LOADER_READY" if args.loader_only else "NATIVE_IMPORTS_READY",
               "source": str(source), "revision": UNIDEX_REVISION, "pytorch3d_revision": None if args.loader_only else PYTORCH3D_REVISION,
               "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(), "gpu_model_verified": False}
    (environment / MARKER).write_text(json.dumps(receipt, indent=2))
    run([uv, "pip", "freeze", "--python", python], stdout=(environment / ".skynet-installed.txt").open("w"))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
