"""Install a pinned optional solver beside the simulation env, without changing it."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REVISION = "3846d3fa207165bb0d498145aac8b885a28ea923"
UTILS_REVISION = "2d8dc1a5abf5899069f9ec73c13de73674f4c897"


def prepare(work_root, python):
    parent = Path(work_root).resolve() / "retargeters/vector-wrist-joint"
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / REVISION
    with (parent / ".install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = target / ".skynet-retargeting-ready.json"
        if receipt.is_file():
            metadata = json.loads(receipt.read_text())
            if metadata["revision"] != REVISION or any(
                hashlib.sha256((target / n).read_bytes()).hexdigest() != digest
                for n, digest in metadata["files"].items()
            ):
                raise ValueError(
                    "Existing retargeting installation changed; refusing to overwrite it"
                )
            return target
        if target.exists():
            raise ValueError(
                "Incomplete retargeting installation exists: " + str(target)
            )
        stage = parent / (REVISION + ".partial")
        if stage.exists():
            shutil.rmtree(stage)
        source = stage / "source"
        stage.mkdir()
        try:
            def run(*args, **kwargs):
                return subprocess.run(args, check=True, **kwargs)
            run(
                "git",
                "clone",
                "--no-checkout",
                "https://github.com/Mingrui-Yu/retargeting.git",
                str(source),
            )
            run("git", "-C", str(source), "checkout", "--detach", REVISION)
            run(
                "git", "-C", str(source), "submodule", "update", "--init", "--recursive"
            )
            actual_utils = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(source / "third_party/utils_python"),
                    "rev-parse",
                    "HEAD",
                ],
                text=True,
            ).strip()
            if actual_utils != UTILS_REVISION:
                raise ValueError("Unexpected upstream utility revision")
            # Never let pip replace numpy, torch, scipy, pinocchio or dex-retargeting.
            run(
                str(python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(stage / "dependencies"),
                "scikit-learn==1.7.2",
                "joblib==1.5.2",
                "threadpoolctl==3.6.0",
            )
            paths = [
                source / "src",
                source / "third_party/utils_python",
                stage / "dependencies",
            ]
            env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, paths)))
            run(
                str(python),
                "-c",
                "from retargeting.core.retargeter import Retargeter; import pinocchio, nlopt, sklearn; print('Optional retargeting imports verified')",
                env=env,
            )
            files = {
                str(p.relative_to(stage)): hashlib.sha256(p.read_bytes()).hexdigest()
                for tree in (
                    source / "src/retargeting",
                    source / "third_party/utils_python/mr_utils",
                    source / "configs",
                    stage / "dependencies",
                )
                for p in tree.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
            }
            (stage / ".skynet-retargeting-ready.json").write_text(
                json.dumps(
                    {
                        "revision": REVISION,
                        "utils_revision": UTILS_REVISION,
                        "files": files,
                    },
                    sort_keys=True,
                )
            )
            stage.rename(target)
            return target
        finally:
            if stage.exists():
                shutil.rmtree(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    print(prepare(args.work_root, args.python))
