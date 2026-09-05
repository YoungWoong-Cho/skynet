"""Idempotent setup of pinned source and existing assets; never edits a shared runtime."""

from __future__ import annotations
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

from skynet_app.cluster_runtime import ClusterClient, WORK_ROOT


def configure(profile, cluster=None):
    cluster = cluster or ClusterClient()
    source = Path(profile["local_source"]).expanduser()
    revision = profile["source_revision"]
    if not re.fullmatch("[a-f0-9]{40}", revision):
        raise ValueError("Setup requires an exact source commit")
    if not (source / ".git").exists():
        raise ValueError(
            f"DexVerse source checkout is missing: {source}. Set local_source in config/capture_pipelines.json."
        )
    archive = subprocess.run(
        ["git", "-C", str(source), "archive", "--format=tar", revision],
        capture_output=True,
        check=False,
        timeout=20,
    )
    if archive.returncode:
        raise ValueError(
            "The pinned DexVerse revision is unavailable in the configured local checkout"
        )
    if len(archive.stdout) > 32 * 1024 * 1024:
        raise ValueError("Source archive exceeds the 32 MB setup limit")
    for key in ("repository", "runtime", "asset_bundle"):
        value = cluster._remote_path(profile[key])
        if not value.startswith(WORK_ROOT + "/"):
            raise ValueError("Setup paths must be inside the cluster workspace")
    archive_sha = hashlib.sha256(archive.stdout).hexdigest()
    payload = {
        "archive": base64.b64encode(archive.stdout).decode(),
        "archive_sha256": archive_sha,
        "profile": profile,
    }
    script = """import base64,hashlib,io,json,os,subprocess,sys,tarfile
from pathlib import Path
payload=json.load(sys.stdin); profile=payload['profile']
root=Path(profile['repository']); bundle=Path(profile['asset_bundle'])
runtime=Path(profile['runtime'])
if not (runtime/'bin/python').is_file(): raise ValueError('Install the pinned Isaac Sim 5.1 / Isaac Lab 2.3.2 Python 3.11 runtime first; this setup does not replace existing runtimes')
if not (bundle/'READY').is_file(): raise ValueError('Verified DexVerse assets are missing. Prepare the asset bundle with ops/datasets/download_dexverse_assets.sbatch first')
raw=base64.b64decode(payload['archive'],validate=True)
if hashlib.sha256(raw).hexdigest()!=payload['archive_sha256']: raise ValueError('Source upload checksum mismatch')
root.mkdir(parents=True,exist_ok=True)
files=0
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as archive:
 for member in archive.getmembers():
  name=Path(member.name)
  if name.is_absolute() or '..' in name.parts: raise ValueError('Invalid source archive path')
  dest=root/name
  if member.isdir(): dest.mkdir(parents=True,exist_ok=True); continue
  if not member.isfile(): raise ValueError('Unsupported source archive entry: '+member.name)
  content=archive.extractfile(member).read()
  if dest.exists():
   if dest.read_bytes()!=content: raise ValueError('Existing source differs from the pinned revision; choose a new repository path: '+str(dest))
  else:
   dest.parent.mkdir(parents=True,exist_ok=True)
   with dest.open('xb') as f: f.write(content)
   dest.chmod(member.mode & 0o777)
  files+=1
links=0
for folder,dirs,files_in_dir in os.walk(bundle/'view',followlinks=True):
 for name in files_in_dir:
  src=Path(folder)/name; dest=root/src.relative_to(bundle/'view')
  if dest.exists(): continue
  dest.parent.mkdir(parents=True,exist_ok=True); dest.symlink_to(src.resolve()); links+=1
required=root/'source/dexverse/dexverse/robot_agents/shadow/retarget/floating_shadow_right.urdf'
if not required.is_file(): raise ValueError('Shadow right-hand assets are missing from the bundle')
probe="import importlib.util,json; names=['isaaclab','dex_retargeting','pinocchio','torch','h5py','scipy','gymnasium','imageio']; print(json.dumps([n for n in names if importlib.util.find_spec(n) is None]))"
check=subprocess.run([str(runtime/'bin/python'),'-c',probe],capture_output=True,text=True,timeout=20)
if check.returncode: raise ValueError('Runtime package check failed: '+check.stderr[-2000:])
missing=json.loads(check.stdout)
if missing: raise ValueError('Runtime packages missing: '+', '.join(missing))
(root/'.skynet-source-revision').write_text(profile['source_revision'])
(root/'.skynet-source-archive-sha256').write_text(payload['archive_sha256'])
print(json.dumps({'status':'READY','source_files_verified':files,'asset_links_added':links,'source_revision':profile['source_revision'],'archive_sha256':payload['archive_sha256'],'detail':'Pinned source, runtime packages and robot assets are ready. GPU availability is checked when a cycle is submitted.'}))
"""
    command = "python3 -c " + shlex.quote(script)
    return json.loads(
        cluster.ssh(profile["gateway"], command, stdin=json.dumps(payload), timeout=55)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "config/capture_pipelines.json",
    )
    parser.add_argument("--pipeline", default="dexverse-shadow-right")
    args = parser.parse_args()
    profile = next(
        (
            p
            for p in json.loads(args.config.read_text())["pipelines"]
            if p["key"] == args.pipeline
        ),
        None,
    )
    if profile is None:
        parser.error("Unknown pipeline")
    print(json.dumps(configure(profile), indent=2))


if __name__ == "__main__":
    main()
