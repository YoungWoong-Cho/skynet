"""Expand the pinned official Shadow Xacro without requiring a ROS installation.

Run with: uv run --no-project --with xacro==2.1.1 python ops/hands/generate_shadow.py
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

import xacro

ROOT = Path(__file__).resolve().parents[2]


def download(url):
    request = urllib.request.Request(
        url, headers={"User-Agent": "Skynet-Hand-Library/1"}
    )
    with urllib.request.urlopen(request, timeout=35) as response:
        return response.read()


def main():
    catalog_path = ROOT / "config/hands.json"
    catalog = json.loads(catalog_path.read_text())
    hand = next(h for h in catalog["hands"] if h["key"] == "shadow")
    repo, revision = hand["repository"], hand["revision"]
    tree = json.loads(
        download(
            f"https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1"
        )
    )
    if tree.get("truncated"):
        raise ValueError("Source tree is incomplete; no model was generated")
    paths = [
        entry["path"]
        for entry in tree["tree"]
        if entry["type"] == "blob"
        and entry["path"].startswith("sr_description/")
        and entry["path"].endswith(".xacro")
    ]
    with tempfile.TemporaryDirectory(prefix="skynet-shadow-") as folder:
        root = Path(folder)

        def fetch(path):
            raw = download(
                f"https://raw.githubusercontent.com/{repo}/{revision}/{path}"
            )
            dest = root / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(
                raw.replace(
                    b"$(find sr_description)", str(root / "sr_description").encode()
                )
            )

        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(fetch, paths))
        for side in ("right", "left"):
            relative = hand["sides"][side]
            doc = xacro.process_file(
                str(root / relative),
                mappings={
                    "side": side,
                    "hand_type": "hand_e",
                    "hand_version": "E3M5",
                    "tip_sensors": "pst",
                },
            )
            raw = (
                doc.toprettyxml(indent="  ")
                .replace(str(root / relative), relative)
                .encode()
            )
            dest = ROOT / f"config/hand_models/shadow/{side}.urdf"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            hand["generated_urdfs"][side] = {
                "path": str(dest.relative_to(ROOT)),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
    print("Generated both Shadow E3M5/PST models from", revision)


if __name__ == "__main__":
    main()
