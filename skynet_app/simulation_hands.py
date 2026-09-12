"""DexVerse adapter bundles built from environment-independent Skynet hands.

Archived v1 bundles remain readable. New adapters pin the canonical hand digest;
changing a simulator or retargeter never changes the physical hand asset.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import threading

from . import hand_bundles
from .hand_bundles import definitions, definition, validate_tree, upload
from .hands import ROOT
from .database import canonical_json

__all__ = ["build", "definitions", "definition", "validate_tree", "upload"]

_BUILD_LOCK = threading.RLock()


def build(robot, library=None, output_root=None):
    directory, hand = hand_bundles.build(
        robot,
        library=library,
        output_root=Path(output_root) / "canonical" if output_root else None,
    )
    runtime_files = {
        name: (ROOT / "ops/xr/hands" / name).read_bytes()
        for name in ("runtime.py", "record.py", "anatomy.py")
    }
    recipe = {
        "hand_digest": hand["digest"],
        "adapter": "dexverse",
        "builder": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {k: hashlib.sha256(v).hexdigest() for k, v in runtime_files.items()},
    }
    digest = hashlib.sha256(canonical_json(recipe).encode()).hexdigest()
    destination = Path(output_root or ROOT / "data/simulation-hands") / robot / digest
    with _BUILD_LOCK:
        if (destination / "manifest.json").is_file():
            return destination, json.loads((destination / "manifest.json").read_text())
        stage = destination.with_name(digest + ".partial")
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        try:
            for name, expected in hand["files"].items():
                src = directory / name
                if hashlib.sha256(src.read_bytes()).hexdigest() != expected["sha256"]:
                    raise ValueError("Canonical hand asset checksum mismatch: " + name)
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, target)
            for side, layout in hand["hands"].items():
                retarget = dict(
                    type="DexPilot",
                    urdf_path="hand.urdf",
                    wrist_link_name=layout["control_frame"],
                    finger_tip_link_names=layout["tips"],
                    target_joint_names=layout["finger_joints"],
                    scaling_factor=1.0,
                    low_pass_alpha=0.8,
                    ignore_mimic_joint=False,
                )
                (stage / ("retarget-" + side + ".json")).write_text(
                    canonical_json({"retargeting": retarget})
                )
            for name, content in runtime_files.items():
                (stage / name).write_bytes(content)
            manifest = dict(
                deepcopy(hand),
                schema="skynet.simulation-hand/v1",
                digest=digest,
                hand_asset={
                    "schema": hand["schema"],
                    "digest": hand["digest"],
                    "source_model_sha256": hand.get("source_model_sha256"),
                    "source_revision": hand["source_revision"],
                },
                environment_adapter="dexverse/v1",
                collision_neighbor_depth=3
                if hand["hand_key"] in {"leap-v1", "inspire-rh56"}
                else 2,
                retargeting_scheme="dexpilot",
                retargeting_mode="dexverse",
            )
            manifest["files"] = {
                str(p.relative_to(stage)): {
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    "size_bytes": p.stat().st_size,
                }
                for p in stage.rglob("*")
                if p.is_file()
            }
            (stage / "manifest.json").write_text(canonical_json(manifest))
            stage.rename(destination)
            return destination, manifest
        finally:
            if stage.exists():
                shutil.rmtree(stage)
