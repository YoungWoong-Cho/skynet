"""Pinned hand descriptions, verified local mesh storage, and versioned poses."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import posixpath
import re
import shutil
import threading
import time
import urllib.request
import uuid
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=32)
def _read_manifest(path: Path, modified_ns: int):
    """Reuse immutable metadata across a model's many asset requests."""
    return json.loads(path.read_text())


def parse_urdf(raw: bytes):
    # Several official models place a copyright comment before the declaration.
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ValueError("URDF document types and entities are unsupported")
    xml = ET.fromstring(re.sub(rb"<\?xml[^>]*\?>", b"", raw))
    if xml.tag != "robot":
        raise ValueError("Description must contain a URDF robot")
    return xml


def mesh_path(description, reference, packages):
    if reference.startswith("package://"):
        package, _, relative = reference[10:].partition("/")
        if package not in packages:
            raise ValueError(f"Unconfigured URDF package: {package}")
        path = posixpath.join(packages[package], relative)
    elif "://" in reference or reference.startswith("/"):
        raise ValueError("Mesh paths must stay inside the pinned source repository")
    else:
        path = posixpath.join(posixpath.dirname(description), reference)
    path = posixpath.normpath(path)
    if path.startswith("../") or path in (".", "..") or "\\" in path:
        raise ValueError("Mesh path leaves its source repository")
    return path


def joint_metadata(xml):
    joints = []
    for joint in xml.findall("joint"):
        kind = joint.get("type")
        if kind == "fixed":
            continue
        if kind not in ("revolute", "continuous", "prismatic"):
            raise ValueError(f"Unsupported joint type: {kind}")
        limits = joint.find("limit")
        if kind != "continuous" and (
            limits is None or not {"lower", "upper"} <= set(limits.attrib)
        ):
            raise ValueError(f"Joint {joint.get('name')} is missing limits")
        lower, upper = (
            (-math.pi, math.pi)
            if kind == "continuous"
            else (float(limits.get("lower")), float(limits.get("upper")))
        )
        if not all(math.isfinite(v) for v in (lower, upper)) or lower > upper:
            raise ValueError("Invalid joint limits")
        mimic = joint.find("mimic")
        joints.append(
            dict(
                name=joint.get("name"),
                type=kind,
                lower=lower,
                upper=upper,
                mimic=dict(mimic.attrib) if mimic is not None else None,
            )
        )
    names = {j["name"] for j in joints}
    for joint in joints:
        seen = {joint["name"]}
        current = joint
        while current["mimic"]:
            parent = current["mimic"].get("joint")
            if parent not in names or parent in seen:
                raise ValueError("Invalid or circular mimic joint relationship")
            seen.add(parent)
            current = next(j for j in joints if j["name"] == parent)
    return joints


class HandLibrary:
    def __init__(self, root=None, config=None):
        self.root = Path(root or ROOT / "data/hands")
        self.catalog = json.loads(
            Path(config or ROOT / "config/hands.json").read_text()
        )["hands"]
        self.lock = threading.RLock()
        self.jobs = {}
        self.executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="hand-install"
        )

    def entry(self, key, side=None):
        hand = next((h for h in self.catalog if h["key"] == key), None)
        if hand is None:
            raise ValueError("Unknown hand model")
        if side is not None and side not in hand["sides"]:
            raise ValueError("This hand side is not supplied by the source")
        return hand

    def directory(self, hand, side):
        return self.root / hand["key"] / hand["revision"] / side

    def model(self, key, side):
        hand = self.entry(key, side)
        path = self.directory(hand, side) / "manifest.json"
        if not path.is_file():
            raise ValueError("Download this model before opening it")
        try:
            model = _read_manifest(path, path.stat().st_mtime_ns)
        except (OSError, ValueError) as exc:
            raise ValueError(
                "Stored model metadata is unreadable; restore its manifest"
            ) from exc
        return dict(model, preview_rotation=hand.get("preview_rotation", [0, 0, 0]))

    def list(self):
        rows = []
        with self.lock:
            for hand in self.catalog:
                row = dict(hand, variants={})
                for side in hand["sides"]:
                    error = hand.get("unsupported_sides", {}).get(side)
                    state = (
                        {"state": "UNSUPPORTED", "error": error}
                        if error
                        else {"state": "NOT_DOWNLOADED"}
                    )
                    if (
                        not error
                        and (self.directory(hand, side) / "manifest.json").is_file()
                    ):
                        model = self.model(hand["key"], side)
                        state = {
                            "state": "READY",
                            "mesh_count": model["mesh_count"],
                            "joint_count": len(model["joints"]),
                            "size_bytes": model["size_bytes"],
                        }
                    row["variants"][side] = dict(
                        self.jobs.get((hand["key"], side), state)
                    )
                rows.append(row)
        return rows

    def start_install(self, key, side):
        hand = self.entry(key, side)
        error = hand.get("unsupported_sides", {}).get(side)
        if error:
            raise ValueError(error)
        with self.lock:
            current = self.jobs.get((key, side), {})
            if current.get("state") == "DOWNLOADING":
                return current
            if (self.directory(hand, side) / "manifest.json").is_file():
                return {"state": "READY"}
            self.jobs[key, side] = {
                "state": "DOWNLOADING",
                "detail": "Downloading the pinned description and meshes",
            }
            self.executor.submit(self._install_job, key, side)
            return dict(self.jobs[key, side])

    def _install_job(self, key, side):
        try:
            self.install(key, side)
            with self.lock:
                self.jobs.pop((key, side), None)
        except Exception as exc:
            with self.lock:
                self.jobs[key, side] = {"state": "FAILED", "error": str(exc)}

    @staticmethod
    def _download(url):
        request = urllib.request.Request(
            url, headers={"User-Agent": "Skynet-Hand-Library/1"}
        )
        with urllib.request.urlopen(request, timeout=35) as response:
            data = response.read(32 * 1024 * 1024 + 1)
        if len(data) > 32 * 1024 * 1024:
            raise ValueError("A model asset exceeds the 32 MB download limit")
        return data

    def install(self, key, side):
        hand = self.entry(key, side)
        if side in hand.get("unsupported_sides", {}):
            raise ValueError(hand["unsupported_sides"][side])
        destination = self.directory(hand, side)
        if (destination / "manifest.json").is_file():
            return self.model(key, side)
        stage = destination.with_name(side + ".partial-" + uuid.uuid4().hex)
        stage.mkdir(parents=True)
        files = {}
        total = 0
        repo, revision = hand["repository"], hand["revision"]

        def fetch(path):
            nonlocal total
            if path in files:
                return (stage / "assets" / path).read_bytes()
            try:
                raw = self._download(
                    f"https://raw.githubusercontent.com/{repo}/{revision}/{path}"
                )
            except Exception as exc:
                raise ValueError(
                    f"Could not download source asset {path}: {exc}"
                ) from exc
            if raw.startswith(b"version https://git-lfs.github.com/spec/v1"):
                match = re.search(rb"oid sha256:([a-f0-9]{64})\nsize ([0-9]+)", raw)
                if not match or int(match[2]) > 32 * 1024 * 1024:
                    raise ValueError("Invalid or oversized Git LFS model asset")
                raw = self._download(
                    f"https://media.githubusercontent.com/media/{repo}/{revision}/{path}"
                )
                if (
                    len(raw) != int(match[2])
                    or hashlib.sha256(raw).hexdigest() != match[1].decode()
                ):
                    raise ValueError("Git LFS model checksum mismatch")
            total += len(raw)
            if total > 160 * 1024 * 1024:
                raise ValueError("The model exceeds the 160 MB storage limit")
            dest = stage / "assets" / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            files[path] = {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
            return raw

        try:
            raw = fetch(hand["sides"][side])
            generated = hand.get("generated_urdfs", {}).get(side)
            if generated:
                raw = (ROOT / generated["path"]).read_bytes()
                if hashlib.sha256(raw).hexdigest() != generated["sha256"]:
                    raise ValueError("Generated hand description checksum mismatch")
            xml = parse_urdf(raw)
            fetch(hand["license_path"])
            for visual in xml.findall(".//visual/geometry/mesh"):
                if Path(visual.get("filename", "")).suffix.lower() not in (
                    ".stl",
                    ".dae",
                    ".glb",
                ):
                    raise ValueError(
                        "Unsupported visual mesh format; supported formats are STL, DAE, and GLB"
                    )
            mesh_paths = set()
            for mesh in xml.findall(".//mesh"):
                path = mesh_path(
                    hand["sides"][side], mesh.get("filename", ""), hand["packages"]
                )
                # Only visual geometry is displayed, but preserve collision meshes for a reusable URDF.
                data = fetch(path)
                mesh_paths.add(path)
                if path.lower().endswith(".dae"):
                    collada = ET.fromstring(data)
                    for image in collada.findall(
                        ".//{*}library_images/{*}image/{*}init_from"
                    ):
                        if image.text:
                            fetch(mesh_path(path, image.text.strip(), {}))
                mesh.set("filename", f"/api/hands/{key}/{side}/assets/{path}")
            joints = joint_metadata(xml)
            model = {
                "schema": "skynet.hand-model/v1",
                "key": key,
                "side": side,
                "revision": revision,
                "source_url": f"https://github.com/{repo}/blob/{revision}/{hand['sides'][side]}",
                "license_url": f"/api/hands/{key}/{side}/assets/{hand['license_path']}",
                "urdf_url": f"/api/hands/{key}/{side}/urdf",
                "joints": joints,
                "mesh_count": len(mesh_paths),
                "size_bytes": total,
                "files": files,
                "installed_at": time.time(),
            }
            (stage / "model.urdf").write_bytes(
                ET.tostring(xml, encoding="utf-8", xml_declaration=True)
            )
            (stage / "manifest.json").write_text(json.dumps(model, indent=2))
            os.rename(stage, destination)
            return self.model(key, side)
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def asset(self, key, side, path):
        model = self.model(key, side)
        if path not in model["files"]:
            raise ValueError("Asset is not part of this installed model")
        root = self.directory(self.entry(key, side), side) / "assets"
        candidate = (root / path).resolve()
        if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
            raise ValueError("Model asset is unavailable")
        return candidate

    def poses(self, key, side):
        self.entry(key, side)
        folder = self.root / "poses" / key / side
        return sorted(
            (json.loads(p.read_text()) for p in folder.glob("*.json")),
            key=lambda p: p["created_at"],
            reverse=True,
        )

    def validate_pose(self, key, side, values, revision):
        model = self.model(key, side)
        if revision != model["revision"]:
            raise ValueError(
                "Pose refers to a different model revision; reload the model"
            )
        expected = {j["name"]: j for j in model["joints"] if not j["mimic"]}
        if set(values) != set(expected):
            raise ValueError(
                "Pose must include exactly the independent joints in this model"
            )
        for joint, value in values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("Joint values must be finite numbers")
            limits = expected[joint]
            if not limits["lower"] - 1e-8 <= value <= limits["upper"] + 1e-8:
                raise ValueError(f"{joint} is outside its joint limits")
        return model

    def export_pose(self, key, side, values, revision):
        self.validate_pose(key, side, values, revision)
        return dict(
            schema="skynet.hand-pose/v1",
            key=key,
            side=side,
            revision=revision,
            joints=values,
            units="radians; prismatic joints in meters",
        )

    def save_pose(self, key, side, name, values, revision):
        self.validate_pose(key, side, values, revision)
        name = name.strip()
        if not name or len(name) > 80:
            raise ValueError("Give the pose a name of 1–80 characters")
        pose = dict(
            id=uuid.uuid4().hex,
            key=key,
            side=side,
            name=name,
            revision=revision,
            joints=values,
            created_at=time.time(),
            units="radians; prismatic joints in meters",
        )
        folder = self.root / "poses" / key / side
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / (pose["id"] + ".json")).open("x") as stream:
            json.dump(pose, stream, allow_nan=False)
        return pose
