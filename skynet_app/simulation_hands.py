"""Build immutable simulation assets from the same pinned URDFs as the Hands page."""

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import threading
import xml.etree.ElementTree as ET

import numpy as np
from .database import canonical_json
from .hands import HandLibrary, ROOT, parse_urdf, joint_metadata

_BUILD_LOCK = threading.RLock()
WRIST_JOINTS = [
    "skynet_x",
    "skynet_y",
    "skynet_z",
    "skynet_roll",
    "skynet_pitch",
    "skynet_yaw",
]


@lru_cache(maxsize=1)
def definitions():
    entries = json.loads((ROOT / "config/simulation_hands.json").read_text())["hands"]
    library = HandLibrary()
    result = []
    for definition in entries:
        source = library.entry(definition["key"])
        for side in definition["sides"]:
            fields = {"side": side, "s": side[0]}
            result.append(
                dict(
                    definition,
                    side=side,
                    robot="skynet_" + definition["key"].replace("-", "_") + "_" + side,
                    name=source["name"] + " · " + side + " hand",
                    revision=source["revision"],
                    palm=definition["palm"].format(**fields),
                    tips=[tip.format(**fields) for tip in definition["tips"]],
                    alignment_rpy=definition["alignment_rpy"][side],
                )
            )
    return result


def definition(robot):
    found = next((item for item in definitions() if item["robot"] == robot), None)
    if found is None:
        raise ValueError("Unknown imported simulation hand")
    return deepcopy(found)


def safe_name(name):
    return "h_" + re.sub(r"[^A-Za-z0-9_]", "_", name)


def link(xml, name, mass=0.001):
    item = ET.SubElement(xml, "link", name=name)
    inertia = ET.SubElement(item, "inertial")
    ET.SubElement(inertia, "mass", value=str(mass))
    ET.SubElement(
        inertia,
        "inertia",
        ixx="0.000001",
        iyy="0.000001",
        izz="0.000001",
        ixy="0",
        ixz="0",
        iyz="0",
    )
    return item


def joint(xml, name, parent, child, kind="fixed", axis=None, rpy=None):
    item = ET.SubElement(xml, "joint", name=name, type=kind)
    ET.SubElement(item, "parent", link=parent)
    ET.SubElement(item, "child", link=child)
    ET.SubElement(item, "origin", xyz="0 0 0", rpy=" ".join(map(str, rpy or [0, 0, 0])))
    if axis:
        ET.SubElement(item, "axis", xyz=axis)
        bound = "2" if kind == "prismatic" else str(math.pi)
        ET.SubElement(
            item, "limit", lower="-" + bound, upper=bound, effort="30", velocity="5"
        )
    return item


def validate_tree(xml):
    links = xml.findall("link")
    names = [item.get("name") for item in links]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate hand link names")
    children = set()
    for j in xml.findall("joint"):
        parent, child = j.find("parent").get("link"), j.find("child").get("link")
        if parent not in names or child not in names or child in children:
            raise ValueError("Hand URDF has a disconnected or multiply-parented link")
        children.add(child)
    roots = set(names) - children
    if len(roots) != 1:
        raise ValueError("Hand URDF must have exactly one root")
    visited = set()
    pending = list(roots)
    while pending:
        parent = pending.pop()
        if parent in visited:
            raise ValueError("Hand URDF contains a cycle")
        visited.add(parent)
        pending.extend(
            j.find("child").get("link")
            for j in xml.findall("joint")
            if j.find("parent").get("link") == parent
        )
    if visited != set(names):
        raise ValueError("Hand URDF contains unreachable links")
    for item in links:
        inertial = item.find("inertial")
        if inertial is None:
            continue  # Fixed fingertip/sensor frames may have no inertial body.
        mass = float(inertial.find("mass").get("value"))
        i = inertial.find("inertia")
        tensor = np.array(
            [
                [float(i.get("i" + a + b, i.get("i" + b + a, "0"))) for b in "xyz"]
                for a in "xyz"
            ]
        )
        if (
            not math.isfinite(mass)
            or mass < 0
            or not np.isfinite(tensor).all()
            or np.linalg.eigvalsh(tensor).min() < -1e-12
        ):
            raise ValueError(f"Invalid physical inertia for {item.get('name')}")
    joint_metadata(xml)
    return next(iter(roots))


def build(robot, library=None, output_root=None):
    spec = definition(robot)
    library = library or HandLibrary()
    source = library.entry(spec["key"], spec["side"])
    try:
        model = library.model(spec["key"], spec["side"])
    except ValueError as exc:
        raise ValueError(
            f"Cannot prepare {spec['name']}. Open this model in the Hands page and download its assets first. {exc}"
        ) from exc
    if model["revision"] != spec["revision"]:
        raise ValueError("Stored hand revision differs from its simulation recipe")
    directory = library.directory(source, spec["side"])
    raw = (directory / "model.urdf").read_bytes()
    runtime_files = {
        name: (ROOT / "ops/xr/hands" / name).read_bytes()
        for name in ("runtime.py", "record.py", "anatomy.py")
    }
    recipe = dict(
        spec=spec,
        model_sha256=hashlib.sha256(raw).hexdigest(),
        assets=model["files"],
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime={k: hashlib.sha256(v).hexdigest() for k, v in runtime_files.items()},
    )
    digest = hashlib.sha256(canonical_json(recipe).encode()).hexdigest()
    destination = Path(output_root or ROOT / "data/simulation-hands") / robot / digest
    with _BUILD_LOCK:
        if (destination / "manifest.json").is_file():
            return destination, json.loads((destination / "manifest.json").read_text())
        xml = parse_urdf(raw)
        original_root = validate_tree(xml)
        source_links = {item.get("name") for item in xml.findall("link")}
        if (
            not {spec["palm"], *spec["tips"]} <= source_links
            or spec["palm"] != original_root
        ):
            raise ValueError(
                "Hand's declared palm or fingertips do not match the pinned URDF"
            )
        original_joints = joint_metadata(xml)
        all_names = source_links | {j.get("name") for j in xml.findall("joint")}
        mapping = {name: safe_name(name) for name in all_names}
        if len(set(mapping.values())) != len(mapping):
            raise ValueError(
                "Hand names collide after conversion to simulator identifiers"
            )
        for item in xml.findall("link") + xml.findall("joint"):
            item.set("name", mapping[item.get("name")])
        for item in xml.findall("joint"):
            for tag in ("parent", "child"):
                item.find(tag).set("link", mapping[item.find(tag).get("link")])
            mimic = item.find("mimic")
            if mimic is not None:
                mimic.set("joint", mapping[mimic.get("joint")])
        # USD also uses visual/collision names as prim paths (Allegro names contain dots).
        for body in xml.findall("link"):
            for tag in ("visual", "collision"):
                for index, item in enumerate(body.findall(tag)):
                    if item.get("name"):
                        item.set("name", safe_name(item.get("name")) + "_" + str(index))
        for item in xml.findall(".//material"):
            if item.get("name"):
                item.set("name", safe_name(item.get("name")))
        prefix = f"/api/hands/{spec['key']}/{spec['side']}/assets/"
        asset_names = {name: "assets/" + name for name in model["files"]}
        for mesh in xml.findall(".//mesh"):
            url = mesh.get("filename", "")
            if not url.startswith(prefix) or url[len(prefix) :] not in model["files"]:
                raise ValueError("Hand mesh is not in its pinned library manifest")
            name = url[len(prefix) :]
            stem = Path(name).stem
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", stem):
                # Isaac's importer derives mesh prim names from filenames even
                # when the enclosing visual has an explicit, valid name.
                asset_names[name] = (
                    "assets/"
                    + safe_name(stem)
                    + "_"
                    + hashlib.sha256(name.encode()).hexdigest()[:8]
                    + Path(name).suffix
                )
            mesh.set("filename", asset_names[name])
        # One canonical palm frame aligns every source model with DexVerse's Shadow convention.
        link(xml, "skynet_palm")
        joint(
            xml,
            "skynet_alignment",
            "skynet_palm",
            mapping[original_root],
            rpy=spec["alignment_rpy"],
        )
        xml.set("name", robot)
        finger_names = [mapping[j["name"]] for j in original_joints if not j["mimic"]]
        retarget = dict(
            type="DexPilot",
            urdf_path="retarget.urdf",
            wrist_link_name="skynet_palm",
            finger_tip_link_names=[mapping[n] for n in spec["tips"]],
            target_joint_names=finger_names,
            scaling_factor=1.0,
            low_pass_alpha=0.8,
            ignore_mimic_joint=False,
        )
        if spec["retargeting_scheme"] == "vector":
            retarget = dict(
                type="vector",
                urdf_path="retarget.urdf",
                target_joint_names=finger_names,
                target_origin_link_names=["skynet_palm"] * len(spec["tips"]),
                target_task_link_names=[mapping[n] for n in spec["tips"]],
                target_link_human_indices=[
                    [0] * len(spec["tips"]),
                    list(range(4, 4 * len(spec["tips"]) + 1, 4)),
                ],
                scaling_factor=1.0,
                low_pass_alpha=0.8,
                ignore_mimic_joint=False,
            )
        if spec.get("retargeting_mode") == "finger_segments":
            chains = [
                [name.format(side=spec["side"], s=spec["side"][0]) for name in chain]
                for chain in spec["finger_chains"]
            ]
            if len(chains) != 5 or any(
                len(chain) != 4
                or any(n not in source_links for n in chain)
                or chain[-1] != spec["tips"][i]
                for i, chain in enumerate(chains)
            ):
                raise ValueError(
                    "Finger-segment mapping does not match the stored hand links"
                )
            retarget.update(
                normal_delta=0.0001,
                target_origin_link_names=[
                    mapping[n] for chain in chains for n in chain[:-1]
                ],
                target_task_link_names=[
                    mapping[n] for chain in chains for n in chain[1:]
                ],
                target_link_human_indices=[
                    [1 + 4 * f + j for f in range(5) for j in range(3)],
                    [2 + 4 * f + j for f in range(5) for j in range(3)],
                ],
            )
        neutral = {
            mapping[j["name"]]: min(j["upper"], max(j["lower"], 0.0))
            for j in original_joints
            if not j["mimic"]
        }
        stage = destination.with_name(digest + ".partial")
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        try:
            (stage / "retarget.urdf").write_bytes(
                ET.tostring(xml, encoding="utf-8", xml_declaration=True)
            )
            # A fixed world base plus six actuated wrist joints keeps the palm controllable in physics.
            previous = "skynet_base"
            link(xml, previous)
            for i, name in enumerate(WRIST_JOINTS):
                next_link = "skynet_palm" if i == 5 else name + "_link"
                if i != 5:
                    link(xml, next_link)
                joint(
                    xml,
                    name,
                    previous,
                    next_link,
                    "prismatic" if i < 3 else "revolute",
                    ["1 0 0", "0 1 0", "0 0 1", "1 0 0", "0 1 0", "0 0 1"][i],
                )
                previous = next_link
            validate_tree(xml)
            (stage / "simulation.urdf").write_bytes(
                ET.tostring(xml, encoding="utf-8", xml_declaration=True)
            )
            (stage / "retarget.json").write_text(
                canonical_json({"retargeting": retarget})
            )
            for name, expected in model["files"].items():
                original = library.asset(spec["key"], spec["side"], name)
                if (
                    original.stat().st_size != expected["size_bytes"]
                    or hashlib.sha256(original.read_bytes()).hexdigest()
                    != expected["sha256"]
                ):
                    raise ValueError("Stored hand asset checksum changed: " + name)
                dest = stage / asset_names[name]
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, dest)
            for name, content in runtime_files.items():
                (stage / name).write_bytes(content)
            files = {
                str(p.relative_to(stage)): {
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    "size_bytes": p.stat().st_size,
                }
                for p in stage.rglob("*")
                if p.is_file()
            }
            manifest = dict(
                schema="skynet.simulation-hand/v1",
                digest=digest,
                robot=robot,
                hand_key=spec["key"],
                side=spec["side"],
                collision_neighbor_depth=spec.get("collision_neighbor_depth", 2),
                retargeting_scheme=spec["retargeting_scheme"],
                retargeting_mode=spec.get("retargeting_mode", "fingertips"),
                source_revision=spec["revision"],
                name=spec["name"],
                palm=mapping[original_root],
                tips=[mapping[n] for n in spec["tips"]],
                finger_joints=finger_names,
                joint_child_links={
                    j.get("name"): j.find("child").get("link")
                    for j in xml.findall("joint")
                    if j.get("name") in finger_names
                },
                wrist_joints=WRIST_JOINTS,
                neutral=neutral,
                action_dimension=6 + len(finger_names),
                finger_limits={
                    mapping[j["name"]]: [j["lower"], j["upper"]]
                    for j in original_joints
                    if not j["mimic"]
                },
                source_names=mapping,
                source_assets=asset_names,
                files=files,
                mimic_joints=[j for j in joint_metadata(xml) if j["mimic"]],
                physics_note="Initial simulation gains: fingers 10 N m/rad, damping 0.2, effort 2 N m. These are simulation tuning values, not hardware ratings. Headset validation is pending.",
            )
            (stage / "manifest.json").write_text(canonical_json(manifest))
            stage.rename(destination)
            return destination, manifest
        finally:
            if stage.exists():
                shutil.rmtree(stage)


def upload(directory, remote_root, transport, gateway):
    """Transfer one content-addressed bundle once; never overwrite an active model."""
    import base64
    import io
    import shlex
    import tarfile

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    remote = remote_root + "/hands/" + manifest["robot"] + "/" + manifest["digest"]
    transport._remote_path(remote)
    marker = transport.ssh(
        gateway,
        "if test -f "
        + shlex.quote(remote + "/READY")
        + "; then cat "
        + shlex.quote(remote + "/READY")
        + "; fi",
        timeout=10,
    ).strip()
    if marker == manifest["digest"]:
        return remote
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in sorted([*manifest["files"], "manifest.json"]):
            archive.add(directory / name, arcname=name, recursive=False)
    raw = buffer.getvalue()
    if len(raw) > 64 * 1024 * 1024:
        raise ValueError("Compressed hand bundle exceeds the 64 MB transfer limit")
    request = dict(
        root=remote,
        digest=manifest["digest"],
        sha256=hashlib.sha256(raw).hexdigest(),
        archive=base64.b64encode(raw).decode(),
    )
    script = """import base64,hashlib,io,json,os,shutil,sys,tarfile,tempfile
from pathlib import Path
p=json.load(sys.stdin); root=Path(p['root']); root.parent.mkdir(parents=True,exist_ok=True)
raw=base64.b64decode(p['archive'],validate=True)
if hashlib.sha256(raw).hexdigest()!=p['sha256']: raise ValueError('Hand archive checksum mismatch')
stage=Path(tempfile.mkdtemp(prefix='.hand-',dir=root.parent))
try:
 total=0
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
  for member in archive:
   name=Path(member.name); total+=member.size
   if name.is_absolute() or '..' in name.parts or not member.isfile() or total>200*1024*1024: raise ValueError('Invalid hand archive')
   dest=stage/name; dest.parent.mkdir(parents=True,exist_ok=True)
   with dest.open('xb') as f: shutil.copyfileobj(archive.extractfile(member),f)
 m=json.loads((stage/'manifest.json').read_text())
 if m['digest']!=p['digest']: raise ValueError('Hand manifest identity mismatch')
 for name,expected in m['files'].items():
  f=(stage/name).resolve()
  if not f.is_relative_to(stage.resolve()) or not f.is_file(): raise ValueError('Hand asset missing: '+name)
  if f.stat().st_size!=expected['size_bytes'] or hashlib.sha256(f.read_bytes()).hexdigest()!=expected['sha256']: raise ValueError('Hand asset checksum mismatch: '+name)
 (stage/'READY').write_text(p['digest'])
 if root.exists():
  if (root/'READY').read_text()!=p['digest']: raise ValueError('Conflicting hand bundle; refusing replacement')
 else: stage.rename(root)
 print(json.dumps({'root':str(root),'digest':p['digest']}))
finally:
 if stage.exists(): shutil.rmtree(stage)
"""
    transport.ssh(
        gateway,
        "python3 -c " + shlex.quote(script),
        stdin=json.dumps(request),
        timeout=60,
    )
    return remote
