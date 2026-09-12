"""Portable, CPU-only geometry for recorded observations and policy targets."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

MAX_VIEWER_BYTES = 64_000_000


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    length = np.linalg.norm(axis)
    if not np.isfinite(length) or length == 0:
        raise ValueError("Invalid joint axis")
    x, y, z = axis / length
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def pose_matrix(pose):
    value = np.asarray(pose, dtype=float).reshape(-1)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("Recorded root pose must contain XYZ and a WXYZ quaternion")
    norm = np.linalg.norm(value[3:])
    if norm < 1e-8:
        raise ValueError("Invalid root quaternion")
    w, x, y, z = value[3:] / norm
    result = np.eye(4)
    result[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    result[:3, 3] = value[:3]
    return result


class HandKinematics:
    """Use the captured robot's URDF and named joint order; never guess a layout."""

    def __init__(self, xml, joint_names, wrist_names=()):
        root = ET.fromstring(xml)
        self.joint_names = list(joint_names)
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("Duplicate recorded joint names")
        self.joints = []
        links = {item.get("name") for item in root.findall("link")}
        children = set()
        for item in root.findall("joint"):
            kind = item.get("type")
            if kind not in {"fixed", "revolute", "continuous", "prismatic"}:
                raise ValueError("Unsupported recorded joint type: " + str(kind))
            origin = item.find("origin")
            xyz = [float(x) for x in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
            rpy = [float(x) for x in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
            transform = np.eye(4)
            transform[:3, 3] = xyz
            transform[:3, :3] = rotation([0, 0, 1], rpy[2]) @ rotation([0, 1, 0], rpy[1]) @ rotation([1, 0, 0], rpy[0])
            axis = item.find("axis")
            mimic = item.find("mimic")
            parent, child = item.find("parent").get("link"), item.find("child").get("link")
            if parent not in links or child not in links or child in children:
                raise ValueError("Invalid hand tree")
            children.add(child)
            self.joints.append(dict(name=item.get("name"), kind=kind, parent=parent, child=child,
                                    origin=transform, axis=[float(x) for x in (axis.get("xyz", "1 0 0") if axis is not None else "1 0 0").split()],
                                    mimic=dict(mimic.attrib) if mimic is not None else None))
        roots = links - children
        if len(roots) != 1:
            raise ValueError("Hand tree must have one root")
        self.root = roots.pop()
        required = {j["name"] for j in self.joints if j["kind"] != "fixed" and not j["mimic"]}
        if not required.issubset(joint_names):
            raise ValueError("Recorded joint names differ from the hand model")
        selected = [j for j in self.joints if j["name"] not in set(wrist_names)]
        self.links = list(dict.fromkeys(n for j in selected for n in (j["parent"], j["child"])))
        self.edges = [[self.links.index(j["parent"]), self.links.index(j["child"])] for j in selected]

    def points(self, values, root_pose):
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.shape != (len(self.joint_names),) or not np.isfinite(values).all():
            raise ValueError("Invalid recorded hand pose")
        angles = dict(zip(self.joint_names, values))
        pending = list(self.joints)
        transforms = {self.root: pose_matrix(root_pose)}
        while pending:
            progressed = False
            for joint in pending[:]:
                if joint["parent"] not in transforms:
                    continue
                value = angles.get(joint["name"], 0.0)
                mimic = joint["mimic"]
                if mimic:
                    if mimic["joint"] not in angles:
                        continue
                    value = angles[mimic["joint"]] * float(mimic.get("multiplier", 1)) + float(mimic.get("offset", 0))
                    angles[joint["name"]] = value
                motion = np.eye(4)
                if joint["kind"] == "prismatic":
                    motion[:3, 3] = np.asarray(joint["axis"]) * value
                elif joint["kind"] != "fixed":
                    motion[:3, :3] = rotation(joint["axis"], value)
                transforms[joint["child"]] = transforms[joint["parent"]] @ joint["origin"] @ motion
                pending.remove(joint)
                progressed = True
            if not progressed:
                raise ValueError("Disconnected or cyclic hand model")
        return np.stack([transforms[n][:3, 3] for n in self.links]).round(7).tolist()


def recorded_urdf(capture, repository, hand_bundle=None):
    if capture.get("kinematics_urdf"):
        import hashlib
        xml = capture["kinematics_urdf"]
        if not isinstance(xml, str) or len(xml) > 4_000_000 or hashlib.sha256(xml.encode()).hexdigest() != capture.get("kinematics_sha256"):
            raise ValueError("Recorded hand geometry checksum changed")
        return xml
    if hand_bundle:
        path = Path(hand_bundle) / "simulation.urdf"
    elif capture.get("robot") in {"floating_shadow_right", "floating_shadow_left", "floating_shadow_bimanual"}:
        path = Path(repository) / "source/dexverse/dexverse/robot_agents/shadow/retarget" / (capture["robot"] + ".urdf")
    else:
        raise ValueError("The recorded hand URDF location is unavailable")
    if not path.is_file() or path.stat().st_size > 4_000_000:
        raise ValueError("The recorded hand URDF is unavailable")
    return path.read_text()


def camera_layout(cameras, *, labels=False):
    names = list(cameras)
    width = max((cameras[n]["width"] for n in names), default=256)
    height = max((cameras[n]["height"] for n in names), default=256)
    columns = min(3, len(names)) or 1
    return [{"id": name, "label": name.removeprefix("scene_").replace("_", " ").title(),
             "rect": [(i % columns) * width + (width - cameras[name]["width"]) // 2,
                      (i // columns) * (height + (32 if labels else 0)) + (32 if labels else 0) + (height - cameras[name]["height"]) // 2,
                      cameras[name]["width"], cameras[name]["height"]],
             **cameras[name]} for i, name in enumerate(names)]


def replay_urdf(xml):
    """Keep the captured transforms without exposing cluster mesh URLs to browsers."""
    root = ET.fromstring(xml)
    for link in root.findall("link"):
        for child in list(link):
            if child.tag in {"visual", "collision", "inertial"}:
                link.remove(child)
    for material in root.findall("material"):
        root.remove(material)
    return ET.tostring(root, encoding="unicode")
