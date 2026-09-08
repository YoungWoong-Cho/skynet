"""Prepare static GLB meshes without relying on an importer's node transforms."""

from copy import deepcopy
import json
import struct

import numpy as np


def bake_gltf_nodes(raw):
    """Bake scene transforms into vertices, preserving materials and textures.

    Isaac's URDF GLB conversion misplaces meshes with nonidentity root TRS.
    Baking in glTF coordinates avoids that conversion defect. The source asset
    stays intact; only the immutable simulation copy is normalized.
    """
    magic, version, length = struct.unpack_from("<4sII", raw)
    if magic != b"glTF" or version != 2 or length != len(raw):
        raise ValueError("Invalid GLB hand mesh")
    chunks, offset = {}, 12
    while offset < len(raw):
        size, kind = struct.unpack_from("<I4s", raw, offset)
        offset += 8
        chunks[kind] = raw[offset : offset + size]
        offset += size
    doc = json.loads(chunks[b"JSON"])
    if doc.get("animations") or doc.get("skins"):
        raise ValueError("Animated/skinned GLB hand meshes are unsupported")
    if len(doc.get("buffers", [])) != 1 or "uri" in doc["buffers"][0]:
        raise ValueError("Hand GLB must contain one embedded buffer")
    original = chunks[b"BIN\x00"]
    binary = bytearray(original)

    def transform(node):
        if "matrix" in node:
            matrix = np.array(node["matrix"], dtype=float).reshape(4, 4).T
        else:
            x, y, z, w = node.get("rotation", [0, 0, 0, 1])
            rotation = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )
            matrix = np.eye(4)
            matrix[:3, :3] = rotation @ np.diag(node.get("scale", [1, 1, 1]))
            matrix[:3, 3] = node.get("translation", [0, 0, 0])
        if not np.isfinite(matrix).all() or np.linalg.det(matrix[:3, :3]) <= 0:
            raise ValueError("Invalid or reflected GLB node transform")
        return matrix

    def accessor(index, semantic, matrix):
        source = doc["accessors"][index]
        size = 4 if semantic == "TANGENT" else 3
        if (
            source["componentType"] != 5126
            or source["type"] != f"VEC{size}"
            or "sparse" in source
            or source.get("normalized")
        ):
            raise ValueError("Unsupported GLB vertex accessor")
        view = doc["bufferViews"][source["bufferView"]]
        values = np.ndarray(
            (source["count"], size),
            dtype="<f4",
            buffer=original,
            offset=view.get("byteOffset", 0) + source.get("byteOffset", 0),
            strides=(view.get("byteStride", size * 4), 4),
        ).astype(float)
        if semantic == "POSITION":
            values[:, :3] = values[:, :3] @ matrix[:3, :3].T + matrix[:3, 3]
        else:
            basis = (
                np.linalg.inv(matrix[:3, :3])
                if semantic == "NORMAL"
                else matrix[:3, :3].T
            )
            values[:, :3] = values[:, :3] @ basis
            lengths = np.linalg.norm(values[:, :3], axis=1, keepdims=True)
            values[:, :3] /= np.maximum(lengths, 1e-12)
        if not np.isfinite(values).all():
            raise ValueError("GLB contains nonfinite vertices")
        binary.extend(b"\x00" * (-len(binary) % 4))
        offset = len(binary)
        data = values.astype("<f4").tobytes()
        binary.extend(data)
        doc["bufferViews"].append(
            dict(buffer=0, byteOffset=offset, byteLength=len(data))
        )
        item = dict(
            componentType=5126,
            count=len(values),
            type=f"VEC{size}",
            bufferView=len(doc["bufferViews"]) - 1,
        )
        if semantic == "POSITION":
            item.update(min=values.min(0).tolist(), max=values.max(0).tolist())
        doc["accessors"].append(item)
        return len(doc["accessors"]) - 1

    visited = set()

    def visit(index, parent):
        if index in visited:
            raise ValueError("GLB node has multiple parents or a cycle")
        visited.add(index)
        node = doc["nodes"][index]
        matrix = parent @ transform(node)
        if "mesh" in node:
            mesh = deepcopy(doc["meshes"][node["mesh"]])
            for primitive in mesh["primitives"]:
                if primitive.get("targets") or primitive.get("extensions"):
                    raise ValueError(
                        "Compressed or morphing GLB hand meshes are unsupported"
                    )
                for semantic in ("POSITION", "NORMAL", "TANGENT"):
                    if semantic in primitive["attributes"]:
                        primitive["attributes"][semantic] = accessor(
                            primitive["attributes"][semantic], semantic, matrix
                        )
            doc["meshes"].append(mesh)
            node["mesh"] = len(doc["meshes"]) - 1
        for child in node.get("children", []):
            visit(child, matrix)
        for key in ("matrix", "translation", "rotation", "scale"):
            node.pop(key, None)

    if len(doc.get("scenes", [])) != 1:
        raise ValueError("Hand GLB must contain one static scene")
    for root in doc["scenes"][0]["nodes"]:
        visit(root, np.eye(4))
    doc["buffers"][0]["byteLength"] = len(binary)
    metadata = json.dumps(doc, separators=(",", ":")).encode()
    metadata += b" " * (-len(metadata) % 4)
    binary.extend(b"\x00" * (-len(binary) % 4))
    return (
        struct.pack("<4sII", b"glTF", 2, 28 + len(metadata) + len(binary))
        + struct.pack("<I4s", len(metadata), b"JSON")
        + metadata
        + struct.pack("<I4s", len(binary), b"BIN\x00")
        + binary
    )
