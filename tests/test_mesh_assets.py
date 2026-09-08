import json
import struct

import numpy as np
import pytest

from skynet_app.mesh_assets import bake_gltf_nodes


def glb(doc, binary):
    metadata = json.dumps(doc).encode()
    metadata += b" " * (-len(metadata) % 4)
    return (
        struct.pack("<4sII", b"glTF", 2, 28 + len(metadata) + len(binary))
        + struct.pack("<I4s", len(metadata), b"JSON")
        + metadata
        + struct.pack("<I4s", len(binary), b"BIN\x00")
        + binary
    )


def fixture():
    binary = np.array([[1, 0, 0, 1, 0, 0], [0, 1, 0, 1, 0, 0]], dtype="<f4").tobytes()
    doc = dict(
        asset={"version": "2.0"},
        scene=0,
        scenes=[{"nodes": [0]}],
        nodes=[
            dict(
                translation=[2, 3, 4], rotation=[0, 0, 2**-0.5, 2**-0.5], children=[1]
            ),
            dict(translation=[1, 0, 0], mesh=0),
        ],
        buffers=[{"byteLength": len(binary)}],
        bufferViews=[
            dict(buffer=0, byteOffset=0, byteLength=len(binary), byteStride=24)
        ],
        accessors=[
            dict(bufferView=0, byteOffset=0, componentType=5126, count=2, type="VEC3"),
            dict(bufferView=0, byteOffset=12, componentType=5126, count=2, type="VEC3"),
        ],
        materials=[
            dict(
                name="Original white plastic",
                pbrMetallicRoughness={"baseColorFactor": [1, 1, 1, 1]},
            )
        ],
        meshes=[
            dict(primitives=[dict(attributes={"POSITION": 0, "NORMAL": 1}, material=0)])
        ],
    )
    return doc, binary


def test_baked_parent_and_mesh_node_transforms_keep_vertices_at_original_world_positions():
    doc, binary = fixture()
    out = bake_gltf_nodes(glb(doc, binary))
    size = struct.unpack_from("<I", out, 12)[0]
    result = json.loads(out[20 : 20 + size])
    data = out[28 + size :]
    node = result["nodes"][1]
    primitive = result["meshes"][node["mesh"]]["primitives"][0]
    assert result["materials"] == doc["materials"]
    assert primitive["material"] == 0
    assert all(
        not set(n) & {"matrix", "translation", "rotation", "scale"}
        for n in result["nodes"]
    )
    for semantic, expected in [
        ("POSITION", [[2, 5, 4], [1, 4, 4]]),
        ("NORMAL", [[0, 1, 0], [0, 1, 0]]),
    ]:
        accessor = result["accessors"][primitive["attributes"][semantic]]
        view = result["bufferViews"][accessor["bufferView"]]
        values = np.frombuffer(
            data, dtype="<f4", count=6, offset=view["byteOffset"]
        ).reshape(2, 3)
        np.testing.assert_allclose(values, expected, atol=1e-6)


@pytest.mark.parametrize(
    "change", ["animation", "skin", "shared_parent", "reflection", "sparse"]
)
def test_unsupported_meshes_fail_instead_of_silently_losing_geometry(change):
    doc, binary = fixture()
    if change == "animation":
        doc["animations"] = [{}]
    if change == "skin":
        doc["skins"] = [{}]
    if change == "shared_parent":
        doc["scenes"][0]["nodes"].append(1)
    if change == "reflection":
        doc["nodes"][1]["scale"] = [-1, 1, 1]
    if change == "sparse":
        doc["accessors"][0]["sparse"] = {}
    with pytest.raises(ValueError):
        bake_gltf_nodes(glb(doc, binary))
