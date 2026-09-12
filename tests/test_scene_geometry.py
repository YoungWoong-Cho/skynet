from types import SimpleNamespace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from skynet_app.adapters.scene_geometry import capture_scene, triangulate


def test_concave_polygon_and_holes_preserve_area():
    points = np.array([[0,0,0], [3,0,0], [3,3,0], [1,1,0], [0,3,0]])
    triangles = np.array(triangulate(points, [5], [0,1,2,3,4])).reshape(-1,3)
    area = sum(np.linalg.norm(np.cross(points[b]-points[a], points[c]-points[a]))/2 for a,b,c in triangles)
    assert area == 6
    assert triangulate(points, [5], [0,1,2,3,4], holes=[0]) == []
    with pytest.raises(ValueError, match="index"):
        triangulate(points, [3], [0,1,20])


def test_real_usd_scene_preserves_rigid_scale_static_meshes_and_instances():
    pytest.importorskip("pxr.Usd")
    from pxr import Usd, UsdGeom, Gf
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    cube = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Cube")
    cube.AddTranslateOp().Set((3,4,5))
    cube.AddScaleOp().Set((2,3,4))
    shape = UsdGeom.Cube.Define(stage, str(cube.GetPath()) + "/visual")
    shape.CreateSizeAttr(0.2);shape.CreateDisplayColorAttr([(1,0,0)])
    UsdGeom.Cube.Define(stage, "/World/envs/env_0/Robot/hand")
    UsdGeom.Cube.Define(stage, "/World/envs/env_1/duplicate")
    hidden = UsdGeom.Cube.Define(stage, "/World/envs/env_0/hidden")
    hidden.CreateVisibilityAttr("invisible")
    mesh = UsdGeom.Mesh.Define(stage, "/World/table")
    mesh.CreatePointsAttr([(0,0,0),(1,0,0),(1,1,0),(0,1,0)])
    mesh.CreateFaceVertexCountsAttr([4]);mesh.CreateFaceVertexIndicesAttr([0,1,2,3])
    mesh.AddTranslateOp().Set((10,0,0))
    prototype = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(prototype, "/model")
    UsdGeom.Sphere.Define(prototype, "/model/visual").CreateRadiusAttr(.1)
    prototype.SetDefaultPrim(prototype.GetPrimAtPath('/model'))
    instance = stage.DefinePrim("/World/instance", "Xform")
    instance.GetReferences().AddReference(prototype.GetRootLayer().identifier)
    instance.SetInstanceable(True)
    asset = lambda path: SimpleNamespace(cfg=SimpleNamespace(prim_path=path))
    scene = SimpleNamespace(stage=stage, env_prim_paths=["/World/envs/env_0"],
                            rigid_objects={"cube":asset("{ENV_REGEX_NS}/Cube")},
                            articulations={"robot":asset("{ENV_REGEX_NS}/Robot")})
    saved = capture_scene(scene)
    assert saved["warnings"] == []
    objects = {item["name"]:item for item in saved["objects"]}
    assert set(objects) == {"cube","static:/World/table","static:/World/instance/visual"}
    assert objects["cube"]["pose"] == [3,4,5,1,0,0,0]
    part = objects["cube"]["parts"][0]
    np.testing.assert_allclose(np.array(part["matrix"]).reshape(4,4), np.diag([2,3,4,1]))
    assert part["size"] == [.2,.2,.2] and part["color"] == [1,0,0]
    table = objects["static:/World/table"]
    assert not table["dynamic"] and len(table["parts"][0]["indices"]) == 6
    assert table["parts"][0]["matrix"][12:15] == [10,0,0]
    assert 'references' not in json.dumps(saved)


def test_large_scene_saved_as_compressed_dataset_not_hdf5_attribute(tmp_path):
    import h5py
    file = Path(__file__).parents[1]/"ops/xr/images.py"
    spec = importlib.util.spec_from_file_location("scene_image_writer", file)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    scene = dict(schema="skynet.scene-geometry/v1", objects=[{"vertices":[.123]*40000}])
    writer = module.ImageWriter(tmp_path, {"action_joint_names":["finger"], "scene_geometry":scene})
    try:
        with h5py.File(writer.temp, "r") as h5:
            assert "scene_geometry" not in json.loads(h5.attrs["metadata"])
            assert h5["scene_geometry"].compression == "gzip"
            assert json.loads(h5["scene_geometry"][:].tobytes()) == scene
    finally:
        writer.discard()
