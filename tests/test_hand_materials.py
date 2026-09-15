"""URDF material resolution and real USD binding precedence regressions."""

from pathlib import Path
import runpy

import pytest

RUNTIME = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "ops/xr/hands/runtime.py")
)


def write_urdf(tmp_path, body):
    path = tmp_path / "hand.urdf"
    path.write_text('<robot name="hand">' + body + "</robot>")
    return path


def test_named_inline_transparent_and_mixed_visual_colors(tmp_path):
    path = write_urdf(
        tmp_path,
        """
      <material name="gray"><color rgba="0.2 0.2 0.2 1"/></material>
      <link name="named"><visual><material name="gray"/></visual></link>
      <link name="inline"><visual><material name="gray"><color rgba="1 0 0 0.5"/></material></visual></link>
      <link name="transparent"><visual><material><color rgba="0 0 0 0"/></material></visual></link>
      <link name="same"><visual><material name="gray"/></visual><visual><material name="gray"/></visual></link>
      <link name="unspecified"><visual/></link>
      <link name="mixed"><visual><material name="gray"/></visual><visual/></link>
      <link name="different"><visual><material name="gray"/></visual><visual><material><color rgba="1 0 0 1"/></material></visual></link>
    """,
    )
    assert RUNTIME["uniform_visual_colors"](path) == {
        "named": (0.2, 0.2, 0.2, 1),
        "inline": (1, 0, 0, 0.5),
        "transparent": (0, 0, 0, 0),
        "same": (0.2, 0.2, 0.2, 1),
        "unspecified": None,
    }


@pytest.mark.parametrize("rgba", ["nan 0 0 1", "0 0 0 2", "0 0 1", ""])
def test_invalid_source_color_fails(tmp_path, rgba):
    path = write_urdf(
        tmp_path,
        f'<link name="thumb"><visual><material><color rgba="{rgba}"/></material></visual></link>',
    )
    with pytest.raises(ValueError, match="Invalid URDF material color"):
        RUNTIME["uniform_visual_colors"](path)


def test_source_material_overrides_transparent_instance_without_changing_physics(
    tmp_path,
):
    pytest.importorskip("pxr.Usd")
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    usd = str(tmp_path / "hand.usda")
    stage = Usd.Stage.CreateNew(usd)
    root = UsdGeom.Xform.Define(stage, "/Hand").GetPrim()
    stage.SetDefaultPrim(root)
    prototype = UsdGeom.Xform.Define(stage, "/Cad").GetPrim()
    mesh = UsdGeom.Cube.Define(stage, "/Cad/mesh").GetPrim()
    cad = UsdShade.Material.Define(stage, "/Cad/Transparent")
    shader = UsdShade.Shader.Define(stage, "/Cad/Transparent/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0)
    cad.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh).Bind(cad)
    for name in ("thumb", "glass"):
        body = UsdGeom.Xform.Define(stage, "/Hand/" + name).GetPrim()
        UsdPhysics.MassAPI.Apply(body).CreateMassAttr(0.04)
        UsdGeom.Xformable(body).AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))
        visual = UsdGeom.Xform.Define(stage, "/Hand/" + name + "/visuals").GetPrim()
        visual.GetReferences().AddInternalReference(prototype.GetPath())
        visual.SetInstanceable(True)
        UsdGeom.Cube.Define(stage, "/Hand/" + name + "/collisions/mesh")
    stage.GetRootLayer().Save()
    urdf = write_urdf(
        tmp_path,
        """
      <link name="thumb"><visual><material><color rgba=".2 .2 .2 1"/></material></visual></link>
      <link name="glass"><visual><material><color rgba="1 0 0 .3"/></material></visual></link>
    """,
    )
    RUNTIME["configure_visual_materials"](urdf, usd)
    first = stage.GetRootLayer().ExportToString()
    RUNTIME["configure_visual_materials"](urdf, usd)
    assert stage.GetRootLayer().ExportToString() == first
    for name, expected in [("thumb", 1), ("glass", 0.3)]:
        visual = stage.GetPrimAtPath("/Hand/" + name + "/visuals")
        assert visual.IsInstance()
        material, _ = UsdShade.MaterialBindingAPI(
            stage.GetPrimAtPath(str(visual.GetPath()) + "/mesh")
        ).ComputeBoundMaterial()
        surface = UsdShade.Shader(
            stage.GetPrimAtPath(str(material.GetPath()) + "/Shader")
        )
        assert surface.GetInput("opacity").Get() == pytest.approx(expected)
        assert stage.GetPrimAtPath("/Hand/" + name).GetAttribute(
            "physics:mass"
        ).Get() == pytest.approx(0.04)
        assert tuple(
            stage.GetPrimAtPath("/Hand/" + name).GetAttribute("xformOp:translate").Get()
        ) == (1, 2, 3)
        assert not stage.GetPrimAtPath(
            "/Hand/" + name + "/collisions/mesh"
        ).GetRelationship("material:binding")
    assert shader.GetInput("opacity").Get() == 0


@pytest.mark.parametrize("surface", ["UsdPreviewSurface", "OmniPBR_Opacity"])
def test_undeclared_zero_opacity_is_repaired_only_on_its_visual(tmp_path, surface):
    pytest.importorskip("pxr.Usd")
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    usd = str(tmp_path / "hand.usda")
    stage = Usd.Stage.CreateNew(usd)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Hand").GetPrim())
    opacity_name = "opacity" if surface == "UsdPreviewSurface" else "opacity_constant"
    for key, opacity in [("zero", 0), ("partial", 0.4)]:
        UsdGeom.Xform.Define(stage, "/Cad_" + key)
        mesh = UsdGeom.Cube.Define(stage, "/Cad_" + key + "/mesh").GetPrim()
        mat = UsdShade.Material.Define(stage, "/Cad_" + key + "/Material")
        shader = UsdShade.Shader.Define(stage, str(mat.GetPath()) + "/Shader")
        if surface == "UsdPreviewSurface":
            shader.CreateIdAttr(surface)
        else:
            shader.GetPrim().CreateAttribute(
                "info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token
            ).Set(surface)
            shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(True)
            shader.CreateInput("enable_opacity_texture", Sdf.ValueTypeNames.Bool).Set(
                False
            )
        shader.CreateInput(opacity_name, Sdf.ValueTypeNames.Float).Set(opacity)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.37)
        UsdShade.MaterialBindingAPI.Apply(mesh).Bind(mat)
    for name, key in [("tip", "zero"), ("glass", "partial")]:
        for component in ("visuals", "collisions"):
            prim = UsdGeom.Xform.Define(
                stage, "/Hand/" + name + "/" + component
            ).GetPrim()
            prim.GetReferences().AddInternalReference("/Cad_" + key)
            prim.SetInstanceable(True)
    stage.GetRootLayer().Save()
    urdf = write_urdf(
        tmp_path, '<link name="tip"><visual/></link><link name="glass"><visual/></link>'
    )
    RUNTIME["configure_visual_materials"](urdf, usd)
    tip = stage.GetPrimAtPath("/Hand/tip/visuals")
    assert not tip.IsInstance()
    shader = UsdShade.Shader(stage.GetPrimAtPath("/Hand/tip/visuals/Material/Shader"))
    assert shader.GetInput(opacity_name).Get() == 1
    assert shader.GetInput("roughness").Get() == pytest.approx(0.37)
    assert stage.GetPrimAtPath("/Hand/glass/visuals").IsInstance()
    for path, value in [
        ("/Cad_zero", 0),
        ("/Hand/tip/collisions", 0),
        ("/Hand/glass/visuals", 0.4),
    ]:
        shader = UsdShade.Shader(stage.GetPrimAtPath(path + "/Material/Shader"))
        assert shader.GetInput(opacity_name).Get() == pytest.approx(value)
    first = stage.GetRootLayer().ExportToString()
    RUNTIME["configure_visual_materials"](urdf, usd)
    assert stage.GetRootLayer().ExportToString() == first
