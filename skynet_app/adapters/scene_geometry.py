"""Portable visual geometry captured from the simulator, with no browser USD dependency.

Mesh points and primitive transforms are frozen in each rigid body's local frame;
recorded rigid-body poses animate them. Static scenery keeps its world transform.
No filenames, external URLs, collision approximations, or executable assets are sent.
"""
import json

import numpy as np

MAX_POINTS = 120_000
MAX_PARTS = 400
MAX_JSON_BYTES = 12_000_000


def triangulate(points, counts, indices, holes=()):
    """Ear-clip planar polygons, preserving winding and respecting USD holes."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Invalid mesh points")
    counts, indices = list(map(int, counts)), list(map(int, indices))
    if any(n < 3 or n > 256 for n in counts) or sum(counts) != len(indices):
        raise ValueError("Invalid or oversized mesh faces")
    if any(i < 0 or i >= len(points) for i in indices):
        raise ValueError("Mesh index outside vertex array")
    triangles, offset = [], 0
    for face, count in enumerate(counts):
        polygon = indices[offset:offset + count]
        offset += count
        if face in holes:
            continue
        vertices = points[polygon]
        normal = np.sum(np.cross(vertices, np.roll(vertices, -1, axis=0)), axis=0)
        axis = int(np.argmax(np.abs(normal)))
        xy = np.delete(points, axis, axis=1)
        vertices = xy[polygon]
        area = np.sum(vertices[:, 0] * np.roll(vertices[:, 1], -1) - np.roll(vertices[:, 0], -1) * vertices[:, 1])
        if abs(area) < 1e-14:
            continue
        direction = 1 if area > 0 else -1
        cross = lambda a, b, c: ((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])) * direction
        remaining = list(polygon)
        while len(remaining) > 3:
            for i, middle in enumerate(remaining):
                a, b, c = remaining[i-1], middle, remaining[(i+1) % len(remaining)]
                if cross(xy[a], xy[b], xy[c]) <= 1e-14:
                    continue
                if any(min(cross(xy[a],xy[b],xy[p]), cross(xy[b],xy[c],xy[p]), cross(xy[c],xy[a],xy[p])) >= -1e-14
                       for p in remaining if p not in (a, b, c)):
                    continue
                triangles.extend((a, b, c))
                remaining.pop(i)
                break
            else:
                raise ValueError("Non-planar, degenerate, or self-intersecting mesh face")
        triangles.extend(remaining)
    return triangles


def _color(prim, UsdGeom, UsdShade):
    color = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
    if color:
        return list(map(float, color[0]))
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if material:
        shader = material.ComputeSurfaceSource()[0]
        if shader:
            for name in ("diffuseColor", "diffuse_color_constant", "base_color"):
                value = shader.GetInput(name).Get()
                if value is not None and hasattr(value, "__len__") and len(value) >= 3:
                    return list(map(float, value[:3]))
    return [.6, .6, .6]


def _part(prim, UsdGeom, UsdShade, remaining_points):
    kind = prim.GetTypeName()
    value = {"color": _color(prim, UsdGeom, UsdShade)}
    if kind == "Mesh":
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        if points is None or len(points) > remaining_points:
            raise ValueError("Mesh exceeds the preview geometry budget")
        value.update(type="mesh", vertices=np.asarray(points).round(7).reshape(-1).tolist(),
                     indices=triangulate(points, mesh.GetFaceVertexCountsAttr().Get() or [], mesh.GetFaceVertexIndicesAttr().Get() or [], mesh.GetHoleIndicesAttr().Get() or []))
        if mesh.GetOrientationAttr().Get() == "leftHanded":
            value["indices"] = [i for a,b,c in zip(*[iter(value["indices"])]*3) for i in (a,c,b)]
    elif kind == "Cube":
        value.update(type="box", size=[float(UsdGeom.Cube(prim).GetSizeAttr().Get())]*3)
    elif kind == "Sphere":
        value.update(type="sphere", radius=float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()))
    elif kind in {"Cylinder", "Cone", "Capsule"}:
        shape = getattr(UsdGeom, kind)(prim)
        value.update(type=kind.lower(), radius=float(shape.GetRadiusAttr().Get()),
                     height=float(shape.GetHeightAttr().Get()), axis=str(shape.GetAxisAttr().Get()))
    elif kind == "Plane" and hasattr(UsdGeom, "Plane"):
        shape = UsdGeom.Plane(prim)
        value.update(type="plane", size=[float(shape.GetWidthAttr().Get()), float(shape.GetLengthAttr().Get())], axis=str(shape.GetAxisAttr().Get()))
    else:
        return None
    return value


def capture_scene(scene, *, stage=None):
    """Capture visible meshes (including USD instances), materials, and primitives."""
    from pxr import Gf, Usd, UsdGeom, UsdShade
    if stage is None:
        stage = getattr(scene, "stage", None)
    if stage is None:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise ValueError("The simulator stage is unavailable")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    env_path = str(getattr(scene, "env_prim_paths", ["/World/envs/env_0"])[0])
    def path_for(asset):
        cfg = getattr(asset, "cfg", None)
        return str(getattr(cfg, "prim_path", "")).replace("{ENV_REGEX_NS}", env_path).replace("/env_.*", "/env_0")
    articulations = getattr(scene, "articulations", {})
    excluded = [path_for(asset) for asset in articulations.values() if path_for(asset)]
    roots, result, warnings = {}, {}, []
    for name, obj in getattr(scene, "rigid_objects", {}).items():
        path = path_for(obj)
        prim = stage.GetPrimAtPath(path) if path else None
        if not prim or not prim.IsValid():
            continue
        matrix = cache.GetLocalToWorldTransform(prim)
        transform = Gf.Transform(matrix)
        rotation = transform.GetRotation().GetQuat()
        pose = [*map(float, transform.GetTranslation()), float(rotation.GetReal()), *map(float, rotation.GetImaginary())]
        rigid = Gf.Matrix4d(1);rigid.SetRotate(rotation);rigid.SetTranslateOnly(transform.GetTranslation())
        roots[path] = (name, rigid.GetInverse())
        result[name] = dict(name=name, type="asset", parts=[], pose=pose, dynamic=True)
    points, parts, geometry_bytes = 0, 0, 0
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if path.startswith("/World/envs/") and not (path == env_path or path.startswith(env_path + "/")):
            continue
        if any(path == p or path.startswith(p + "/") for p in excluded):
            continue
        if prim.GetTypeName() in {"PointInstancer", "BasisCurves", "NurbsCurves"}:
            warnings.append("Unsupported scene geometry: " + path)
            continue
        if not prim.IsA(UsdGeom.Gprim):
            continue
        image = UsdGeom.Imageable(prim)
        if image.ComputeVisibility() == "invisible" or image.ComputePurpose() in {"guide", "proxy"}:
            continue
        if parts >= MAX_PARTS:
            warnings.append("Scene exceeds the preview part budget; remaining geometry omitted")
            break
        try:
            part = _part(prim, UsdGeom, UsdShade, MAX_POINTS - points)
            if part is None:
                warnings.append("Unsupported scene geometry: " + path)
                continue
            matrix = cache.GetLocalToWorldTransform(prim)
            parent = next((p for p in sorted(roots, key=len, reverse=True) if path == p or path.startswith(p + "/")), None)
            if parent:
                name, inverse = roots[parent]
                matrix = matrix * inverse
            else:
                name = "static:" + path
                result[name] = dict(name=name, type="asset", parts=[], dynamic=False)
            # Gf uses row vectors; this layout is also Three.js's column-major array.
            part["matrix"] = np.asarray(matrix).round(9).reshape(-1).tolist()
            # Leave room for object names, poses, and warnings in the sidecar.
            part_bytes = len(json.dumps(part, allow_nan=False).encode("utf-8"))
            if geometry_bytes + part_bytes > MAX_JSON_BYTES - 100_000:
                warnings.append("Scene exceeds the preview size budget; remaining geometry omitted")
                break
            result[name]["parts"].append(part)
            geometry_bytes += part_bytes
            parts += 1; points += len(part.get("vertices", [])) // 3
        except (ValueError, TypeError, RuntimeError) as error:
            warnings.append(path + ": " + str(error))
    return dict(schema="skynet.scene-geometry/v1", objects=[v for v in result.values() if v["parts"]],
                warnings=warnings[:30], appearance="Saved surface colors; renderer-specific textures and lighting are not replayed")


def scene_snapshot(scene):
    try:
        return capture_scene(scene)
    except (ImportError, ValueError, AttributeError, RuntimeError) as error:
        return dict(schema="skynet.scene-geometry/v1", objects=[], warnings=["Scene capture unavailable: " + str(error)])
