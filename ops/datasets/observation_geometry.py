"""CPU-only, deterministic camera geometry used by observation preparation.

Depth is metric distance to the image plane; camera coordinates follow the ROS
optical convention (x right, y down, z forward). No simulator imports belong here.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def keyed_seed(*parts):
    return int.from_bytes(hashlib.sha256(canonical_json(parts).encode()).digest()[:8], "little")


def rigid_transform(value):
    value = np.asarray(value, dtype=np.float64)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(value[:3, :3]), 1.0, atol=1e-5)):
        raise ValueError("Camera pose must be a finite rigid world_from_camera transform")
    return value


def pose_from_ros(position, quaternion_wxyz):
    p = np.asarray(position, dtype=np.float64)
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    if p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Invalid camera position or ROS quaternion")
    norm = np.linalg.norm(q)
    if not np.isclose(norm, 1.0, atol=1e-4):
        raise ValueError("Camera quaternion is not normalized")
    w, x, y, z = q / norm
    result = np.eye(4)
    result[:3, :3] = [[1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)]]
    result[:3, 3] = p
    return rigid_transform(result)


def intrinsic_matrix(value):
    k = np.asarray(value, dtype=np.float64)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1], atol=1e-8)
            or not np.isfinite(np.linalg.det(k)) or abs(np.linalg.det(k)) < 1e-12):
        raise ValueError("Invalid pinhole camera intrinsic matrix")
    return k


def transform_points(xyz, transform):
    t = rigid_transform(transform)
    return np.asarray(xyz, dtype=np.float64) @ t[:3, :3].T + t[:3, 3]


def unproject(depth, intrinsics, *, valid_mask=None, rgb=None):
    """Return all valid camera XYZ and optional RGB, preserving pixel pairing."""
    depth = np.asarray(depth)
    if depth.ndim != 2 or depth.dtype.kind != "f" or not all(depth.shape):
        raise ValueError("Metric depth must be a nonempty H x W float image")
    k = intrinsic_matrix(intrinsics)
    valid = np.isfinite(depth) & (depth > 0)
    if valid_mask is not None:
        mask = np.asarray(valid_mask)
        if mask.dtype != np.bool_ or mask.shape != depth.shape:
            raise ValueError("Depth validity mask differs from its pixels")
        valid &= mask
    rows, columns = np.nonzero(valid)
    pixels = np.stack([columns, rows, np.ones(len(rows))], axis=1)
    xyz = (pixels @ np.linalg.inv(k).T) * depth[rows, columns, None]
    colors = None
    if rgb is not None:
        rgb = np.asarray(rgb)
        if rgb.shape != (*depth.shape, 3) or rgb.dtype != np.uint8:
            raise ValueError("RGB must be aligned H x W x 3 uint8 pixels")
        colors = rgb[rows, columns].astype(np.float32)
    return xyz, colors


def sample_points(points, count, *, method, seed, padding="zeros"):
    """Sample once after merging/cropping; return values and real-point validity."""
    points = np.asarray(points)
    if (points.ndim != 2 or points.shape[1] not in (3, 6) or not np.isfinite(points).all()
            or type(count) is not int or not 1 <= count <= 100_000
            or method not in {"random", "fps"} or padding not in {"zeros", "repeat"}):
        raise ValueError("Invalid point sampling recipe")
    rng = np.random.default_rng(seed)
    n = len(points)
    selected_count = min(n, count)
    if not n:
        return np.zeros((count, points.shape[1]), dtype=np.float32), np.zeros(count, dtype=bool)
    if n <= count:
        selected = np.arange(n)
    elif method == "random":
        selected = rng.choice(n, count, replace=False)
    else:
        # Deterministic FPS on XYZ only. Stable argmax resolves equal distances.
        xyz = points[:, :3].astype(np.float64, copy=False)
        selected = np.empty(selected_count, dtype=np.int64)
        distances = np.full(n, np.inf)
        used = np.zeros(n, dtype=bool)
        candidate = int(rng.integers(n))
        tree = None
        if n > 2048:
            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(xyz)
            except ImportError:
                pass
        for i in range(selected_count):
            selected[i] = candidate
            radius_squared = distances[candidate]
            used[candidate] = True
            if tree is not None and np.isfinite(radius_squared):
                # Any point farther than the current global maximum nearest
                # distance cannot improve. This spatial bound preserves exact
                # FPS while avoiding N coordinate differences at every step.
                ids = np.asarray(tree.query_ball_point(xyz[candidate], np.nextafter(np.sqrt(max(0, radius_squared)), np.inf)), dtype=np.int64)
                delta = xyz[ids] - xyz[candidate]
                distances[ids] = np.minimum(distances[ids], np.einsum("ij,ij->i", delta, delta))
            else:
                delta = xyz - xyz[candidate]
                distances = np.minimum(distances, np.einsum("ij,ij->i", delta, delta))
            distances[used] = -1
            candidate = int(np.argmax(distances))
    result = np.zeros((count, points.shape[1]), dtype=np.float32)
    result[:selected_count] = points[selected]
    valid = np.zeros(count, dtype=bool)
    valid[:selected_count] = True
    if count > n and padding == "repeat":
        result[n:] = points[rng.integers(n, size=count-n)]
        # Repeated padding is explicitly distinguishable from original samples.
    return result, valid


def point_cloud(views, recipe, *, identity, frame_id):
    """Build per-view or merged XYZ/XYZRGB from unsampled calibrated depth.

    Each view contains camera_id, depth, intrinsics, world_from_camera and may
    contain valid_mask/rgb. Crop coordinates are in the requested output frame.
    """
    if not views or len({v["camera_id"] for v in views}) != len(views):
        raise ValueError("Point cloud needs distinct camera inputs")
    view_by_id = {v["camera_id"]: v for v in views}
    frame = recipe["coordinate_frame"]
    if frame == "world":
        from_world = np.eye(4)
    elif frame == "camera" and len(views) == 1:
        from_world = np.linalg.inv(rigid_transform(views[0]["world_from_camera"]))
    elif isinstance(frame, str) and frame.startswith("camera:") and frame[7:] in view_by_id:
        from_world = np.linalg.inv(rigid_transform(view_by_id[frame[7:]]["world_from_camera"]))
    else:
        raise ValueError("Merged camera coordinates require an explicit reference camera")
    channels = recipe.get("channels", "xyz").lower()
    if channels not in {"xyz", "xyzrgb"}:
        raise ValueError("Point cloud channels must be xyz or xyzrgb")
    all_points = []
    for view in sorted(views, key=lambda item: item["camera_id"]):
        if channels == "xyzrgb" and view.get("rgb") is None:
            raise ValueError("XYZRGB requires aligned RGB for every input camera")
        valid_mask = np.isfinite(view["depth"]) & (np.asarray(view["depth"]) > 0)
        if view.get("valid_mask") is not None:
            valid_mask &= view["valid_mask"]
        if recipe.get("depth_range") is not None:
            lower, upper = recipe["depth_range"]
            if not np.isfinite([lower, upper]).all() or not 0 <= lower < upper:
                raise ValueError("Invalid metric point-cloud depth range")
            valid_mask &= (np.asarray(view["depth"]) >= lower) & (np.asarray(view["depth"]) <= upper)
        xyz, rgb = unproject(view["depth"], view["intrinsics"], valid_mask=valid_mask,
                             rgb=view.get("rgb") if channels == "xyzrgb" else None)
        xyz = transform_points(xyz, from_world @ rigid_transform(view["world_from_camera"]))
        if rgb is not None:
            color_range = recipe.get("color_range", "0_255")
            if color_range not in {"0_255", "0_1"}:
                raise ValueError("Unsupported point cloud color range")
            all_points.append(np.concatenate([xyz, rgb / (255 if color_range == "0_1" else 1)], axis=1))
        else:
            all_points.append(xyz)
    points = np.concatenate(all_points, axis=0)
    crop = recipe.get("crop")
    if crop is not None:
        lo, hi = np.asarray(crop["min"], dtype=float), np.asarray(crop["max"], dtype=float)
        if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or not (hi > lo).all():
            raise ValueError("Point cloud crop must contain finite increasing XYZ bounds")
        points = points[((points[:, :3] >= lo) & (points[:, :3] <= hi)).all(axis=1)]
    return sample_points(points, recipe["num_points"], method={"farthest_point": "fps"}.get(recipe.get("sampling"), recipe.get("sampling", "random")),
                         seed=keyed_seed(identity, int(frame_id), recipe.get("seed", 0)),
                         padding="repeat" if recipe.get("insufficient_points") == "repeat_with_mask" else recipe.get("padding", "zeros"))
