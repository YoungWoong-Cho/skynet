"""Frozen Skynet point-cloud inputs for the unchanged native UniDex model.

The official configured dataset uses 10,000 input points; Uni3D's 1,024 feature
dimension is independent. Existing datasets retain their explicitly saved count.
This module is portable to training and evaluation capsules without model imports.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re


CONTRACT = "skynet.unidex-pointcloud-faas/v1"


def default_pointcloud_recipe():
    return {"modality": "point_cloud", "camera_ids": ["scene_front"],
            "width": 256, "height": 256, "dtype": "float32", "channels": "XYZRGB",
            "coordinate_frame": "camera:scene_front", "camera_convention": "ros_optical",
            "color_range": "0_1", "num_points": 10000, "sampling": "farthest_point", "seed": 0,
            "crop": None, "depth_range": [0.01, 5.0],
            "processing_order": ["unproject", "world_transform", "merge", "crop", "sample"],
            "insufficient_points": "repeat_with_mask"}


def validate_pointcloud_recipe(recipe):
    """Validate the supported camera recipe, retaining its explicit point count."""
    expected = default_pointcloud_recipe()
    if not isinstance(recipe, dict) or set(recipe) - {*expected, "name"}:
        raise ValueError("UniDex point-cloud recipe is missing or contains unsupported fields")
    count = recipe.get("num_points")
    # The native Uni3D requests 512 grouping centers before KNN aggregation.
    if type(count) is not int or not 512 <= count <= 100000:
        raise ValueError("UniDex point-cloud recipe requires an explicit count between 512 and 100000")
    expected["num_points"] = count
    if any(key not in recipe or recipe[key] != value for key, value in expected.items()):
        raise ValueError("UniDex point-cloud recipe differs from the supported camera and sampling contract")
    return deepcopy(expected)


def dataset_pointcloud_recipe(metadata):
    """One recipe across all episodes; partial planning metadata may omit shapes."""
    if not isinstance(metadata, dict) or metadata.get("contract") != CONTRACT:
        raise ValueError("UniDex requires its prepared point-cloud/FAAS dataset contract")
    episodes = metadata.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("UniDex dataset is missing its frozen point-cloud recipes")
    result = None
    for episode in episodes:
        stream = (episode.get("streams") or {}).get("scene_front_pointcloud") or {}
        recipe = validate_pointcloud_recipe(stream.get("recipe"))
        if result is not None and recipe != result:
            raise ValueError("UniDex dataset contains different point-cloud recipes")
        if "shape" in stream and stream["shape"] != [episode.get("steps"), recipe["num_points"], 6]:
            raise ValueError("UniDex point-cloud shape differs from its frozen recipe")
        if "dtype" in stream and stream["dtype"] != recipe["dtype"]:
            raise ValueError("UniDex point-cloud dtype differs from its frozen recipe")
        result = recipe
    return result


def collection_pointcloud_recipe(selections):
    if not selections:
        raise ValueError("UniDex requires at least one dataset with a frozen point-cloud recipe")
    recipes = [dataset_pointcloud_recipe(selection.get("metadata")) for selection in selections]
    if any(recipe != recipes[0] for recipe in recipes[1:]):
        raise ValueError("UniDex training datasets contain different point-cloud recipes")
    return recipes[0]


def validate_checkpoint_inputs(receipt, selections):
    expected = [{key: selection.get(key) for key in ("position", "version_id", "manifest_sha256")}
                for selection in selections]
    if receipt.get("schema") == "skynet.unidex-run/v2":
        matches = len(expected) == 1 and receipt.get("manifest_sha256") == expected[0]["manifest_sha256"]
    else:
        matches = bool(expected) and receipt.get("datasets") == expected
    if not matches:
        raise ValueError("UniDex checkpoint training datasets differ from the frozen training inputs")


def checkpoint_pointcloud_recipe(receipt, selections=None):
    """Resolve old receipts only from their checksum-pinned dataset manifests.

    Inference needs the input recipe, not training HDF5 values: read and verify
    the small immutable manifests without hashing or loading the physical clouds.
    No point count is inferred from today's adapter default or checkpoint weights.
    """
    recipe = None
    if "pointcloud_recipe" in receipt:
        recipe = validate_pointcloud_recipe(receipt["pointcloud_recipe"])
    if selections is not None:
        validate_checkpoint_inputs(receipt, selections)
        verified = []
        for selection in selections:
            sha = selection.get("manifest_sha256")
            path = selection.get("path")
            if not isinstance(path, str) or not path or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
                raise ValueError("UniDex checkpoint recipe requires checksum-pinned training manifests")
            path = Path(path)
            if path.is_dir():
                path = path / "manifest.json"
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != sha:
                raise ValueError("UniDex checkpoint training manifest checksum mismatch")
            verified.append({**selection, "metadata": json.loads(payload)})
        recorded = collection_pointcloud_recipe(verified)
        if recipe is not None and recorded != recipe:
            raise ValueError("UniDex checkpoint point-cloud recipe differs from its frozen training inputs")
        recipe = recorded
    if recipe is None:
        raise ValueError("UniDex checkpoint lacks a point-cloud recipe; provide its checksum-pinned training datasets")
    return recipe
