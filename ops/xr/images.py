"""Fixed scene camera configuration for adapter simulation evaluation."""
import hashlib
import json

CAMERAS = {"scene_front": "third_person_camera", "scene_left": "third_person_camera_left", "scene_right": "third_person_camera_right"}




def configure_cameras(cfg):
    # Observations are read directly from sensors, without observation histories.
    cfg._apply_observation_preset("state")
    cfg._apply_multiview_cameras(True)
    for name in CAMERAS.values():
        camera = getattr(cfg.scene, name, None)
        if camera is None:
            raise ValueError("Training image camera is unavailable: " + name)
        camera.width = camera.height = 256
        camera.data_types = ["rgb"]
        camera.update_period = 0.0
    for name in vars(cfg.scene):
        if name not in CAMERAS.values() and "camera" in name:
            setattr(cfg.scene, name, None)
    cfg.observations.debug_vis = None


def camera_recipe(cfg):
    """Freeze camera geometry without creating sensors in the XR environment."""
    import copy
    prepared = copy.deepcopy(cfg)
    configure_cameras(prepared)
    for name in vars(prepared.scene):
        spawn = getattr(getattr(prepared.scene, name, None), "spawn", None)
        if spawn is not None and any(s in type(spawn).__name__ for s in ["MultiAsset", "MultiUsd"]):
            raise ValueError("Training images require a fixed scene asset layout")
    result = {}
    for name, sensor in CAMERAS.items():
        camera = getattr(prepared.scene, sensor)
        spawn = camera.spawn
        result[name] = dict(
            sensor=sensor, prim_path=camera.prim_path, width=256, height=256,
            offset=dict(pos=list(camera.offset.pos), rot=list(camera.offset.rot), convention=camera.offset.convention),
            projection={key: getattr(spawn, key, None) for key in ["focal_length", "focus_distance", "horizontal_aperture", "vertical_aperture", "horizontal_aperture_offset", "vertical_aperture_offset", "clipping_range", "projection_type"]},
        )
    return json.loads(json.dumps(result, allow_nan=False))


def training_image_request(recipe):
    raw = json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return dict(schema="skynet.training-images/v2", mode="saved_states", recipe=recipe, recipe_sha256=hashlib.sha256(raw.encode()).hexdigest())
