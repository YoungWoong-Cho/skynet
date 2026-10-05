"""Frozen render identities and a saved-state-only DexVerse camera backend.

Imports stay CPU-only until entering DexVerseRenderer. The default scene camera
policy is explicit and versioned here; it is not a claim about every upstream
version's defaults. Actual calibration is recorded for every control frame.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import ExitStack, contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import runpy
import shutil
import signal
import subprocess
import tempfile
import sys

import numpy as np

try:
    from observation_geometry import canonical_json, keyed_seed, pose_from_ros
except ImportError:
    from ops.datasets.observation_geometry import canonical_json, keyed_seed, pose_from_ros

CAMERA_POLICY = "skynet.scene-cameras/v1"
SUPPORTED_SOURCES = {"30cc673e27684b9f10186fa6bea731aed246bc9f", "917092d28764549f0f6a77872022c93ef3373c48"}
PROJECTION_KEYS = {"focal_length", "focus_distance", "horizontal_aperture", "vertical_aperture",
                   "horizontal_aperture_offset", "vertical_aperture_offset", "clipping_range", "projection_type"}


def check_renderer_driver(runtime):
    """Reject only the Isaac 5.1/driver pairs with evidenced RTX startup crashes.

    This probes the allocated GPU node immediately before native initialization;
    it is not cached cluster health and does not change any shared driver.
    NVIDIA confirmation: forums.developer.nvidia.com/t/371957
    Local minimal SimulationApp reproduction: L40S/A40, driver 595.84.
    """
    if not re.search(r"(?:^|[-_])isaacsim-5\.1(?:\.0)?(?:[_-]|$)", Path(runtime).name.lower()):
        return
    probe = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                           check=True, capture_output=True, text=True, timeout=10)
    blocked = sorted(set(line.strip() for line in probe.stdout.splitlines()) & {"595.84", "595.71.05"})
    if blocked:
        raise RuntimeError(
            "Camera rendering is unavailable on this GPU node: Isaac Sim 5.1 and NVIDIA driver "
            + ", ".join(blocked)
            + " have a known RTX startup crash (librtx.scenedb). Ask the cluster administrator to provide "
              "a compatible driver, such as NVIDIA's validated Linux 580.65.06, or a separately verified "
              "compatible Isaac runtime. No shared driver or runtime has been changed."
        )


@contextmanager
def capture_deadline(seconds=120):
    """Keep the native capture scheduler from holding a producer indefinitely."""
    def expired(*_):
        raise TimeoutError("Camera capture timed out before producing an image")
    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, expired)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *old_timer)
        signal.signal(signal.SIGALRM, previous)


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("Render identity requires pinned SHA-256 asset metadata")
    return value


def default_scene_camera_recipes(source_revision):
    if source_revision not in SUPPORTED_SOURCES:
        raise ValueError("This source revision requires explicitly frozen camera recipes")
    def camera(sensor, prim, position, rotation):
        return {"sensor": sensor, "prim_path": "{ENV_REGEX_NS}/" + prim, "width": 256, "height": 256,
                "offset": {"pos": position, "rot": rotation, "convention": "world"},
                "projection": {"focal_length": 24.0, "focus_distance": 400.0, "horizontal_aperture": 20.955,
                               "clipping_range": [0.1, 10.0]}}
    def side_rotation(yaw):
        pitch = math.atan2(1.5 - 0.6, 1.5)
        cy, sy, cp, sp = math.cos(yaw/2), math.sin(yaw/2), math.cos(pitch/2), math.sin(pitch/2)
        return [cy*cp, -sy*sp, cy*sp, sy*cp]
    return {"scene_front": camera("third_person_camera", "Camera", [-1.5, 0.0, 1.5], [0.985, 0.0, 0.175, 0.0]),
            "scene_left": camera("third_person_camera_left", "CameraLeft", [0.0, 1.5, 1.5], side_rotation(-math.pi/2)),
            "scene_right": camera("third_person_camera_right", "CameraRight", [0.0, -1.5, 1.5], side_rotation(math.pi/2))}


def camera_recipe_from_capture(camera, source_revision, camera_id="scene_front"):
    """Accept frozen renderer settings or a measured, fixed-scene calibration.

    Recorded calibrations omit USD prim and clipping/focus settings. These come
    only from the same pinned source recipe; measured intrinsics/pose are used
    explicitly and checked again against sensor output during live evaluation.
    """
    if all(key in camera for key in ("prim_path", "offset", "projection")):
        offset, projection = camera["offset"], camera["projection"]
        if not isinstance(offset, dict) or not isinstance(projection, dict):
            raise ValueError("Invalid frozen camera offset or projection")
        position, rotation = np.asarray(offset.get("pos"), dtype=float), np.asarray(offset.get("rot"), dtype=float)
        if (position.shape != (3,) or rotation.shape != (4,) or not np.isfinite(position).all()
                or not np.isfinite(rotation).all() or not np.isclose(np.linalg.norm(rotation), 1., atol=.01)
                or offset.get("convention") not in {"world", "ros", "opengl"}):
            raise ValueError("Invalid frozen camera pose")
        if (not isinstance(camera.get("prim_path"), str) or not camera["prim_path"]
                or not set(projection).issubset(PROJECTION_KEYS)):
            raise ValueError("Invalid frozen camera prim or projection")
        for key in ("focal_length", "horizontal_aperture"):
            value = projection.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid frozen camera projection: " + key)
        return deepcopy(camera)
    try:
        from observation_geometry import intrinsic_matrix
    except ImportError:
        from ops.datasets.observation_geometry import intrinsic_matrix
    base = deepcopy(default_scene_camera_recipes(source_revision)[camera_id])
    if camera.get("sensor") != base["sensor"] or camera.get("mount") != "fixed_scene":
        raise ValueError("Frozen camera must identify the pinned fixed-scene sensor")
    width, height = camera.get("width"), camera.get("height")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("Invalid frozen camera dimensions")
    k = intrinsic_matrix(camera.get("intrinsic_matrix"))
    if (not np.allclose(k[[0, 1], [1, 0]], 0, atol=1e-8)
            or not np.allclose(k[:2, 2], [width / 2, height / 2], atol=1e-5)):
        raise ValueError("Pinned scene cameras require centered, zero-skew intrinsics")
    measured_pose = pose_from_ros(camera.get("position_world"), camera.get("quaternion_world_ros"))
    # This legacy capture form has no separately frozen render recipe. Verify
    # the source-pinned conversion camera before reconstructing its calibration;
    # otherwise a recording camera could silently differ from prepared pixels.
    rotation = np.asarray(base["offset"]["rot"], dtype=float)
    expected_pose = pose_from_ros(base["offset"]["pos"], rotation / np.linalg.norm(rotation))
    expected_pose[:3, :3] = expected_pose[:3, :3] @ np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    focal = base["projection"]["focal_length"]
    expected_focal_pixels = width * focal / base["projection"]["horizontal_aperture"]
    if (not np.allclose(measured_pose, expected_pose, rtol=1e-5, atol=1e-5)
            or not np.allclose([k[0, 0], k[1, 1]], expected_focal_pixels, rtol=1e-5, atol=1e-5)):
        raise ValueError("Recorded calibration differs from its pinned observation camera recipe")
    base.update(width=width, height=height,
                offset=dict(pos=camera["position_world"], rot=camera["quaternion_world_ros"], convention="ros"))
    base["projection"].update(horizontal_aperture=float(width * focal / k[0, 0]),
                              vertical_aperture=float(height * focal / k[1, 1]))
    return base


def render_identity(profile, renderer_revision):
    """Portable identity; caller pins remote bundle metadata before planning."""
    revision = profile["source_revision"]
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Observation renderer needs an exact DexVerse source revision")
    cameras = profile.get("camera_recipe") or profile.get("cameras")
    if cameras is None:
        cameras = default_scene_camera_recipes(revision)
    identity = {"source_revision": revision, "renderer_revision": _sha(renderer_revision),
                "camera_policy": CAMERA_POLICY, "seed": 0, "task": profile["task"], "robot": profile["robot"],
                "runtime": PurePosixPath(profile.get("runtime", "isaacsim-5.1.0_isaaclab-2.3.2_py311")).name,
                "hand_bundle_digest": (profile.get("hand_bundle") or {}).get("digest"),
                "cameras": deepcopy(cameras)}
    if profile.get("hand_bundle"):
        identity["hand_bundle_manifest_sha256"] = _sha(profile["hand_bundle"].get("manifest_sha256"))
    if profile.get("asset_bundle"):
        identity.update(asset_bundle_version=PurePosixPath(profile["asset_bundle"]).name,
                        asset_bundle_manifest_sha256=_sha(profile.get("asset_bundle_manifest_sha256")),
                        asset_inventory_sha256=_sha(profile.get("asset_inventory_sha256")))
    if profile.get("asset_files"):
        identity["asset_files"] = {name: _sha(value["sha256"]) for name, value in sorted(profile["asset_files"].items())}
    return json.loads(canonical_json(identity))


def _verify_assets(profile):
    """Verify frozen bundle index, then hash only scene dependencies actually used."""
    receipts = {}
    bundle_path = profile.get("asset_bundle")
    if bundle_path:
        bundle = Path(bundle_path)
        manifest_path = bundle / "manifest.json"
        if _digest(manifest_path) != _sha(profile.get("asset_bundle_manifest_sha256")):
            raise ValueError("Frozen simulation asset bundle manifest changed")
        ready = (bundle / "READY").read_text().split()
        if ready != [profile["asset_bundle_manifest_sha256"], "manifest.json"]:
            raise ValueError("Simulation asset bundle is not published and ready")
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("schema_version") != "dataset-bundle/v1" or manifest.get("status") != "ready"
                or manifest.get("bundle_id") != "bundle:dexverse/simulation-assets@" + bundle.name):
            raise ValueError("Unsupported immutable simulation asset bundle")
        inventory = bundle / manifest["integrity"]["inventory_path"]
        if (not inventory.resolve().is_relative_to(bundle.resolve()) or inventory.stat().st_size > 32_000_000
                or _digest(inventory) != _sha(profile.get("asset_inventory_sha256"))
                or manifest["integrity"]["inventory_sha256"] != profile["asset_inventory_sha256"]):
            raise ValueError("Frozen simulation asset inventory changed")
        for row in inventory.read_text().splitlines():
            sha, size, relative = row.split("\t", 2)
            relative_path = PurePosixPath(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError("Invalid simulation asset inventory path")
            path = (bundle / manifest["view"]["root"] / relative).resolve()
            receipts[str(path)] = {"sha256": _sha(sha), "size_bytes": int(size),
                                   "path": str(bundle / manifest["view"]["root"] / relative)}
            # The simulator checkout may expose bundle files through an overlay
            # or a verified copy rather than directly through the bundle view.
            receipts[str((Path(profile["repository"]) / relative).resolve())] = receipts[str(path)]
    for logical, receipt in profile.get("asset_files", {}).items():
        path = Path(receipt["path"]).resolve()
        if _digest(path) != _sha(receipt["sha256"]):
            raise ValueError("Frozen scene asset changed: " + logical)
        receipts[str(path)] = {"sha256": receipt["sha256"], "size_bytes": path.stat().st_size}
    return receipts


def bind_scene_assets(scene, receipts):
    """Resolve frozen source asset names into the explicitly selected bundle view."""
    keys = {"asset_path", "usd_path", "mdl_path", "texture_file", "filename"}
    seen = set()
    def bind(value, key=""):
        if isinstance(value, str) and key in keys and Path(value).is_absolute():
            receipt = receipts.get(str(Path(value).resolve()))
            return receipt.get("path", value) if receipt else value
        if value is None or isinstance(value, (str, int, float, bool)) or callable(value):
            return value
        if id(value) in seen:
            return value
        seen.add(id(value))
        if isinstance(value, dict):
            for name, item in value.items():
                value[name] = bind(item, str(name))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                value[i] = bind(item)
        elif hasattr(value, "__dict__"):
            for name, item in vars(value).items():
                if not name.startswith("_"):
                    setattr(value, name, bind(item, name))
        return value
    bind(scene)


def configure_cameras(cfg, jobs, *, replay=True):
    """Configure only requested native sensors and output modalities, no history."""
    cfg._apply_observation_preset("state")
    # This only creates camera configs; unrequested sensors are removed below.
    cfg._apply_multiview_cameras(True)
    selected = {}
    for job in jobs:
        name, recipe = job["camera_id"], job["recipe"]
        camera = deepcopy(recipe["camera"])
        camera.update(width=recipe["width"], height=recipe["height"])
        if name in selected and selected[name]["camera"] != camera:
            raise ValueError("Batch contains incompatible recipes for one physical camera")
        selected.setdefault(name, {"camera": camera, "modalities": set()})["modalities"].add(job["modality"])
    sensor_ids = [value["camera"]["sensor"] for value in selected.values()]
    if len(set(sensor_ids)) != len(sensor_ids):
        raise ValueError("Two camera IDs alias the same physical sensor in one render batch")
    for name in vars(cfg.scene):
        if "camera" in name and name not in sensor_ids and "body" not in name:
            setattr(cfg.scene, name, None)
    for name, request in selected.items():
        recipe = request["camera"]
        camera = getattr(cfg.scene, recipe["sensor"], None)
        if camera is None or camera.prim_path != recipe["prim_path"]:
            raise ValueError("Frozen camera is unavailable or attached to a different prim: " + name)
        camera.width, camera.height = recipe["width"], recipe["height"]
        camera.offset.pos, camera.offset.rot = tuple(recipe["offset"]["pos"]), tuple(recipe["offset"]["rot"])
        camera.offset.convention = recipe["offset"]["convention"]
        if camera.offset.convention not in {"world", "ros", "opengl"}:
            raise ValueError("Unknown frozen camera coordinate convention")
        if not set(recipe["projection"]).issubset(PROJECTION_KEYS):
            raise ValueError("Unsupported frozen camera projection field")
        for key, value in recipe["projection"].items():
            if not hasattr(camera.spawn, key):
                raise ValueError("Pinned renderer does not support camera projection: " + key)
            setattr(camera.spawn, key, tuple(value) if isinstance(value, list) else value)
        camera.data_types = sorted("rgb" if kind == "rgb" else "distance_to_image_plane" for kind in request["modalities"])
        camera.update_period = 0.0
        camera.update_latest_camera_pose = True
    # Observations are read directly from sensors. Manager histories/noise and
    # all reset/startup randomization are excluded from the capture recipe.
    if replay:
        cfg.observations = {}
        cfg.events = {}
        cfg.recorders, cfg.terminations = {}, {}
    cfg.num_rerenders_on_reset = 4
    return selected


def configure_capture_settings():
    """Use identical full-frame RTX and explicit capture settings for replay/rollout."""
    import carb
    settings = carb.settings.get_settings()
    for index, value in enumerate((0.0, 0.0, 1.0, 1.0)):
        settings.set_float(f"/rtx/dataWindowNDC/{index}", value)
    settings.set_bool("/rtx/dataWindow/fitOutputToDataWindow", True)
    settings.set_bool("/exts/omni.replicator.core/Orchestrator/enabled", True)
    settings.set_bool("/omni/replicator/captureOnPlay", False)
    settings.set_bool("/omni/replicator/captureMotionBlur", False)


def read_camera(sensor, render, modalities, *, camera_id="scene_front", render_mode=None):
    """Read settled native pixels and their calibration without advancing physics."""
    def array(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    for _ in range(32):
        ready = True
        for annotator in sensor._annotators.values():
            output = annotator.get_data()
            if isinstance(output, dict):
                output = output["data"]
            ready = ready and bool(output.size)
        if ready:
            break
        render()
    else:
        raise RuntimeError("Camera render product stayed empty without physics stepping: "
                           + camera_id + "; render mode=" + str(render_mode))
    sensor.update(0.0, force_recompute=True)
    data = sensor.data
    result = {"intrinsics": array(data.intrinsic_matrices)[0].copy(),
              "world_from_camera": pose_from_ros(array(data.pos_w)[0], array(data.quat_w_ros)[0])}
    for modality in modalities:
        key = "rgb" if modality == "rgb" else "distance_to_image_plane"
        values = array(data.output[key])[0]
        result[modality] = values[:, :, :3].copy() if modality == "rgb" else values.reshape(values.shape[:2]).astype("float32")
    return result


def capture_current_frame(env):
    """Settle offscreen products with the replay scheduler and zero physics time."""
    import omni.replicator.core as rep
    import omni.timeline
    timeline = omni.timeline.get_timeline_interface()
    before = timeline.get_current_time()
    with capture_deadline():
        rep.orchestrator.step(rt_subframes=4, pause_timeline=False, delta_time=0.0, wait_for_render=True)
    if not np.isclose(timeline.get_current_time(), before, rtol=0, atol=1e-12):
        raise RuntimeError("Observation capture advanced the simulation timeline")
    env.scene.update(dt=env.physics_dt)


def install_frozen_hand(profile, *, staging_root=None, asset_receipts=None):
    """Install verified hand assets into a private, parent-owned temporary area.

    Used by observation replay and live policy evaluation. The caller retains
    the returned directory until native simulation stops; its parent process
    owns final cleanup because Kit may exit without returning to Python.
    """
    bundle = profile.get("hand_bundle")
    if not bundle:
        return None, None, None
    original = Path(bundle["root"])
    if _digest(original / "manifest.json") != _sha(bundle.get("manifest_sha256")):
        raise ValueError("Frozen hand manifest changed")
    manifest = json.loads((original / "manifest.json").read_text())
    if manifest.get("digest") != bundle["digest"]:
        raise ValueError("Renderer installed a different frozen hand bundle")
    if staging_root is not None:
        staging = Path(staging_root)
        if not staging.is_absolute() or staging.is_symlink():
            raise ValueError("Renderer staging root must be an absolute nonsymlink directory")
        staging.mkdir(parents=True, exist_ok=True)
    work = tempfile.TemporaryDirectory(prefix="observation-hand-", dir=staging_root)
    hand_root = Path(work.name)
    try:
        for name, receipt in manifest["files"].items():
            source = (original / name).resolve()
            destination = hand_root / name
            if (not source.is_relative_to(original.resolve()) or not destination.resolve().is_relative_to(hand_root.resolve())
                    or _digest(source) != receipt["sha256"] or source.stat().st_size != receipt["size_bytes"]):
                raise ValueError("Frozen hand asset changed: " + name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if asset_receipts is not None:
                asset_receipts[str(destination.resolve())] = receipt
        shutil.copyfile(original / "manifest.json", hand_root / "manifest.json")
        sys.path.insert(0, str(hand_root))
        from runtime import install, validate_environment
        manifest = install(hand_root)
        return work, manifest, validate_environment
    except BaseException:
        work.cleanup()
        raise


def verify_scene_assets(profile, asset_receipts, generated_asset_root=None):
    """Verify loaded USD layers using the same frozen inventory for replay/rollout."""
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    runtime = Path(profile["runtime"]).resolve()
    generated = Path(generated_asset_root).resolve() if generated_asset_root is not None else None
    verified = {}
    for layer in stage.GetUsedLayers():
        filename = str(layer.realPath)
        if not filename or filename.startswith(("anon:", "omniverse://", "http://", "https://")):
            continue
        path = Path(filename).resolve()
        receipt = asset_receipts.get(str(path))
        if receipt is None and not path.is_relative_to(runtime) and not (generated and path.is_relative_to(generated)):
            raise ValueError("Loaded USD layer is absent from its pinned inventory: " + str(path))
        if receipt:
            actual_sha = _digest(path)
            if path.stat().st_size != receipt["size_bytes"] or actual_sha != receipt["sha256"]:
                raise ValueError("Frozen simulation asset changed: " + str(path))
            verified[str(path)] = actual_sha
    return verified


class DexVerseRenderer:
    def __init__(self, sources, jobs, *, progress=None, staging_root=None):
        self.sources, self.jobs = sources, jobs
        self.env = self.app = None
        self.capture_metadata = {}
        self.hand_work = None
        self.generated_asset_root = None
        self._contexts = ExitStack()
        self.progress = progress or (lambda phase, **fields: None)
        self.staging_root = staging_root

    def __enter__(self):
        selected_sources = [self.sources[key] for key in dict.fromkeys(j["episode_key"] for j in self.jobs)]
        self.profile = deepcopy(selected_sources[0]["profile"])
        for source in selected_sources:
            # One simulator per compatible session/runtime, never one per file.
            for key in ("repository", "source_revision", "task", "robot", "hand", "hand_bundle", "asset_bundle"):
                if source["profile"].get(key) != self.profile.get(key):
                    raise ValueError("A render batch must use one compatible scene, hand and runtime")
            for job in [j for j in self.jobs if j["episode_key"] == source["episode_key"]]:
                identity = render_identity(source["profile"], job["spec"]["render_identity"]["renderer_revision"])
                if {k: v for k, v in identity.items() if k != "cameras"} != job["spec"]["render_identity"]:
                    raise ValueError("Worker profile differs from its frozen render identity")
                if identity["cameras"][job["camera_id"]] != job["recipe"]["camera"]:
                    raise ValueError("Worker camera differs from its frozen render recipe")
        check_renderer_driver(self.profile["runtime"])
        root = Path(self.profile["repository"])
        marker = root / ".skynet-source-revision"
        if marker.read_text().strip() != self.profile["source_revision"]:
            raise ValueError("Renderer source revision differs from the recording")
        self.asset_receipts = _verify_assets(self.profile)
        recorder = root / "scripts/record_demos.py"
        for package in (root / "source/dexverse", root / "source/isaaclab"):
            if package.is_dir():
                sys.path.insert(0, str(package))
        self.previous_argv = list(sys.argv)
        sys.argv = [str(recorder), "--task", self.profile["task"], "--robot_type", self.profile["robot"],
                    "--teleop_device", "keyboard", "--enable_cameras", "--headless",
                    # DexVerse imports Pinocchio transitively; this upstream
                    # flag also enables Isaac Lab's Gf.Matrix4d compatibility patch.
                    "--enable_pinocchio",
                    "--device", self.profile.get("device", "cuda:0"), "--teleop_retargeter", "absolute",
                    # Use the pinned kit's standard headless lifecycle while
                    # excluding mutable shared renderer preferences.
                    "--kit_args=--/app/settings/loadUserConfig=false --/app/settings/persistent=false"]
        try:
            self.ns = runpy.run_path(str(recorder), run_name="skynet_observation_prepare")
            self.app = self.ns["simulation_app"]
            # Keep Isaac Sim's default fast shutdown. The worker commits its
            # verified result before close(), which may terminate this process.
            # Freeze RTX's documented full-frame, no-overscan projection.
            # Calibration and exported pixels must use the same image extent.
            # Full [0,0,1,1] already fits the output exactly. This also avoids
            # Replicator cropping an empty one-dimensional startup buffer.
            # Isaac Lab's headless kit disables Replicator's capture graph. We
            # use explicit, zero-delta capture instead of automatic playback.
            configure_capture_settings()
            import torch
            self.torch = torch
            # Isaac Lab keeps tensors across episodes. Keep their creation,
            # reset/restore, capture and cleanup in one inference context;
            # a capture-only context makes the next reset mutate inference
            # tensors outside that context.
            self._contexts.enter_context(torch.inference_mode())
            self._create_environment()
            return self
        except BaseException as exc:
            self.__exit__(*sys.exc_info())
            if isinstance(exc, SystemExit):
                raise RuntimeError("Pinned DexVerse renderer could not initialize; see the render job log") from exc
            raise
        finally:
            sys.argv = self.previous_argv

    def _create_environment(self):
        from wrist import configure_virtual_wrist
        self.hand_work, manifest, validate_environment = install_frozen_hand(
            self.profile, staging_root=self.staging_root, asset_receipts=self.asset_receipts)
        if manifest:
            self.profile["hand_manifest"] = manifest
            self.generated_asset_root = (Path(self.hand_work.name) / "usd").resolve()
        cfg, _ = self.ns["create_environment_config"]()
        self.cameras = configure_cameras(cfg, self.jobs)
        # ManagerBase skips setup for {}, but ObservationManager then requires
        # initialized group tables. One empty, non-concatenated group is valid.
        from isaaclab.managers import ObservationGroupCfg
        cfg.observations = {"policy": ObservationGroupCfg(concatenate_terms=False)}
        bind_scene_assets(cfg.scene, self.asset_receipts)
        self.scene_asset_digests = {}
        self.scene_configuration_digest = self._scene_identity(cfg)
        for name in vars(cfg.scene):
            spawn = getattr(getattr(cfg.scene, name, None), "spawn", None)
            if spawn is not None and any(term in type(spawn).__name__ for term in ("MultiAsset", "MultiUsd")):
                raise ValueError("Saved-state observation rendering requires a fixed scene asset layout: " + name)
        # Fixed configuration and independent random seeds prevent camera request
        # order from changing per-episode scene realization.
        cfg.seed = 0
        cfg.scene.num_envs = 1
        self.env = self.ns["create_environment"](cfg)
        if manifest:
            validate_environment(self.env, manifest)
        configure_virtual_wrist(self.env.scene["robot"], manifest, ["left", "right"] if self.profile["hand"] == "both" else [self.profile["hand"]])
        self._verify_used_assets()
        # No action replay or simulation integration is permitted during capture.
        def no_step(*args, **kwargs):
            raise RuntimeError("Observation rendering must not advance physics")
        self.env.step = no_step
        self.env.sim.step = no_step

    def _scene_identity(self, cfg):
        """Fingerprint the non-camera scene and verify local source asset bytes."""
        def freeze(value, key=""):
            if value is None or isinstance(value, (str, int, float, bool)):
                if isinstance(value, float) and not math.isfinite(value):
                    return {"configured_number": repr(value)}
                if isinstance(value, str) and key in {"asset_path", "usd_path", "mdl_path", "texture_file", "filename"}:
                    path = Path(value)
                    if path.is_absolute() and path.is_file():
                        path = path.resolve()
                        if self.generated_asset_root is not None and path.is_relative_to(self.generated_asset_root):
                            # Freshly converted USD may embed its temporary path.
                            # Its immutable inputs define scene equivalence.
                            return {"generated_hand_asset": str(path.relative_to(self.generated_asset_root)),
                                    "hand_bundle_manifest_sha256": self.profile["hand_bundle"]["manifest_sha256"]}
                        sha = _digest(path)
                        receipt = self.asset_receipts.get(str(path))
                        if receipt is None and not self._runtime_or_generated_asset(path):
                            raise ValueError("Scene asset is absent from its pinned inventory: " + str(path))
                        if receipt and (sha != receipt["sha256"] or path.stat().st_size != receipt["size_bytes"]):
                            raise ValueError("Frozen source scene asset changed: " + str(path))
                        self.scene_asset_digests[str(path)] = sha
                        # Preserve source bytes as the identity, not a machine's mount path.
                        return {"asset_sha256": sha}
                return value
            if isinstance(value, dict):
                return {str(k): freeze(v, str(k)) for k, v in sorted(value.items())}
            if isinstance(value, (list, tuple)):
                return [freeze(item) for item in value]
            if callable(value):
                return getattr(value, "__module__", "") + "." + getattr(value, "__qualname__", type(value).__name__)
            if hasattr(value, "to_dict"):
                return freeze(value.to_dict())
            return type(value).__module__ + "." + type(value).__qualname__
        scene = {}
        sensor_names = {v["camera"]["sensor"] for v in self.cameras.values()}
        for name, value in vars(cfg.scene).items():
            if "camera" in name and "body" not in name or name in sensor_names:
                continue
            scene[name] = freeze(value)
        return hashlib.sha256(canonical_json(scene).encode()).hexdigest()

    def _runtime_or_generated_asset(self, path):
        runtime = Path(self.profile["runtime"]).resolve()
        return path.is_relative_to(runtime) or (self.generated_asset_root is not None and path.is_relative_to(self.generated_asset_root))

    def _verify_used_assets(self):
        # USD enumerates referenced layers after loading; verify the frozen
        # physical assets actually used, without rereading the complete bundle.
        verified = verify_scene_assets(self.profile, self.asset_receipts, self.generated_asset_root)
        self.scene_asset_digests.update(verified)
        # Imported URDF geometry is baked into generated USD layers. Its source
        # bundle is already fully verified by install()/read_bundle().
        self.verified_asset_paths = sorted(verified)

    def begin_episode(self, source, payload, episode, step_dt):
        from trajectory import restore_episode_conditions
        from scene_restore import restore_state
        from recording_metadata import state_metadata
        self.restore_state = restore_state
        seed = keyed_seed(source["sha256"], source.get("episode_index", 0), "scene") % (2**31)
        random.seed(seed)
        np.random.seed(seed)
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)
        self.env.reset()
        if step_dt is not None and not np.isclose(self.env.step_dt, step_dt, rtol=0, atol=1e-9):
            raise ValueError("Renderer control timing differs from the original recording")
        restore_state(self.env, episode["states"][0])
        if payload["schema_version"] == 3 and "object_pose" in self.env.command_manager.active_terms:
            target = self.env.command_manager.get_command("object_pose")
            goal = np.asarray(episode.get("goal_pose"))
            if goal.shape != tuple(target.shape[1:]) or not np.isfinite(goal).all():
                raise ValueError("Original recording has no restorable task goal")
            target.copy_(self.torch.as_tensor(goal, device=self.env.device)[None])
        restore_episode_conditions(self.env, payload, episode)
        _, metadata = state_metadata(self.env, self.profile)
        # Preview geometry can be large; the training capture needs the immutable
        # robot/controller layout, not a duplicate of the preview mesh payload.
        metadata.pop("scene_geometry", None)
        metadata.update(render_mode="saved_states", alignment="pre_action_state", cameras={k: v["camera"] for k, v in self.cameras.items()},
                        rendered_scene_sha256=self.scene_configuration_digest,
                        rendered_assets_sha256=hashlib.sha256(canonical_json({"scene": sorted(self.scene_asset_digests.values()),
                            "hand_manifest": (self.profile.get("hand_bundle") or {}).get("manifest_sha256")}).encode()).hexdigest())
        saved = payload.get("skynet_state_metadata", {})
        for key in ("robot", "task", "action_joint_names", "robot_joint_names", "action_scale", "action_offset"):
            if key in saved and saved[key] != metadata.get(key):
                raise ValueError("Renderer robot/controller layout differs from recorded " + key)
        if len(metadata["action_joint_names"]) != np.asarray(episode["actions"]).shape[1]:
            raise ValueError("Renderer action layout differs from the original recording")
        self.capture_metadata = metadata

    def capture(self, state, jobs):
        if not self.app.is_running() or self.app.is_exiting():
            raise RuntimeError("Renderer exited before finishing requested observations")
        self.restore_state(self.env, state)
        # Explicit capture pumps offscreen render products even at time 0.
        # Replicator 1.12.27 documents delta_time=0 as no timeline advance;
        # subframes settle rendering while physics remains paused.
        capture_current_frame(self.env)
        result = {}
        for camera_id in dict.fromkeys(j["camera_id"] for j in jobs):
            sensor = self.env.scene[self.cameras[camera_id]["camera"]["sensor"]]
            result[camera_id] = read_camera(sensor, self.env.sim.render,
                {j["modality"] for j in jobs if j["camera_id"] == camera_id},
                camera_id=camera_id, render_mode=self.env.sim.render_mode)
        return result

    def __exit__(self, *_):
        try:
            try:
                if self.env is not None:
                    self.progress("closing_env")
                    self.env.close()
            finally:
                try:
                    if self.app is not None:
                        self.progress("closing_app")
                        self.app.close(wait_for_replicator=False)
                finally:
                    if self.hand_work is not None:
                        self.progress("closing_assets")
                        self.hand_work.cleanup()
        finally:
            try:
                self.progress("closing_context")
            finally:
                self._contexts.close()
