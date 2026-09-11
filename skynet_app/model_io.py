"""Declarative policy I/O, resolved from immutable model settings and dataset metadata.

This module is the single resolver for adapter inspection, submission previews,
and attempt receipts. It performs no filesystem or repository inspection.
"""
from copy import deepcopy
from typing import Any
from pydantic import Field
from .experiments import CanonicalModel
from .training_contracts import lookup


class IOAxis(CanonicalModel):
    name: str
    value: int | None = Field(default=None, ge=1)
    paths: list[str] = Field(default_factory=list)
    length: bool = False


class IOStream(CanonicalModel):
    name: str
    modality: str
    axes: list[IOAxis] = Field(default_factory=list)
    cameras: list[str] = Field(default_factory=list)
    when: dict[str, list[Any]] = Field(default_factory=dict)
    note: str = ""


class ModelIOContract(CanonicalModel):
    inputs: list[IOStream] = Field(default_factory=list)
    outputs: list[IOStream] = Field(default_factory=list)
    note: str = ""


def axis(name, value=None, *paths, length=False):
    return IOAxis(name=name, value=value, paths=list(paths), length=length)


def recorded_joint_io(*, history=1, action_steps=100, history_paths=(), action_paths=(), observation_selector=None, image_size=None):
    history_axis = axis("timesteps", history, *history_paths)
    state_axis = axis("joints", None, "dataset.policy_to_source_indices", "dataset.capture.robot_joint_names", length=True)
    action_axis = axis("values", None, "dataset.policy_to_source_indices", "dataset.capture.action_joint_names", length=True)
    rgb = IOStream(name="RGB", modality="RGB", cameras=["scene_front", "scene_left", "scene_right"],
                   axes=[history_axis, axis("height", image_size[0]) if image_size else axis("height", None, "camera.height"), axis("width", image_size[1]) if image_size else axis("width", None, "camera.width"), axis("channels", 3)],
                   when={observation_selector: ["rgb"]} if observation_selector else {})
    return ModelIOContract(inputs=[rgb, IOStream(name="Joint positions", modality="state", axes=[history_axis, state_axis])],
        outputs=[IOStream(name="Joint commands", modality="actions", axes=[axis("steps", action_steps, *action_paths), action_axis])])


def adapter_io_contract(slug):
    """Built-in declarations; configurable dimensions always prefer saved values."""
    if slug == "xpolicylab-dp":
        return recorded_joint_io(history=2, action_steps=16, image_size=(240, 320),
            history_paths=("spec.native.config.observation_steps",), action_paths=("spec.native.config.action_steps",),
            observation_selector="native.config.observation_mode")
    if slug == "xpolicylab-act":
        return recorded_joint_io(action_steps=50, image_size=(480, 640), action_paths=("spec.native.config.action_steps",))
    if slug in {"egoverse-hpt", "egoverse-pi"}:
        from .adapters.egoverse_models import ALGORITHMS

        contract = ModelIOContract(note="Input and output follow the selected native model preset.")
        for model in ALGORITHMS[slug.removeprefix("egoverse-")][1]:
            preset = adapter_io_contract("egoverse-" + model.replace("_", "-").replace(".", ""))
            for direction in ("inputs", "outputs"):
                for stream in getattr(preset, direction):
                    stream.when["native.config.model_preset"] = [model]
                    getattr(contract, direction).append(stream)
        return contract
    if slug in {"egoverse-act", "egoverse-hpt-joints"}:
        is_act = slug == "egoverse-act"
        return recorded_joint_io(
            history_paths=() if is_act else ("spec.native.config.model_overrides.robomimic_model.trunk.observation_horizon",),
            action_paths=("spec.native.config.model_overrides.robomimic_model.chunk_size",) if is_act else
                ("spec.native.config.model_overrides.robomimic_model.head_specs.skynet_joints.action_horizon",))
    if slug and slug.startswith("egoverse-"):
        return native_egoverse_io(slug.removeprefix("egoverse-"))
    return None


def native_egoverse_io(model):
    """I/O of the pinned e17cf98 native configs, before batch stacking."""
    prefix = "spec.native.config.model_overrides.robomimic_model."
    if model.startswith("pi05-"):
        return ModelIOContract(inputs=[
            IOStream(name="RGB camera slots", modality="RGB", axes=[axis("views", 3), axis("height", 224), axis("width", 224), axis("channels", 3)], note="Missing views are masked."),
            IOStream(name="Proprioception", modality="state", note="Embodiment-specific state, also encoded in the prompt."),
            IOStream(name="Language prompt", modality="language", axes=[axis("maximum tokens", 128, prefix + "tokenizer_max_length")]),
        ], outputs=[IOStream(name="Padded action tensor", modality="actions", axes=[axis("steps", 100, prefix + "config.model.action_horizon"), axis("padded values", 32, prefix + "config.model.action_dim")])],
        note="Images resized to 224 × 224; missing views masked. Actions are decoded for each embodiment.")
    if not model.startswith("hpt-"):
        return None
    cotrain = "cotrain" in model
    qwen = "qwen" in model
    human = cotrain or not (model == "hpt-bc-flow-eva" or qwen)
    eva = cotrain or not human
    keypoints = "keypoints" in model
    shared = "shared-head" in model
    history = axis("timesteps", 1, prefix + "trunk.observation_horizon")
    cameras = ["front_img_1"] + (["left_wrist_img", "right_wrist_img"] if eva and not qwen else [])
    inputs = [IOStream(name="RGB", modality="RGB", cameras=cameras, axes=[history, axis("height", None, "camera.height"), axis("width", None, "camera.width"), axis("channels", 3)])]
    outputs = []
    for domain, enabled, dimensions in [("human_bimanual", human, 138 if keypoints else 12), ("eva_bimanual", eva, 14)]:
        if not enabled:
            continue
        state_key = "state_keypoints" if keypoints else "state_joint_positions" if cotrain and domain == "eva_bimanual" else "state_ee_pose"
        inputs.append(IOStream(name=domain + " · " + state_key, modality="state", axes=[history, axis("values", dimensions, prefix + f"stem_specs.{domain}.{state_key}.input_dim")]))
        head = "shared" if shared else domain
        # The shared head predicts 14 values even for the 12-value human state.
        output_dim = 14 if shared else dimensions
        if model == "hpt-cotrain-enc-dec-base":
            output_dim = None
        outputs.append(IOStream(name=domain + " actions", modality="actions", axes=[axis("steps", 100, prefix + f"head_specs.{head}.action_horizon"), axis("values", output_dim, prefix + f"head_specs.{head}.infer_ac_dims.{domain}")]))
    if qwen:
        inputs.append(IOStream(name="Language annotation", modality="language", axes=[axis("maximum tokens", 128, prefix + "shared_stem_specs.annotation.max_length")]))
    return ModelIOContract(inputs=inputs, outputs=outputs)


def _set_path(document, path, value):
    parts = path.split(".")
    if any(part in {"__proto__", "prototype", "constructor"} for part in parts):
        return
    target = document
    for part in parts[:-1]:
        if not isinstance(target.get(part), dict):
            target[part] = {}
        target = target[part]
    target[parts[-1]] = deepcopy(value)


def resolve_model_io(manifest, spec=None, *, legacy=False):
    manifest = manifest or {}
    spec = deepcopy(spec or {})
    train = manifest.get("train") or {}
    # UI previews can be incomplete. Never overwrite a user's choice (including 0).
    for field in train.get("input_fields") or []:
        if field.get("default") is not None and lookup(spec, field["path"]) is None:
            _set_path(spec, field["path"], field["default"])
    contract = train.get("model_io")
    if not contract and legacy:
        # Older receipts predate I/O declarations. Only infer our recorded-joint
        # integrations; use the receipt's own settings and dataset, not the registry.
        if manifest.get("slug") == "egoverse-dp-joints":
            # This retired adapter is still describable in immutable history.
            contract = adapter_io_contract("egoverse-hpt-joints")
        elif manifest.get("slug") in {"xpolicylab-dp", "xpolicylab-act", "egoverse-act", "egoverse-hpt-joints"}:
            contract = adapter_io_contract(manifest["slug"])
    if isinstance(contract, ModelIOContract):
        contract = contract.model_dump(mode="json")
    contract = ModelIOContract.model_validate(contract) if contract else None
    requirements = train.get("data_requirements") or {}
    assignments = lookup(spec, "data.bundle.assignments") or []
    assignments = [a for a in assignments if a.get("role") == "training_data"]
    metadata = assignments[0].get("version", {}).get("metadata", {}) if len(assignments) == 1 else {}
    # A hand-entered path must not silently inherit another dataset's dimensions.
    if metadata:
        configured = lookup(spec, "native.config.dataset_path")
        actual = lookup(assignments[0], "config.location.path") or lookup(assignments[0], "version.path")
        fingerprint = lookup(spec, "native.config.dataset_manifest_sha256")
        actual_fingerprint = lookup(assignments[0], "version.manifest_sha256")
        if (configured and configured != actual) or (fingerprint and fingerprint != actual_fingerprint):
            metadata = {}
    if not contract:
        observations = requirements.get("observations") or ["Configured observations"]
        return {"schema_version": "skynet.model-io/v1", "source": "requirements", "resolved": False,
                "entries": [["Input · " + item.replace("_", " "), "Size defined by the model / dataset"] for item in observations]
                + [["Output · " + (requirements.get("action_representation") or "actions").replace("_", " "), "Size defined by the model / dataset"]],
                "note": "Exact dimensions are not declared for this adapter."}
    overrides = lookup(spec, "native.config.model_overrides")
    if isinstance(overrides, dict):
        expanded = {}
        for path, value in overrides.items():
            _set_path(expanded, path, value)
        _set_path(spec, "native.config.model_overrides", expanded)
    context = {"spec": spec, "dataset": metadata}
    entries, resolved = [], True
    for direction in ("inputs", "outputs"):
        for stream in getattr(contract, direction):
            if any(lookup(spec, path) not in choices for path, choices in stream.when.items()):
                continue
            for camera_name in stream.cameras or [None]:
                context["camera"] = lookup(metadata, "capture.cameras." + camera_name) or {} if camera_name else {}
                shape, names = [], []
                for dim in stream.axes:
                    value, specified = None, False
                    for path in dim.paths:
                        candidate = lookup(context, path)
                        if candidate is None:
                            continue
                        specified = True
                        if dim.length and isinstance(candidate, (list, dict)):
                            candidate = len(candidate)
                        if type(candidate) is int and candidate > 0:
                            value = candidate
                        break
                    if not specified:
                        value = dim.value
                    resolved &= value is not None
                    shape.append(str(value) if value is not None else "?")
                    names.append(dim.name)
                name = stream.name + (" · " + camera_name if camera_name else "")
                size = " × ".join(shape) + " (" + " × ".join(names) + ")" if shape else stream.note or "Variable length"
                if shape and stream.note:
                    size += "; " + stream.note
                entries.append([("Input" if direction == "inputs" else "Output") + " · " + name, size])
    return {"schema_version": "skynet.model-io/v1", "source": "saved_configuration" if legacy else "declared_configuration",
            "resolved": bool(resolved), "entries": entries,
            "note": " ".join(filter(None, [contract.note, "Per sample; batch dimension omitted." if resolved else "? = dimension not recorded or no dataset selected."]))}


def preview_spec(values, bundle=None):
    spec = {"data": {"bundle": bundle}} if bundle else {}
    for path, value in (values or {}).items():
        if path.startswith(("native.config.", "native.overrides.", "train.")):
            _set_path(spec, path, value)
    return spec
