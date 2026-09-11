"""Supported native EgoVerse algorithms and their pinned configuration names."""

import re

MODEL_LEARNING_RATES = {
    "act": 5e-5,
    "hpt_bc_keypoints_base": 5e-5,
    "hpt_cotrain_enc_dec_base": 5e-5,
    "hpt_bc_flow_aria": 3e-4,
    "hpt_bc_flow_human": 3e-4,
    "hpt_bc_flow_mecka": 3e-4,
    "hpt_bc_flow_scale": 3e-4,
    "hpt_cotrain_flow_seperate_head": 3e-4,
    "hpt_bc_pickplace_qwen_pertoken": 2e-4,
    "hpt_bc_pickplace_qwen_pooled": 2e-4,
}

# The incomplete PI base and the EgoBridge alias are not duplicate models.
MODELS = {
    "act": ("ACT", True),
    "hpt_joints": ("HPT flow · recorded joints", True),
    **{
        name: (label, False)
        for name, label in [
            ("hpt_bc_flow_aria", "HPT flow · Aria"),
            ("hpt_bc_flow_eva", "HPT flow · EVA"),
            ("hpt_bc_flow_human", "HPT flow · human"),
            ("hpt_bc_flow_mecka", "HPT flow · Mecka"),
            ("hpt_bc_flow_scale", "HPT flow · Scale"),
            ("hpt_bc_keypoints_base", "HPT flow · human keypoints"),
            ("hpt_cotrain_enc_dec_base", "HPT co-training · encoder-decoder"),
            ("hpt_bc_pickplace_qwen_pertoken", "HPT · Qwen per token"),
            ("hpt_bc_pickplace_qwen_pooled", "HPT · Qwen pooled"),
            ("hpt_cotrain_flow_seperate_head", "HPT co-training · separate heads"),
            ("hpt_cotrain_flow_shared_head", "HPT co-training · shared head"),
            ("hpt_cotrain_mecka_flow_shared_head", "HPT co-training · Mecka"),
            ("hpt_cotrain_scale_flow_shared_head", "HPT co-training · Scale"),
            ("pi0.5_bc_aria", "π0.5 · Aria"),
            ("pi0.5_bc_eva", "π0.5 · EVA"),
            ("pi0.5_bc_mecka", "π0.5 · Mecka"),
            ("pi0.5_bc_scale", "π0.5 · Scale"),
            ("pi0.5_cotrain_eva_aria", "π0.5 co-training · EVA / Aria"),
            ("pi0.5_cotrain_mecka_scale", "π0.5 co-training · Mecka / Scale"),
        ]
    },
}


ALGORITHMS = {
    "act": ("ACT", ["act"]),
    "hpt": ("HPT", [name for name in MODELS if name.startswith("hpt")]),
    "pi": ("PI", [name for name in MODELS if name.startswith("pi")]),
}
NATIVE_TARGETS = {name: f"egomimic.algo.{name}.{label}" for name, (label, _) in ALGORITHMS.items()}
REMOVED_MODEL_MESSAGE = (
    "The synthetic EgoVerse Diffusion Policy adapter was removed. "
    "Its historical results remain available; choose native ACT, HPT, or PI for new training, resume, or evaluation."
)


def model_algorithm(model):
    if model == "dp_joints":
        raise ValueError(REMOVED_MODEL_MESSAGE)
    for algorithm, (_, models) in ALGORITHMS.items():
        if model in models:
            return algorithm
    raise ValueError("Choose a declared native EgoVerse model preset")


def model_contracts(model):
    model_algorithm(model)
    return (["skynet.egoverse-rgb-joints/v1"] if MODELS[model][1] else [f"egoverse.native-{model}/v1"])


def retired_manifest(source):
    """Recognize the removed adapter even when its old manifest was cloned."""
    manifest = source.get("adapter_manifest") or {}
    if source.get("adapter") == "egoverse-dp-joints" or manifest.get("slug") == "egoverse-dp-joints":
        return True
    argv = (manifest.get("train") or {}).get("argv") or []
    return any("egoverse_runtime.py" in str(value) for value in argv) and (
        "dp_joints" in argv or "--model=dp_joints" in argv
    )


def execution_compatibility_error(source, native_config):
    """Block known defective frozen capsules without rewriting their history."""
    if retired_manifest(source):
        return REMOVED_MODEL_MESSAGE
    if native_config.get("model_preset") != "hpt_joints":
        return None
    manifest = source.get("adapter_manifest") or {}
    files = (manifest.get("train") or {}).get("capsule_files") or {}
    runtime = files.get("adapter-support/egoverse_runtime.py", "")
    if re.search(r'''["']state_ee_pose["']\s*:\s*["']joint_positions["']''', runtime):
        return (
            "This pinned EgoVerse HPT adapter omits joint-state inputs. "
            "Select the latest EgoVerse HPT adapter and start a fresh training run; "
            "do not resume or rerun this pinned variant."
        )
    return None
