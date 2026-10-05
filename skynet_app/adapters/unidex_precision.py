"""Versioned Skynet execution precision around the unchanged UniDex model.

The original implementation is FP32. These optional execution policies preserve
its FP32 input coordinates and grouping, while permitting lower-precision neural
network operations. They do not replace model layers, targets, or the loss.
The existing evaluator remains FP32; training precision is not inherited there.
This module is portable to a frozen training capsule and imports Torch lazily.
"""
from contextlib import contextmanager
from functools import wraps


SCHEMA = "skynet.unidex-execution-precision/v1"


def precision_recipe(precision="fp32", float32_matmul_precision="highest", fsdp_parameter_dtype="auto"):
    """Describe the explicit mode combinations used by the bounded GPU benchmark."""
    if (precision, float32_matmul_precision) not in {
        ("fp32", "highest"), ("fp32", "high"), ("bf16", "highest"),
    }:
        raise ValueError("UniDex execution precision requires FP32 highest/high or BF16 highest")
    if fsdp_parameter_dtype not in {"auto", "float32"} or (
        fsdp_parameter_dtype == "float32" and precision != "bf16"
    ):
        raise ValueError("An explicit FSDP parameter dtype requires BF16 compute with float32 parameters")
    mixed = precision == "bf16"
    recipe = {
        "schema": SCHEMA,
        "precision": precision,
        "float32_matmul_precision": float32_matmul_precision,
        "geometry": "fp32_guard" if mixed or float32_matmul_precision == "high" else "native",
        "input_dtype": "float32",
        "embedding_output_dtype": "float32" if mixed else "native",
        "fsdp_mixed_precision": {
            "param_dtype": "bfloat16", "reduce_dtype": "float32", "buffer_dtype": "float32",
            "cast_root_forward_inputs": False, "cast_forward_inputs": False,
        } if mixed else None,
        "evaluation_precision": "fp32",
    }
    # Keep every existing auto recipe byte-for-byte identical after JSON
    # serialization. Only the new explicit override extends the pinned receipt.
    if fsdp_parameter_dtype == "float32":
        recipe["fsdp_parameter_dtype"] = "float32"
        recipe["fsdp_mixed_precision"]["param_dtype"] = "float32"
    return recipe


def validate_precision_recipe(recipe):
    if not isinstance(recipe, dict):
        raise ValueError("UniDex execution precision recipe is missing")
    expected = precision_recipe(recipe.get("precision"), recipe.get("float32_matmul_precision"),
                                recipe.get("fsdp_parameter_dtype", "auto"))
    if recipe != expected:
        raise ValueError("UniDex execution precision recipe is unsupported or has changed")
    return expected


def training_precision_kwargs(recipe):
    """Additional FSDPStrategy kwargs; default FP32 keeps the existing strategy."""
    recipe = validate_precision_recipe(recipe)
    config = recipe["fsdp_mixed_precision"]
    if config is None:
        return {}
    import torch
    from torch.distributed.fsdp import MixedPrecision
    return {"mixed_precision": MixedPrecision(**{
        key: getattr(torch, value) if key.endswith("_dtype") else value
        for key, value in config.items()
    })}


@contextmanager
def _exact_geometry(device_type):
    import torch
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        with torch.autocast(device_type=device_type, enabled=False):
            yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _float_output(_module, _arguments, output):
    # The native assembly allocates from the FP32 pointcloud dtype. Indexed
    # assignment requires its embedding/projector source to have that dtype.
    return output.float()


def apply_policy_precision(policy, recipe):
    """Apply once to this initialized policy before FSDP wraps it.

    No upstream class/global function or state_dict key is modified. Matmul
    precision is process-scoped, as in PyTorch; training uses one policy per
    process. Callers pin nondefault recipes in their checkpoint identity.
    Already rounded coordinates are rejected rather than silently upcast.
    """
    import torch
    recipe = validate_precision_recipe(recipe)
    installed = getattr(policy, "_skynet_execution_precision", None)
    if installed is not None:
        if installed != recipe:
            raise ValueError("An initialized UniDex policy cannot change its execution precision")
        torch.set_float32_matmul_precision(recipe["float32_matmul_precision"])
        return recipe
    group = None
    if recipe["geometry"] == "fp32_guard":
        try:
            group = policy.pointcloud_encoder.group_divider
            original_group = group.forward
            if recipe["precision"] == "bf16":
                embeddings = (policy.embed_tokens, policy.multi_modal_projector)
        except AttributeError as error:
            raise ValueError("UniDex policy is missing the native precision boundaries") from error

        @wraps(original_group)
        def guarded_group(xyz, colors):
            if xyz.dtype != torch.float32 or colors.dtype != torch.float32:
                raise ValueError("UniDex geometry inputs lost their original FP32 precision")
            with _exact_geometry(xyz.device.type):
                return original_group(xyz, colors)

    torch.set_float32_matmul_precision(recipe["float32_matmul_precision"])
    if group is not None:
        group.forward = guarded_group
    if recipe["precision"] == "bf16":
        for module in embeddings:
            module.register_forward_hook(_float_output)
    policy._skynet_execution_precision = validate_precision_recipe(recipe)
    return recipe
