# Native UniDex runtime

Pinned source: `unidex-ai/UniDex` at `97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d`.
Model: official `PointCloudUniDexTrain` / `PointCloudUniDexInference` (PaliGemma + Uni3D), with the original 82D flow-matching loss. This adapter does not provide a simulator rollout evaluator yet.

## Isolated installation

`bootstrap_unidex.py --environment <new-prefix> --source <new-checkout> --python <existing-python3.11> --loader-only` installs only the packages needed for CPU Convert validation. It never changes the supplied Python environment or downloads model weights. It refuses existing target environments without its ownership marker.

Omit `--loader-only` and use `--install-cuda-toolkit` to install the SHA-verified NVIDIA CUDA 12.6 compiler, headers and runtime under this isolated environment (about 52 MB downloaded). This does not install or change a system driver. Alternatively run in a compatible CUDA-toolkit environment to install the full pinned runtime and compile PyTorch3D at `33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba` (v0.7.9). The bootstrap verifies imports, not model quality or GPU training readiness. It saves the exact installed package inventory and lock checksum. Use `.skynet-unidex-runtime.json` to distinguish `LOADER_READY` from `NATIVE_IMPORTS_READY`; neither status claims a full pretrained forward pass.

The direct requirements are in `.in` files. The Linux Python 3.11 `.lock` files include exact transitive versions and package hashes. Regenerate explicitly with `uv pip compile --python-version 3.11 --python-platform x86_64-manylinux_2_28 --generate-hashes` when changing the runtime.

## CPU data validation

The frozen `unidex_runtime.py` accepts:

```
--repository <pinned-source> --dataset <manifest-directory>
--manifest-sha <sha256> --output <validation-directory> --verify-only
```

No Torch, GPU, tokenizer or pretrained weights are needed for this path. It verifies immutable shared HDF5 references, complete disjoint episode membership, exact FAAS and pointcloud shapes, control period, action semantics and sample conversion. It saves `loader-validation.json`.

Shared pointcloud arrays use ROS optical XYZ and RGB in [0,1]. The reader flips Y/Z in memory for native OpenGL coordinates. FAAS absolute wrist poses use the same OpenGL camera frame. Every action in a sampled chunk is anchored to the one current measured wrist pose. Actions retain controller-target semantics, rather than substituting future measured state. Incomplete trailing chunks are excluded, with no padding or new masked loss.

## Pretrained assets

Training requires explicit `--base-weights`, `--pointcloud-weights`, and `--weights-provenance`. Tokenizer/network auto-download is disabled. The provenance JSON is:

```json
{
  "schema": "skynet.unidex-weights/v1",
  "base": {
    "origin": "repository/model id and immutable source revision",
    "files": {"model.safetensors": "sha256", "tokenizer.json": "sha256"}
  },
  "pointcloud": {"origin": "repository/model id and immutable source revision", "sha256": "sha256"}
}
```

`base.files` must cover every file in the provided local PaliGemma directory, including tokenizer/config files. PaliGemma language tensors and the Uni3D encoder must completely match the pinned model shapes. Missing tensors never silently retain random weights. The published `model-b.pt` filename is not evidence that a checkpoint matches the configured large encoder.

`--initialize-checkpoint` plus `--initialize-checkpoint-sha` means weights-only initialization. `--resume-checkpoint` means a full Lightning restart and requires the new adapter's exact dataset/model/normalizer/training identity plus optimizer and scheduler state. On resume the original initialization flags may remain in the frozen command; their fingerprint must match the saved initialization receipt and the original weights are not loaded again. No old dataset-format or checkpoint compatibility is implemented. Existing history and weight files remain intact.

The asset-bound FAAS maps cover all seven registered captures: legacy Shadow, Allegro V4, LEAP V1, Inspire RH56, WUJI1, WUJI2 and Sharpa Wave. Shadow/Allegro/LEAP/Inspire follow audited original UniDex mappings. WUJI1, WUJI2 and Sharpa are explicit Skynet anatomical extensions using the original FAAS slots and each registered asset's own geometry, signs, rest convention and wrist frame; they are not upstream asset aliases or claims of pretrained transfer. See `docs/wuji1-faas.md`, `docs/wuji2-faas.md` and `docs/sharpa-faas.md`. A released full UniDex checkpoint may already have trained on a supposedly held-out hand; its use must not be described as unseen-embodiment training.
