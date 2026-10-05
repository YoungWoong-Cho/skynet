# Native UniDex runtime

Pinned source: `unidex-ai/UniDex` at `97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d`.
Model: official `PointCloudUniDexTrain` / `PointCloudUniDexInference` (PaliGemma + Uni3D), with the original 82D flow-matching loss. Skynet supplies a separate DexVerse simulator rollout evaluator around the official inference model.

## Native configuration and GPU memory

Skynet reads all pinned upstream YAML with OmegaConf, matching official Hydra
scalar parsing and interpolation. Bare scientific notation such as `1e-8` must
remain numeric in optimizer and model settings, including checkpoint receipts.

The official example uses FP32 with eight H800 GPUs. A single 48 GB L40S cannot
hold this full model, gradients and AdamW states in FP32: the first optimizer
update was observed to exhaust memory even after forward/backward succeeded.
Adding ordinary DDP replicas does not divide these states between GPUs.

Skynet adds the explicit `native.config.distributed_strategy` option:

- `ddp`: official DDP strategy for multiple GPUs, ordinary execution on one.
- `fsdp`: Lightning FSDP on at least two GPUs, partitioning model, gradient and
  optimizer storage. This is a Skynet execution change; the official model,
  objective, pretrained initialization and selected precision remain unchanged.
  Full checkpoint files retain the portable model and optimizer state contract.

Keep effective batch size fixed when changing GPU count:
`per-GPU batch × GPU count × gradient accumulation`. For example, batch 4,
four GPUs and accumulation 1 preserve the effective batch of 16 from batch 4,
one GPU and accumulation 4. The distributed strategy is pinned in the run identity.

New Skynet submissions default to the tested L40S profile: FSDP, automatic
recommended allocation of four GPUs, batch 4 per GPU and accumulation 1,
128 GB host RAM and 12 CPUs. This is a Skynet resource preset, not an official
UniDex training requirement. Explicit GPU counts, other GPU types and DDP
remain configurable; the recommendation is not a universal VRAM guarantee.

Skynet's checkpoint policy uses the common submission fields `save_every_steps`
and `keep_last` (defaults: 1,000 updates and three checkpoints). Numbered
step checkpoints contain full training state; `last.ckpt` links to the newest
file without a second serialization. The final completed update is saved even
when training ends mid-epoch. These are Skynet checkpoint policies, independent
of the official model and loss. Validation runs at epoch boundaries and does
not determine which checkpoint is considered latest.

### Optional Skynet execution settings

The original UniDex example uses FP32 and DDP. Skynet's FSDP resource preset
keeps FP32, `float32_matmul_precision=highest`,
`fsdp_sharding_strategy=FULL_SHARD`, and `fsdp_parameter_dtype=auto` as defaults.
Explicitly selecting those defaults preserves existing checkpoint identities.

Skynet adds two optional acceleration modes: FP32 with
`native.config.float32_matmul_precision=high` (Tensor Core matrix operations),
or `train.precision=bf16` with the matrix precision field left at `highest`.
These change numerical execution precision, not the original model, objective,
optimizer, or selected global batch. Pointcloud input coordinates and original
FPS/KNN grouping remain FP32 with autocast and TF32 disabled for that boundary.
BF16 also keeps gradient reduction, BatchNorm parameters/statistics, and root
inputs in FP32. The native embedding assembly receives matching FP32 outputs.

With FSDP, the separate Skynet option
`native.config.fsdp_sharding_strategy=SHARD_GRAD_OP` retains unsharded
parameters through accumulation. Skynet's strategy enters the root FSDP
`no_sync()` context around the forward and backward of nonfinal microbatches,
deferring gradient synchronization until the optimizer boundary. The default
Lightning `FULL_SHARD` path does not use this override. Retaining parameters
and unsynchronized gradients can require substantially more GPU memory.

Under BF16 with `fsdp_parameter_dtype=auto`, local gradients accumulated in
`no_sync()` are BF16. The FP32 `reduce_dtype` applies to the subsequent
distributed reduction; it does not make that preceding local accumulation
FP32. The optional `native.config.fsdp_parameter_dtype=float32` retains FP32
parameters and accumulated gradients while using BF16 autocast for eligible
operations. It requires both BF16 and FSDP and uses more memory. Buffers and
gradient reduction stay FP32 in either BF16 configuration.

These are Skynet execution changes, not original UniDex defaults, and do not
claim numerical equivalence to the FP32 baseline or between sharding modes.
Nondefault precision and sharding recipes are pinned in checkpoint identity;
resuming with a different recipe is rejected. Evaluation continues to use the
existing FP32 inference path and does not automatically inherit training
precision or sharding settings.

`native.config.validation_enabled=false` skips validation data loading,
validation inference, and the sanity check. It preserves the existing training
split and does not move held-out episodes into training. This is a Skynet
execution option; the default remains the original epoch-based validation.

## Isolated installation

`bootstrap_unidex.py --environment <new-prefix> --source <new-checkout> --python <existing-python3.11> --loader-only` installs only the packages needed for CPU Convert validation. It never changes the supplied Python environment or downloads model weights. It refuses existing target environments without its ownership marker.

Omit `--loader-only` and use `--install-cuda-toolkit` to install the SHA-verified NVIDIA CUDA 12.6 compiler, headers and runtime under this isolated environment (about 52 MB downloaded). This does not install or change a system driver. Alternatively run in a compatible CUDA-toolkit environment to install the full pinned runtime and compile PyTorch3D at `33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba` (v0.7.9). The bootstrap verifies imports, not model quality or GPU training readiness. It saves the exact installed package inventory and lock checksum. Use `.skynet-unidex-runtime.json` to distinguish `LOADER_READY` from `NATIVE_IMPORTS_READY`; neither status claims a full pretrained forward pass.

The direct requirements are in `.in` files. The Linux Python 3.11 `.lock` files include exact transitive versions and package hashes. Regenerate explicitly with `uv pip compile --python-version 3.11 --python-platform x86_64-manylinux_2_28 --generate-hashes` when changing the runtime.

## CPU data validation

The official configured UniDex dataset uses 10,000 input points. New Skynet
conversions now request that count; Uni3D's separate feature dimension remains
1,024. Skynet readers and checkpoints pin the complete observation recipe.
Existing 1,024-point datasets keep their saved recipe, and mixing recipes in
training or between a checkpoint and an evaluation target is rejected.
Verified RGB/depth captures can be reused across conversion-worker revisions
without relabeling their provenance. Derived point clouds separately pin the
current worker revision, so a calculation change cannot reuse a stale result.

Skynet also pads training epoch draws to a whole optimizer batch when using
gradient accumulation. This repeats selected windows, not frames inside an
action chunk. The sampler receipt records the exact padding: 877 windows with
four ranks, microbatch four and accumulation eight become 896 draws, or seven
complete updates of 128 samples. Validation data is unchanged.

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

## Simulator evaluation

The original UniDex inference model predicts 82D FAAS action chunks. Skynet adds
the frozen target-hand FAAS inverse mapping, live RGB/depth-to-pointcloud
observations, separate policy and Isaac Lab processes, and episode accounting.
The simulator uses the target recording's verified hand assets, camera recipe,
and pinned DexVerse source. It reloads checksum-verified manifests at runtime;
submitted context files contain compact dataset identities while the database
retains complete planning provenance.

An unseen-hand evaluation explicitly selects a target dataset and rejects a
target hand present in any training input. This is a Skynet evaluation contract,
not an original UniDex training option. Standard task evaluation starts from
seeded simulator resets; the distinct training-episode suite replays a saved
initial state and must not be substituted for unseen-hand evaluation.

The original DexVerse task defines its episode duration and success predicate.
Skynet additionally requires ten consecutive simulator steps satisfying that
predicate. A completed evaluation can have zero successful episodes:
execution status and task success rate are reported separately.
