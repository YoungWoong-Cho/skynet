# XPolicyLab adapters

The audited source is [XPolicyLab at 9c98a3aaf02d](https://github.com/XPolicyLab/XPolicyLab/tree/9c98a3aaf02d05c6f9999a5a0a7a42090555ddf3/policy).
There are 44 policy directories, excluding the `demo_policy` template. 38 contain a native training entrypoint. Their manifests appear in **Experiments → Adapters** with the suffix **Native**.

## Existing ACT and DP

**ACT · Skynet recordings** imports the original `ACTPolicy`, including its transformer and L1/KL loss. Its major model defaults match `policy/ACT/train.sh`: 6000 epochs, chunk 50, hidden dimension 512, feedforward dimension 3200, KL weight 10, learning rate 1e-5 and batch 16. Skynet supplies the RGB/joint loader, training-only normalization, split handling, training loop, checkpoint format and distributed execution. It is a bridge for Skynet recordings, not a call to upstream `train.sh`.

**Diffusion Policy · Skynet recordings** uses XPolicyLab diffusion components but has a custom future-action window and state encoder, loader, training loop and fixed EMA. Its state/RGB presets are not upstream DP defaults. The original RGB recipe has horizon 8, observation history 3, 6 executed actions, 600 epochs, batch 128, 100 inference steps, cosine scheduling with 500 warmup steps, weight decay 1e-6 and dynamic EMA. Existing Skynet defaults are 300 epochs, observation history 2, a 16-step future-action chunk, 20 inference steps, weight decay 1e-4 and fixed EMA 0.995. Existing pinned runs and data contracts are preserved.

**ACT · Native** and **DP · Native** invoke the original launchers. They need the original input layout; they do not accept a Skynet converted dataset merely because its extension is HDF5 or Zarr.

## Native policy coverage

Training entrypoints are registered for A1, Abot_M0, ACT, AHA_WAM, Being_H05, Dexbotic_DM0, DP, DreamZero, EventVLA, FastWAM, G05, GalaxeaVLA, GigaWorldPolicy, GO1, GR00T_N17, H_RDT, Hy_Embodied_05_VLA, InternVLA_A1, LDA_1B, LingBot_VA, LingBot_VLA, Mem_0, OLA_SEM, OpenVLA_OFT, OpenWAM, Pi_0, Pi_05, Pi_0_Fast, RDT_1B, RISE, SmolVLA, Spirit_v15, starVLA, TinyVLA, X_VLA, X_WAM, Xiaomi_Robotics_0 and Xiaomi_Robotics_1.

Dexora_1B, InternVLA_A1_5, Meituan_Robotics_0, MolmoAct2, OpenDM and Spatial_Forcing have no `train.sh` in this commit. They are not registered as fabricated training implementations. Inference code is not evidence of a training implementation.

Each native adapter declares its own input format, original source link, prerequisites and GPU limits. Most VLA/WAM policies need pretrained weights and a policy-specific environment. Hy-VLA additionally needs external source missing from the pinned checkout. SmolVLA's original recipe hardcodes 14 state dimensions and three RGB cameras; it cannot consume Shadow's 28-joint data as-is.

Registration means the launcher is integrated, **not** that GPU training, quality, automatic resume or simulator evaluation has been validated. Native adapters intentionally do not claim a checkpoint/resume/evaluation contract that has not been implemented. Outputs remain in the run's artifact tree for inspection.

## Native inputs

Use the policy's own `process_data.sh` and preparation instructions. Native input layouts differ: LeRobot, HDF5, Zarr, point clouds, generated data registration/configuration and model-specific statistics. Skynet's current recording converters do not produce these native formats automatically.

A registered native dataset pins the policy, source revision, launch configuration and a checksum inventory. Its `format` is `xpolicylab-native-<lowercase policy directory>/v1`; its data contract is `skynet.xpolicylab-native/v1`. For example:

```json
{
  "format": "xpolicylab-native-act/v1",
  "contract": "skynet.xpolicylab-native/v1",
  "policy": "ACT",
  "source_revision": "9c98a3aaf02d05c6f9999a5a0a7a42090555ddf3",
  "launch": {
    "bench_name": "RoboDojo",
    "task_name": "cube",
    "env_cfg_type": "arx_x5",
    "action_type": "joint",
    "environment": {},
    "required_paths": ["env_cfg", "XPolicyLab/policy/ACT/data"]
  },
  "files": {
    "env_cfg/robot/example.yaml": {"size_bytes": 123, "sha256": "replace-with-file-sha256"},
    "XPolicyLab/policy/ACT/data/example.hdf5": {"size_bytes": 456, "sha256": "replace-with-file-sha256"}
  }
}
```

This illustrates the schema, not a runnable ACT dataset. Preserve every filename/layout required by the selected policy. `files` must enumerate the real files with their real sizes and hashes. Set registry metadata `contract` and validation status only after checking the native preparation output; setting a label alone does not make data compatible. The dataset version also pins the SHA-256 of this `manifest.json`.

`launch.environment` contains the native recipe's documented options. `${WORKSPACE}` expands to the private run workspace; `${OUTPUT}` expands to the run artifacts directory. Use these placeholders for portable dataset/config/output paths. Credentials must remain in Skynet connections. GPU allocation, runtime search paths and process-count overrides cannot replace Skynet's allocation.

EventVLA instead uses `data_mix`, `memory_ablation_mode`, `keyframe_memory_policy` (`teacher` or `predict`) and optional `extra_args`. Hy-VLA uses its documented environment and optional `extra_args`. Mem_0 has `stage` (`execution` by default, `planning` or `both`); RISE has `stage` (`all` by default, `advantage` or `policy`). Extra arguments are rejected where the upstream shell API does not accept them. Standard launchers receive the experiment's seed; EventVLA/Hy-VLA keep their native seed configuration.

## Execution and isolation

1. Verify policy/source identity, allocation and all declared input checksums.
2. Export the pinned tracked policy and utilities into `artifacts/native-workspace/XPolicyLab`. Shared Git checkout changes, cached data, checkpoints and `.git` are excluded.
3. Copy listed native inputs into that workspace. Inputs cannot replace tracked executable code, and input data may be modified only in the private copy. This can consume additional cluster storage; it is not a zero-copy data loader.
4. Check required paths and native prerequisites. TinyVLA fails before its interactive weight-download prompt. Selected installed runtimes are reused; no dependency installation is performed by Skynet.
5. Apply source-hash-checked infrastructure changes in the private copy: allocated process count, local communication settings, available port, selected conda environment, and disabling SmolVLA's automatic Hub publication. Model/training parameters are untouched. The exact patch set is in `xpolicy_native_catalog.json`.
6. Call upstream `train.sh` with structured arguments and preserve Slurm's `CUDA_VISIBLE_DEVICES`. Input prompts receive EOF. Nonzero exit status and failed shell pipelines are propagated. stdout/stderr are captured by the existing Slurm logging path.
7. Record source/data identity, command, working directory, infrastructure patch hashes and exit status in `artifacts/native-launch.json`. Native checkpoints, metrics and generated files remain under the run's artifact workspace; no fabricated common metrics or checkpoint format is reported.

Absolute paths inside user-supplied native configuration remain that configuration's responsibility; prefer the workspace placeholders. Large external frameworks and pretrained assets must be prepared explicitly. The catalog does not install or download 38 training stacks, submit GPU jobs, or assert that every model fits a particular GPU.

Tests cover policy-specific argument APIs, immutable source/input handling, checksum and path failures, Slurm device mapping, infrastructure patch identity, GPU contract validation and propagation of a real child process failure. Full training requires policy-specific prepared data, weights, installed dependencies and a separate GPU validation run.
