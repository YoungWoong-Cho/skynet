# Adapter training contracts

Adapter manifests are the source of truth for trainer requirements. The existing
`train.input_fields`, `data_binding`, `supported_canonical_fields` and argument
mappings remain the submission path; there is no separate DP submission system.

- `train.data_requirements` explains the observations and action representation.
  Dataset versus simulation/custom workflows are distinguished explicitly.
- Dataset bindings can accept multiple verified contracts and select contracts
  using another declared input (for example, observation mode). The same checks
  apply to collection and imported registry versions. A matching extension alone
  is not proof of a matching robot schema. Existing bridge runtime validators
  remain responsible for their exact repository configuration and robot layout.
- `train.presets` pins a versioned set of defaults and its reference. The selected
  preset is retained in the experiment; user overrides remain explicit.
- Typed input bounds and cross-field limits reject invalid settings before launch.
  Current built-ins reject unsupported canonical overrides even if their values
  happen to equal generic schema defaults. Historical pinned versions preserve
  their previous interpretation and their manifest hashes.
- Common-parameter receipts include supported native settings and their canonical
  aliases. Unmapped schema defaults are not reported as applied hyperparameters.
  The DP capsule additionally writes `applied-settings.json` and the fully resolved
  `training-config.yaml`; its declared JSONL progress source feeds progress/ETA
  and the existing tracking bridge.

The conversion catalog advertises formats/contracts; it derives compatible
adapters from registry declarations. ACT export does not imply ACT training support.
No arbitrary conversion to GR00T/OpenPI/Robomimic is claimed: those need their
actual observation, action, embodiment and configuration mappings.

## DP presets and runtime

`dexverse-state/v1` uses joint positions without RGB, an encoder with hidden width
256 and output width 128 per observation, observation history 2, future action
chunks 16, DDPM squared-cosine training with 100 timesteps and 20 inference steps,
AdamW LR 1e-4 / weight decay 1e-4, nominal effective batch 256, gradient clipping 1,
fixed EMA 0.995, at most 300 epochs and held-out validation early stopping.
The two encoded observations are concatenated for U-Net conditioning. Patience 20,
constant LR, zero warmup, seed 42 and the pinned optimizer betas/epsilon are
implementation choices: the [paper](https://arxiv.org/html/2607.08751v1) does not
specify them. The encoder history aggregation is also explicit in the saved config.
This is a policy training preset, not a reproduction of the paper's task suite,
collection distribution, or rollout success rates.

`rgb-joints/v2` uses the same training code with the pinned three-view ResNet18
encoder and batch 8. Pixels are divided by 255, then ImageNet-normalized once.
Both modes use training-only joint/action normalization, complete episode splits,
correct accumulation (including partial batches), EMA validation, best/latest
checkpoints, and finite-loss checks. A prediction returns all 16 future actions;
observation history never consumes part of that action chunk.

New collection recordings retain joint names, command scale/offset and timing
without requiring camera capture. State conversion reads this metadata directly.
Older recordings lacking it can use their existing verified image sidecars for
joint layout, or their already prepared Zarr data can train directly in state mode.
Original recordings remain immutable; converted versions, provenance, copies and
reference-protected deletion use the existing dataset lifecycle.

Existing jobs keep their pinned capsule. Changing from RGB to state changes the
model architecture and requires a new run. Autonomous simulator evaluation is
still not integrated; this change validates training and held-out loss.
