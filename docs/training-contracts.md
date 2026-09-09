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
- Editing a preset value switches the form to Custom. The submitted custom values
  stay explicit; choosing a preset restores its defaults.
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
adapters from registry declarations. DP and ACT have registered training adapters;
the shared XPolicyLab HDF5 remains an intermediate export format.
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
model architecture and requires a new run.

## ACT and recorded-task evaluation

ACT uses the pinned upstream CVAE Transformer, its L1/KL objective and AdamW.
The registered HDF5 split and training-only normalization feed aligned RGB/joint
samples and forward action chunks. Padding is masked; no one-frame action shift
is applied. The versioned preset declares architecture and training settings.
Best/latest model files, applied settings, loss logs and a final result are saved;
the best checkpoint is registered for inference. ACT training resume is not
advertised because these model checkpoints do not contain optimizer state.

Both adapters expose the same DexVerse recorded-task evaluator. The evaluation
suite binds its task from the immutable training bundle. Policy inference stays
in the policy runtime and communicates with a separate Isaac runtime on the same
allocated GPU. Evaluation checks source revisions, checkpoint/dataset checksums,
joint order, action scale/offset, control rate and RGB camera calibration. It runs
closed-loop episodes with the task's success criterion held for ten consecutive
steps, records videos and publishes episode success and aggregate success rate.
Completed episodes can be recovered from an identity-checked ledger.

The simulator bridge currently reconstructs native DexVerse robot configurations.
It does not reconstruct imported hand bundles or arbitrary scene overrides.
Isaac evaluation requires the operator's existing license acceptance to be
configured in the server environment (`OMNI_KIT_ACCEPT_EULA=YES`). Keep that
operator setting outside source control; the local server can load it from its
private `data/operator.env` file with `--env-file`. Readiness runs a GPU scene,
reset, camera and video check. Each simulator process uses a private temporary
directory so another cluster user's IsaacLab logs cannot block startup.

Validation on 2026-09-09: browser-submitted two-epoch jobs 3796701 (DP state,
51 real episodes) and 3796686 (ACT RGB, two real episodes) completed with losses
and registered checkpoints. Full-session ACT preparation produced 51 episodes
(41 train / 10 validation). GPU inference test 3796722 loaded both checkpoints,
produced finite 16x28 / 50x28 action chunks and verified seeded resets.
Browser-submitted ACT A40 training job 3800608 also completed at 2/2 epochs.
Real simulator evaluations 3800601 (DP state) and 3800610 (ACT RGB) each completed
one 1,200-step episode, published 1/1 progress and a task-success result, and
produced a 20-second video loaded and scrubbed in the browser. Neither short-test
checkpoint picked up the cube. These tests establish execution, not policy quality
or benchmark success. DP RGB rollout is supported by the same bridge but was not
part of these end-to-end tests. Backend regression tests: 525 passed; all live UI
regression suites passed.

GPU readiness passed on `heistotron`, and both rollouts passed on `consu`
(NVIDIA driver 580.178.04). A readiness attempt on `voltron` with driver 610.57.04
crashed inside the RTX renderer before scene creation. That node's runtime/driver
issue remains unresolved; a successful readiness check on one node does not
certify every cluster node. No cluster driver or node exclusions were changed.

W&B uploads batch queued history samples without changing their steps or timestamps.
Rate limits persist a retry delay across restarts. Reconciliation retries queued
metrics even after training finishes, so an upload delay does not require rerunning
training.
