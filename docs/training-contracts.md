# Adapter training contracts

Adapter manifests are the source of truth for trainer requirements. The existing
`train.input_fields`, `data_binding`, `supported_canonical_fields` and argument
mappings remain the submission path.

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
  ACT capsules additionally write `applied-settings.json`; their declared JSONL
  progress source feeds progress/ETA and the existing tracking bridge.

The conversion catalog derives adapter requirements from registry declarations.
Prepared datasets reference shared recording streams; adapter-specific loaders
consume those streams without duplicating the original recording payloads.
No arbitrary conversion to GR00T/OpenPI/Robomimic is claimed: those need their
actual observation, action, embodiment and configuration mappings.

## ACT and recorded-task evaluation

ACT uses the pinned upstream CVAE Transformer, its L1/KL objective and AdamW.
The registered HDF5 split and training-only normalization feed aligned RGB/joint
samples and forward action chunks. Padding is masked; no one-frame action shift
is applied. The versioned preset declares architecture and training settings.
Best/latest model files, applied settings, loss logs and a final result are saved;
the best checkpoint is registered for inference. ACT training resume is not
advertised because these model checkpoints do not contain optimizer state.

ACT adapters expose the DexVerse recorded-task evaluator. The evaluation
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

Validation on 2026-09-09: browser-submitted two-epoch ACT RGB job 3796686
(two real episodes) completed with losses and a registered checkpoint.
Full-session ACT preparation produced 51 episodes (41 train / 10 validation).
GPU inference test 3796722 produced finite 50×28 ACT action chunks and verified
seeded resets. Browser-submitted ACT A40 training job 3800608 completed at 2/2
epochs. ACT simulator evaluation 3800610 completed one 1,200-step episode,
published 1/1 progress and a task-success result, and produced a 20-second video
loaded and scrubbed in the browser. The short-test checkpoint did not pick up
the cube. These tests establish execution, not policy quality or benchmark success.

GPU readiness passed on `heistotron`, and the ACT rollout passed on `consu`
(NVIDIA driver 580.178.04). A readiness attempt on `voltron` with driver 610.57.04
crashed inside the RTX renderer before scene creation. That node's runtime/driver
issue remains unresolved; a successful readiness check on one node does not
certify every cluster node. No cluster driver or node exclusions were changed.

W&B uploads batch queued history samples without changing their steps or timestamps.
Rate limits persist a retry delay across restarts. Reconciliation retries queued
metrics even after training finishes, so an upload delay does not require rerunning
training.

GPU statistics use the shared training runner. New capsules freeze a dependency-free
collector that samples allocated NVIDIA GPUs every 15 seconds and stores JSONL
under `state/gpu-stats/<slurm-job-id>/<node>.jsonl`. CUDA device identifiers take
precedence over Slurm GRES indices, which can differ on this cluster. Collection
failures do not stop training. Existing running capsules can be sampled about once
a minute through a bounded, overlapping Slurm step without restarting the trainer.

The W&B bridge publishes utilization, memory, power, temperature and clock readings
to the System stream, with independent offsets and the existing durable queue and
rate-limit retry behavior. Native W&B adapters retain ownership of their SDK's
System stream. Collection does not change batch size or other training settings.

Validation on 2026-09-09: all four A40s in training job 3802587 produced real GPU
samples, and an authenticated W&B System-history read returned the stored readings.
The in-app browser was not signed in to W&B, so chart display was not visually
verified. GPU collection, stream separation, retry, replay and existing training
progress regression checks passed (78 tests).
