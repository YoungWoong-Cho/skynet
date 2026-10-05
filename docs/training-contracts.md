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

## Skynet submission metadata

Skynet uploads execution files separately from the Slurm script, using the
existing batch capsule uploader. A directory named by the checksum manifest's
SHA256 makes each attempt's inputs immutable. Identical uploads are reusable;
conflicting content is rejected. The script verifies the manifest and every file
before materialization and retains its exact immutable copy for submission recovery.
Training, retry and evaluation all use this shared path.

The original request remains in the experiment revision; its former on-disk
`requested-spec.json` was actually a duplicate of `resolved-spec.json` and is no
longer emitted. The adapter plan and manifest remain in the immutable attempt
snapshot instead of additional duplicate files. Runtime JSON is serialized
compactly. These are Skynet transport changes, not upstream model changes.

For validated HAT recording datasets, the derived native dataset selections
keep identities, manifest hashes, episode source hashes, hand IDs, timing and
split membership. Full render receipts and geometry remain in the authoritative
frozen dataset bundle and pinned dataset manifest. The HAT loader verifies and
loads that manifest; evaluation still receives the complete frozen geometry.
Other/custom contracts retain their existing metadata.

## Multiple datasets and submission order

The shared `data_binding` declares `cardinality: one` (the default) or `many`.
`many` uses one JSON field with `value_path: selection`, covering a complete role.
The server resolves its ordered `data_selections` into immutable bundle assignments
and derives a list of `{position, version_id, path, manifest_sha256, metadata}`.
It rejects conflicting overrides, repeated results, repeated source recordings,
and overlapping training/validation recordings. No merged dataset or new file
copy is created. Presets retain exact versions, locations and selection order.
Adapters without a data binding skip dataset selection; single-input adapters
continue to reject multiple inputs.

Submit order is Algorithm and code → Training data → Runtime → Training settings
→ GPU and time → Checkpoint and variants → Tracking and evaluation. A fieldset is
shown and enabled only after every preceding step is valid; hidden steps retain
their input values. Data checks the union of the
adapter's contracts before the later observation-mode setting exists. Training
settings checks the selected mode; Resources checks batch/GPU divisibility. The
server repeats complete checks for every sweep variant before creating a run.

The browser uses committed-change notifications to refresh catalogs. A focus or
visibility change flushes pending changes while the stream is connected; it does
not invalidate every catalog. Disconnected streams still refresh on focus and
resynchronize on reconnect.

## Multi-hand co-training and held-out hands

**Skynet integration (HAT):** HAT accepts multiple prepared datasets through the
shared binding. Conversion retains the original recording frequency; experiment
frequency and chunk length determine windows within each episode. Mixed source
frequencies need an explicit common frequency that divides every source rate.
Train/validation eligibility is checked over the whole collection, including
excluded short episodes. `window_proportional` visits all windows; the optional
`hand_balanced` sampler assigns equal hand quotas (within one sample), then samples
that hand's windows with replacement. Sampling is deterministic by seed and epoch,
with one global sequence sharded across distributed workers. Distributed tail
padding may repeat samples to keep rank lengths equal.
`train.max_steps` fixes the optimizer-update budget.

A HAT evaluation runs on a prepared HAT target dataset. Without a selected target,
a single-input run is evaluated on its own training dataset; multi-input runs and
the Unseen hand option need a separate target. A separate target fixes one hand,
task and camera, without adding its demonstrations to training. The Unseen hand
option rejects any target hand present anywhere in the run's input datasets. This
means unseen in this run's training inputs; it does not assert that pretrained
weights have never seen that hand. Compare checkpoints with the same initialization
provenance, optimizer steps, frequency, chunk length and held-out evaluation seeds.

The Skynet HAT bridge reuses the three measured RGB scene-camera calibrations from
conversion and the verified hand bundles, encodes the simulator state into HAT128
slots through the FAAS codec, and decodes predictions through HAT's
position-optimizer retargeting into the target hand's native commands. Target data and simulation assets are frozen in
the evaluation stage, included in retry identity, and protected from deletion or
retirement while referenced. Each new simulator worker must pass the existing
unscored reset/observe/predict/advance/reset probe before scoring episodes.

Use the existing preset and sweep mechanisms for repeat seeds and settings. Create
each hand subset explicitly in its preset so the training membership is reviewable;
keep the evaluation target fixed across the 1–6-hand comparisons.

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


## Human Policy / HAT across verified hands

**Original model / official implementation:** `RogerQi/human-policy` commit
`2d9d73cc5a3859094ef705f35b8f2faecfc2bc4f`, `hat_linear.yaml`, its DINOv2 ViT-S/14
backbone, CVAE Transformer, native image transform (224 × 308), AdamW and
L1 + 10 × KL + 2 × EEF loss are called directly. State/action tensors retain
128 slots; right wrist is 30:39 and palm plus five fingertips is 40:58. The
public README uses a 100-step chunk, batch 64 and 50,000 updates (the public
trainer names this argument `num_epochs`). These are examples, not WUJI2 requirements.

**Skynet implementation:** `human-policy-hat` uses the existing adapter registry,
recording preparation, asset-bound hand geometry, shared RGB files, experiment
compiler, Slurm jobs, progress reader and checkpoint collection. Verified right-hand
measured states and controller targets become metric palm-relative fingertips
and camera-OpenGL wrist poses. Head, left-hand and H1 joint slots are zero.
Three fixed scene views replace the official two egocentric views; this is a
Skynet input change. Human/humanoid co-training and paper reproduction are not claimed.
The existing FAAS codec is only an intermediate source of verified wrist geometry;
HAT receives 128-dimensional wrist/fingertip tensors, not FAAS82 joint slots.

The versioned Skynet fit preset uses 3,000 optimizer updates, batch 16, LR 1e-4,
50-step chunks, seed 42 and validation every 200 updates. Blank control frequency
preserves source timing. Windows follow the recorded pre-action alignment, pad
within each episode and retain fixed episode splits. Statistics use training
frames only. Training keeps the original color jitter and 0.1 state masking;
validation is deterministic and also measures prior-inference wrist/fingertip MAE.
The policy runtime reuses the existing ACT Python environment through a separate
HAT source profile. DINOv2 code is pinned to
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`; only the visual backbone is pretrained.
Best/latest weights and exact input/settings receipts are retained. Full-state
resume is not implemented. Fit loss and finite prior action chunks do not
establish closed-loop cube-pick success.

**Vision-backbone ablation:** the same pinned repository also ships
`act_resnet.yaml`, which differs from `hat_linear.yaml` only in the visual
backbone: ImageNet ResNet18 fine-tuned at LR 1e-5 with `ACT_linear` features
instead of the frozen DINOv2 ViT-S/14 with CLS-concatenated `linear` features.
Both are official configs; choosing one per run (`model_config`, preset
`hat-fit-resnet/v1`) is a Skynet option. The config name is saved in each new
checkpoint and rollout rebuilds that config. Checkpoints saved before this option
carry no name and are loaded as `hat_linear`, the only config they could have used.

**Skynet rollout implementation:** HAT now composes the existing
`dexverse_recorded` evaluator and immutable checkpoint/target capsules. The
recorded target-hand asset, controller mapping, three RGB cameras and source frequency
are checked again in the worker. Official HAT prior inference is reused without
changing the network or checkpoint. Its metric fingertip outputs are converted
to the target hand's controlled finger joints using the already installed `dex-retargeting==0.4.6`
`PositionOptimizer`; this multi-hand bridge is a Skynet addition, not an upstream HAT
robot controller. It uses metric scale 1, no temporal regularization, the exact
recorded joint limits, and the existing verified camera/wrist action codec.
The saved chunk is executed in full at its saved frequency (50 commands at
60 Hz for the 51-episode fit), without temporal ensembling or interpolation.

The existing Skynet evaluator uses the pinned DexVerse task's success predicate
and requires ten consecutive successful simulation steps. It records per-episode
success/failure, videos and action traces through the existing evaluation system.
A submitted or successfully initialized evaluation does not establish a task
success rate; that rate must come from completed rollout episodes.


**Skynet multi-hand study:** the same HAT architecture accepts the seven verified
right-hand assets. Finger slots derive from the existing anatomical map and URDF
ancestry, including the extra thumb/pinky joints in Shadow and Sharpa. Four-finger
Allegro/LEAP leave the absent pinky slot zero. Retargeting uses only present tips;
Inspire mimic joints use the installed MimicJointKinematicAdaptor, with each chain
composed back to its independent controller joint. No target demonstrations fit
these mappings. The exact runtime URDF and independent FK are checked before use.

The many-dataset binding passes the frozen resolved spec to the shared recording
reader. The existing source-frame selector and hand-balanced sampler are reused.
The seven-hand main recipe fixes 1,800 actually consumed original source frames,
30 Hz/chunk30, batch16, LR1e-4 and 8,000 optimizer updates. Normalization uses only
those selected source frames; validation and target hands do not enter statistics.
The native HAT AdamW optimizer has no added warmup/decay schedule.
Validation is disabled in the main recipe and the final step-00008000 checkpoint
is retained. The historical 3,000-step WUJI2 fit uses its own immutable version,
60Hz/chunk50 and validation-selected checkpoint; it is separate from the study.

Evaluation targets are frozen independently of training selections. Unseen-hand
requests reject any target identity found in any source dataset. The worker binds
its controller and RGB cameras to the target manifest and verifies saved training
input IDs/checksums against the policy's frozen source selections. Checkpoint
normalization is never recomputed on the target. Static multi-hand compatibility
is not evidence of zero-shot task success; only the planned rollout ledger can
establish that result. PH2D is the upstream human dataset, and HAT is the policy
model; this baseline adds neither PH2D demonstrations nor a pretrained HAT policy.
