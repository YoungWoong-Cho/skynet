# EgoVerse model adapters

The adapters use EgoVerse revision `e17cf98fe4bc234c564b37abc9e155f25e76d566`.
`egoverse_manifest.py` defines the model inventory once. All entries share the
same native trainer bridge, dataset checks, checkpoint receipt and evaluator.
Model implementations remain in EgoVerse's `egomimic/algo/` and `egomimic/models/`.

## From recorded demonstrations

1. Data → Recording → the session's **Prepare for training**.
2. Choose **EgoVerse · RGB and joints** and prepare the dataset.
3. Once ready, choose **Use in experiment**. This binds the verified cluster copy.
4. Select **EgoVerse · ACT**, **HPT flow · recorded joints**, or
   **Diffusion Policy · recorded joints**. Keep the pinned repository revision
   and **EgoVerse native** runtime.
5. Set the training values and GPU allocation, preview, and submit.
6. From the completed Training Run, choose **Start evaluation**, select
   **EgoVerse held-out predictions**, choose episodes/seeds/resources and submit.
7. Open the evaluation row, then an episode's **Detail** for metrics, video and logs.

The existing dataset registry owns immutable versions, verified locations,
training references and deletion protection. Conversion preserves the original
recordings, frame alignment, joint order and calibrated scene camera identity.
It writes separate native Zarr stores for training and validation episodes.
Native normalization is computed from training episodes only.

## Model coverage

The registry exposes 22 entries: 20 complete upstream model configurations plus
recorded-joint HPT and diffusion configurations. The incomplete `pi0.5_base`
configuration and the `egobridge` alias are not duplicate adapter entries.

| Group | Input |
| --- | --- |
| ACT; HPT flow recorded joints; diffusion recorded joints | Converted teleoperation RGB, measured joints and commanded joints |
| HPT flow Aria, EVA, human, Mecka, Scale, human keypoints | The matching native embodiment schema |
| HPT co-training encoder-decoder, separate/shared heads, Mecka, Scale | Matching datasets for every configured domain |
| HPT Qwen per-token and pooled | Native robot data with language annotations; Qwen model assets |
| π0.5 Aria, EVA, Mecka, Scale and the two co-training variants | Native domain data, language annotations and pretrained π0.5 weights |

Recorded robot joints cannot provide missing human poses, Cartesian actions,
language labels or additional co-training domains. These adapters reject the
recorded-joint contract instead of relabeling its values as another embodiment.

The diffusion adapter uses EgoVerse's native `DiffusionPolicy` head with its HPT
backbone. The upstream repository does not contain a model YAML selecting that
head. Its wiring is an explicitly labeled Skynet configuration, not an upstream
benchmark preset. It is independent of the XPolicyLab DP implementation.

## Native dataset imports

Use the registry's existing resource/version/bundle workflow with format
`egoverse-episodes-zarr/v1`, role `training_data`, and contract
`egoverse.native-<upstream_model_name>/v1`.

The registered directory must contain:

- `manifest.json`: `episodes` (relative `path` and `steps`), disjoint `split.train`
  and `split.validation` indices, `capture.step_dt`, and SHA-256/size receipts for
  every payload file under `files`.
- Native episode Zarr directories with the selected embodiment's keys and metadata.
- `data.yaml`: a resolved native `MultiDataModuleWrapper` configuration. Use
  `LocalEpisodeResolver` with folders inside this directory, covering exactly the
  manifest's corresponding train/validation episodes. Supply the appropriate
  native key maps, horizons, transformations and all model domains.
- `evaluator.yaml`: the resolved matching native `egomimic.eval.*` configuration,
  including visualization and inverse-coordinate transforms. Its frames must match
  the dataset. Both YAML files must be included in the manifest's file receipts.

Register the manifest SHA-256, the same metadata and a verified cluster location.
Training and evaluation verify the immutable files; the checkpoint embeds the
resolved native configuration and dataset identity. Evaluation reuses checkpoint
normalization and never recomputes it from held-out data.

## Execution and defaults

The bridge invokes native `trainHydra.train`, `ModelWrapper`, optimizers, schedulers
and Lightning checkpointing. Source defaults remain visible: 2,000 epochs,
100 training batches per epoch, 80 validation batches, validation every 200 epochs,
and batch size 32 per GPU. Learning rates follow each model YAML.
There is no added early-stopping rule. Precision defaults to native BF16; choose
FP32 explicitly for older GPUs such as the cluster's RTX 6000.

Single-node multi-GPU execution uses Lightning DDP. HPT and π0.5 permit unused
parameters for optional model heads. Cluster transport settings are inherited
from the existing shared training environment. Slurm launch variables are removed
inside the native trainer so its workers are not mistaken for independent Slurm
allocations. The native final checkpoint is saved even for a short test run.

The native training outlier filter is enabled by default and exposed explicitly.
It can reject all samples in small collections. Disable **Filter training outliers**
when intentionally testing such data. Held-out evaluation keeps episode boundaries
and checks finite model inputs without applying the training quantile filter.

π0.5 uses the separate **EgoVerse π0.5** runtime. Its OpenPI submodule is pinned to
`981483dca0fd9acba698fea00aa6e52d56a66c58`; the Transformers patch specified by
upstream `pi05.md` is confined to that environment. Its pretrained weight directory
is a required experiment input. Runtime probes verify imports, the patch, CUDA
forward/backward execution, and video encoding/decoding on an allocated GPU.

## Evaluation meaning

This integration performs **offline held-out prediction evaluation**. Recorded-joint
models report raw joint MSE over real future frames, excluding end padding. Native
embodiment models use their supplied native evaluator and visualization.
Videos show recorded observations/predictions. They are not autonomous simulator
rollouts and do not report a fabricated task success rate. An episode's prediction
metric is shown in the existing result table and Detail modal.

## Validation

Browser verification used the real Shadow right-hand collection `6a257afb`:
51 episodes converted, with 41 training and 10 held-out episodes. Bounded native
training tests completed for ACT (Slurm 3802608, four A40s), HPT flow (3802923,
four A40s), and diffusion (3802925, one A40). These one-epoch tests validate
execution and checkpoint production, not policy quality or convergence.

Held-out ACT (3802739), HPT (3802937), and diffusion (3802940) evaluation
completed through the browser; their episode
modals displayed the numeric prediction error, playable video, and Slurm logs.
The separate native and π0.5 runtimes passed GPU readiness probes. Actual π0.5
training still requires pretrained weights, and the other native embodiment
models require their matching datasets before end-to-end validation.
