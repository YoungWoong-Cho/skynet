# Training images and policy data exports

In **Data → Collect**, enable **Capture training images** before starting a new session. The web form enables it by default; API clients retain the state-only default unless they send `image_capture: true`. An active session's setting cannot change. Existing immutable session capsules and the collection adapter registry are unaffected.

Each successful episode retains its original DexVerse trajectory pickle (actions T, scene states T+1), a frozen camera recipe, and original frame times. No additional camera sensors are created in the headset process: Isaac Lab XR explicitly does not support them. When collection ends, Skynet closes XR and starts a separate headless renderer under the same GPU lock. Wait for **Preparing training images** to finish before exporting or starting another session.

The renderer restores each saved pre-action scene state without advancing physics and writes three 256 × 256 RGB scene views to an HDF5 sidecar, alongside exact joint states, original commands, control and wall timestamps, calibration, joint names, action scale/offset, and robot revision. These are images reconstructed from saved simulator states after collection, not pixels captured during the headset session. The native recordings and their original receipts remain unchanged. Separate image receipts and HDF5 metadata bind each sidecar to its original recording checksum and frozen recipe. Scene assets must have fixed layouts; randomized multi-asset scenes are rejected.

Image preparation increases post-session time, transfer time, and disk use. Failures preserve original recordings and prevent incomplete visual exports. Each episode is limited to 6,000 control steps and a 4 GB sidecar. Image preparation has a 30-minute limit. The CPU exporter accepts at most 1,000 episodes and 20 GB of source images per preparation, subject to available disk space.

## Dataset preparation and management

Use **Data → Recording → Prepare for training** on a session, or the same action in **Review recordings**. Choose the target policy and review the episode split. Every recording in that session is included automatically, and the verified dataset is always copied to the training cluster. At least two episodes and a non-zero validation split are required for DP. **Registry** manages saved datasets and their versions; it has no preparation action. Existing partial revisions remain immutable and never change the recordings included in a new preparation. Registry registration, editing, version publication, import details, derivations, bundle creation and bundle previews use the shared application modal.

| Policy / recipe | Prepared payload | Training in Skynet |
| --- | --- | --- |
| Diffusion Policy | Zarr with RGB cameras NCHW at 320 × 240, state/action vectors and episode boundaries | Supported through the XPolicyLab DP adapter |
| ACT | Episode HDF5 with actions, qpos, cameras at 640 × 480, and task configuration | Export only; no ACT trainer installed |
| XPolicyLab demonstrations | HDF5 per episode with state, action and vision groups | Intermediate export; further policy requirements apply |

Other registered adapters remain visible with their specific missing requirements. A file extension alone does not establish compatibility. In particular, OpenPI's current LIBERO bridge and GR00T's GR1 bridge cannot consume these Shadow recordings without embodiment and observation mapping.

One logical dataset groups immutable original revisions and prepared formats. An original revision freezes source receipts, selected episodes, split and seed. **Prepare another format** restores that revision; changing the selection or split creates a new revision. Formats share a checksum-keyed source cache, so preparing another format does not download the original files again. Each prepared version records the exact converter, dependency lock, source revision, joint/camera mapping, validation results and checksums. Normalization is fitted using training episodes only. Validation episodes are kept separate by the actual DP loader.

Preparation runs in one durable queue. Progress survives reloads; interrupted work can be retried. Output is published only after validation. Transfers reuse verified local output, verify every destination file, and atomically publish a content-addressed cluster directory. DP additionally runs the real pinned policy data loader before enabling **Use in experiment**. Failed transfers preserve the downloadable local copy. Content versions stay immutable; verified local and cluster locations are managed separately.

**View dataset** shows each format, recording revision, split, copies, downloads, failures and experiments using it. **Use in experiment** selects the compatible adapter, verified dataset bundle, pinned repository revision and configured DP runtime. The normal experiment preview and submission flow follows. The run verifies every dataset file again before training. Training checkpoints and held-out loss are produced by the upstream DP model and loop; autonomous DexVerse rollout evaluation is not integrated.

A local prepared copy can be removed only after the cluster copy is reverified and no experiment pins the version. Original recordings are never deleted by this action. Archiving a dataset preserves its files and existing run references. Older state-only conversions remain readable through their historical registry/API entries; the duplicate conversion UI has been retired.

## DP runtime

The `skynet-dp` profile in `config/clusters/skynet.json` pins Python 3.11, Torch 2.7.0/cu128 and the XPolicyLab source revision. Its isolated environment adds Hydra 1.3.2, Diffusers 0.11.1, Hugging Face Hub 0.25.2, Transformers 4.26.1, Tokenizers 0.13.3, Zarr 2.18.7, Numcodecs 0.15.1, NumPy 1.26.4, Numba 0.61.2, Dill 0.3.8, Einops 0.8.1, Pandas 2.2.3 and tqdm 4.67.1. The existing simulator environment is unchanged. The profile's package and source checks run before training; the adapter capsule freezes Skynet's dataset and checkpoint integration for reproducibility.

## Exact mappings

The three distinct fixed scene views map to XPolicyLab camera slots as follows:

| Dataset slot | Rendered sensor | Mount |
| --- | --- | --- |
| `cam_head` | `third_person_camera` | Fixed scene front |
| `cam_left_wrist` | `third_person_camera_left` | Fixed scene left |
| `cam_right_wrist` | `third_person_camera_right` | Fixed scene right |

The camera slot names are required by the supplied policy data loaders. These are **not wrist-mounted cameras**. Matching views and calibration must be configured during evaluation; this is not a claim of pretrained camera compatibility.

A floating hand's six virtual wrist joints occupy the `arm_dim` portion; commanded finger joints occupy `ee_dim`. Bimanual data is reordered to left wrist, left fingers, right wrist, right fingers for XPolicyLab. The manifest stores the inverse-use mapping back to the simulator's command order. Wrist translation uses metres; rotation and fingers use radians. Actions remain raw joint-position commands, including the recorded scale/offset; they are not silently converted into physical-arm poses or gripper scalars.

The exporter verifies complete episodes, source and sidecar checksums, exact pre-action state and action equality, frame counts, simulation timestamps, RGB layout, robot/task identity, camera calibration consistency, and lossless joint reordering before publishing anything.

## Converter provenance and extension

Vendored files in `ops/datasets/xpolicylab/` are unchanged from XPolicyLab commit `9c98a3aaf02d05c6f9999a5a0a7a42090555ddf3`; their SHA-256 checksums and Apache license are retained. The exporter uses its `pack_robot_state` and `decode_image_bit` helpers. Skynet streams the DP/ACT conversion operations to bound memory use instead of accumulating a whole image dataset in RAM. Regression tests execute the unmodified upstream converters and compare every training array against the streamed result.

To support another format, add an explicit recipe in `skynet_app/dataset_formats.py`, declare the matching adapter data contract and a converter with its observation/robot requirements and tests. A free-text format label does not create conversion support. Other XPolicyLab converters may consume the shared HDF5 export if their required modalities and robot semantics match; depth, point clouds, language annotations, or special action layouts cannot be invented from RGB alone.

The post-session renderer requires `h5py` in its Isaac environment. The app installs HDF5, OpenCV, Zarr 2, and PyYAML from the locked dependencies. `ops/xr/check_images.py` can verify six synthetic frames on an idle simulator under its session lock; test files must stay outside real session directories and must never be registered as demonstrations.
