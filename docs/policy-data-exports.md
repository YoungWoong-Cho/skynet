# Training images and policy data exports

In **Data → Collect**, enable **Capture training images** before starting a new session. The web form enables it by default; API clients retain the state-only default unless they send `image_capture: true`. An active session's setting cannot change. Existing immutable session capsules and the collection adapter registry are unaffected.

Each successful episode retains its original DexVerse trajectory pickle (actions T, scene states T+1), a frozen camera recipe, and original frame times. No additional camera sensors are created in the headset process: Isaac Lab XR explicitly does not support them. When collection ends, Skynet closes XR and starts a separate headless renderer under the same GPU lock. Wait for **Preparing training images** to finish before exporting or starting another session.

The renderer restores each saved pre-action scene state without advancing physics and writes three 256 × 256 RGB scene views to an HDF5 sidecar, alongside exact joint states, original commands, control and wall timestamps, calibration, joint names, action scale/offset, and robot revision. These are images reconstructed from saved simulator states after collection, not pixels captured during the headset session. The native recordings and their original receipts remain unchanged. Separate image receipts and HDF5 metadata bind each sidecar to its original recording checksum and frozen recipe. Scene assets must have fixed layouts; randomized multi-asset scenes are rejected.

Image preparation increases post-session time, transfer time, and disk use. Failures preserve original recordings and prevent incomplete visual exports. Each episode is limited to 6,000 control steps and a 4 GB sidecar. Image preparation has a 30-minute limit. After the session and its image renderer finish, Skynet archives the completed session on sky2 and verifies the archive before removing the redundant workstation copy. See [collection storage](collection-storage.md) for the verification and cleanup policy.

## Dataset preparation and management

Use **Data → Recording → Prepare for training** on a session, or the same action in **Review recordings**. Choose the target policy and review the episode split. Every recording in that session is included automatically. Preparation waits for the verified sky2 archive, then reads its original recordings and image sidecars directly on the cluster. There is no local preparation destination. At least two episodes and a non-zero validation split are required for trainable formats. **Registry** manages saved datasets and their versions; it has no preparation action. Existing partial revisions remain immutable and never change the recordings included in a new preparation. Registry registration, editing, version publication, import details, derivations, bundle creation and bundle previews use the shared application modal.

| Policy / recipe | Prepared payload | Training in Skynet |
| --- | --- | --- |
| Diffusion Policy | Zarr with RGB cameras NCHW at 320 × 240, state/action vectors and episode boundaries | Supported through the XPolicyLab DP adapter |
| Diffusion Policy, state only | Zarr with joint observations, actions and episode boundaries | Supported through the XPolicyLab DP adapter |
| ACT | Episode HDF5 with actions, qpos, cameras at 640 × 480, and task configuration | Supported through the XPolicyLab ACT adapter |
| EgoVerse recorded joints | Native episode Zarr with recorded joints, commands and three RGB scene views | Supported by the native ACT and HPT recorded-joints presets |
| XPolicyLab demonstrations | HDF5 per episode with state, action and vision groups | Intermediate export; further policy requirements apply |

Other registered adapters remain visible with their specific missing requirements. A file extension alone does not establish compatibility. In particular, OpenPI's current LIBERO bridge and GR00T's GR1 bridge cannot consume these Shadow recordings without embodiment and observation mapping.

One logical dataset groups immutable original revisions and prepared formats. An original revision freezes source receipts, selected episodes, split and seed. **Prepare another format** restores that revision; changing the selection or split creates a new revision. Formats reuse the verified sky2 originals without copying them to the Mac. Each prepared version records the exact converter, dependency lock, source revision, joint/camera mapping, validation results and checksums. Normalization is fitted using training episodes only. Validation episodes remain separate in the corresponding training loader.

Preparation runs as a resumable Slurm job with four CPUs, no GPU allocation, 32 GB of memory and a one-hour limit in the configured normal queue. It accepts at most 1,000 episodes and 20 GB of source images. Frozen converter code, requests and receipts identify each attempt; a lost submission reply is recovered with the same submission identity. A background monitor continues after the browser closes and resumes after an app restart. Conversion, ZIP creation, file verification and the pinned training-loader check run on cluster storage. Successful output is published to an immutable directory named by its manifest checksum and registered with a verified cluster location. Failures preserve the archived originals and expose the preparation log and retry action.

**View dataset** shows each format, recording revision, split, location, downloads, failures and experiments using it. **Download** and **Manifest** explicitly stream the selected sky2 artifact to the browser; Skynet does not keep a Mac payload cache. **Use in experiment** selects the compatible adapter, verified dataset bundle, pinned repository revision and matching runtime. The normal experiment preview and submission flow follows. The run verifies every dataset file again before training. Training checkpoints and held-out loss use the selected policy implementation; autonomous DexVerse rollout evaluation is not integrated.

Existing Mac preparation files are migrated before cleanup. When the same prepared dataset is already verified on the cluster, Skynet reverifies it and transfers the existing ZIP unchanged; otherwise a CPU job verifies and packages the transferred files. The local manifest, exact file inventory and ZIP checksum must still match before removal. Unlisted or changed files are preserved for explicit review. Registered versions and valid cluster-pinned training references retain their identities. Failed partial outputs and shared legacy caches follow the separate verified operator-cache migration described in [collection storage](collection-storage.md). Archiving a dataset entry in Registry remains a visibility action that preserves its files and run references. Older registered state-only datasets retain their registry entries and stored files. The retired state-only conversion API and worker have been removed; current preparation uses `/api/data/exports`. See [retired conversion routes](collection-conversion.md).

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

The post-session renderer requires `h5py` in its Isaac environment. XPolicyLab conversion uses the configured cluster policy environment. EgoVerse conversion uses its frozen dependency list, including Zarr 3 and SimpleJPEG, in a cluster-side `uv` environment. The operator app retains only control metadata and converter code for new preparations. `ops/xr/check_images.py` can verify six synthetic frames on an idle simulator under its session lock; test files must stay outside real session directories and must never be registered as demonstrations.
