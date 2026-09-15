# Recorded observations and policy data exports

**Data → Collect** saves original DexVerse trajectories: T actions, T+1 scene states, control timestamps, named joint/controller metadata, frozen hand identity and preview geometry. Ending a session does not run an image or video renderer. The verified originals are archived on sky2; see [collection storage](collection-storage.md).

**View** reads recorded values and shows the saved 3D scene, keypoints and the recording's frozen hand meshes. It does not start a simulator, encode video, or prepare training camera observations. Old HDF5 sidecars are readable as a bounded metadata fallback. Historical files that never saved joint order, scene shapes or a hand bundle show the missing-data explanation; the viewer does not guess them from today's model catalog.

**Convert** prepares the selected policy's declared camera observations, reusing matching immutable artifacts. RGB is generated only when required. Point-cloud requirements retain metric depth and calibration, so another crop, point count or sampling configuration can reuse depth without rendering the scene again. See [observation preparation](observation-preparation.md) for contracts, identity, recovery and cleanup.

## Dataset preparation and management

Use **Convert** in **Data → Recordings** or the dataset preparation dialog. Choose the target policy and episode split. Preparation waits for the verified sky2 archive and reads the original files directly on the cluster. Missing images do not make a recording ineligible: the observation phase prepares them. Single-episode preparation is supported by the Skynet-controlled recipes; ACT Native retains its original minimum of two episodes and 80/20 split. Dataset registration, editing and preparation use the shared application modal.

| Policy / recipe | Prepared payload | Training in Skynet |
| --- | --- | --- |
| Diffusion Policy | Zarr with RGB cameras NCHW at 320 × 240, state/action vectors and episode boundaries | Supported through the XPolicyLab DP adapter |
| Diffusion Policy, state only | Zarr with joint observations, actions and episode boundaries | Supported through the XPolicyLab DP adapter |
| ACT | Episode HDF5 with actions, qpos, cameras at 640 × 480, and task configuration | Supported through the XPolicyLab ACT adapter |
| EgoVerse recorded joints | Native episode Zarr with recorded joints, commands and three RGB scene views | Supported by the native ACT and HPT recorded-joints presets |
| XPolicyLab demonstrations | HDF5 per episode with state, action and vision groups | Intermediate export; further policy requirements apply |

Other registered adapters remain visible with their specific missing requirements. A file extension alone does not establish compatibility. In particular, OpenPI's current LIBERO bridge and GR00T's GR1 bridge cannot consume these Shadow recordings without embodiment and observation mapping.

One logical dataset groups immutable original revisions and prepared formats. An original revision freezes source receipts, selected episodes, split and seed. **Prepare another format** restores that revision; changing the selection or split creates a new revision. Formats reuse the verified sky2 originals without copying them to the Mac. Each prepared version records the exact converter, dependency lock, source revision, joint/camera mapping, validation results and checksums. Normalization is fitted using training episodes only. Validation episodes remain separate in the corresponding training loader.

Observation rendering runs as a durable Slurm producer on a configured Isaac-compatible GPU. Point-cloud derivation runs on CPUs from verified depth/RGB artifacts. A producer owns its artifacts independently of the Convert that first requested them; concurrent conversions share it and app restart recovers the same submission token. Render requests split at 32 source episodes and a 3.5 MB payload budget. Point-cloud derivation uses one source episode per CPU producer, so expensive multi-camera sampling can progress and retry independently. Original episodes remain limited to 6,000 control steps.

Once observations are ready, format conversion runs on four CPUs, with 32 GB memory and a one-hour limit in the configured normal queue. Frozen code, source receipts and result checksums identify each attempt. The selected native training loader verifies the final dataset before publication. Existing v1 output formats remain self-contained: each adapter's required array layout is materialized in its own prepared dataset. Shared observation generation avoids repeated simulator work; it does not eliminate these format-specific output copies. Failures preserve originals and expose a retry action.

**View dataset** shows each format, recording revision, split, location, downloads, failures and experiments using it. **Download** and **Manifest** explicitly stream the selected sky2 artifact to the browser; Skynet does not keep a Mac payload cache. **Use in experiment** selects the compatible adapter, verified dataset bundle, pinned repository revision and matching runtime. The normal experiment preview and submission flow follows. The run verifies every dataset file again before training. Training checkpoints and held-out loss use the selected policy implementation. DexVerse rollout evaluation requires compatible task, hand/action layout and camera contracts; a prepared file alone does not establish cross-hand zero-shot compatibility.

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

The on-demand observation renderer requires `h5py` in its Isaac environment. XPolicyLab conversion uses the configured cluster policy environment. EgoVerse conversion uses its frozen dependency list, including Zarr 3 and SimpleJPEG, in a cluster-side `uv` environment. The operator app retains only control metadata and converter code for new preparations. `ops/xr/check_images.py` can verify six synthetic frames on an idle simulator under its session lock; test files must stay outside real session directories and must never be registered as demonstrations.
