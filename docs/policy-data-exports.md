# Adapter-selected recording datasets

**Data → Collect** retains original DexVerse PKL trajectories: T commands, T+1 scene states and the recording's available timing, controller, hand and scene metadata. Ending a session does not render observations or video. **View** displays recorded values and frozen 3D assets without starting a simulator. Raw recordings remain immutable.

## Convert

Choose a name for the converted dataset, an **Adapter** and a whole-episode train/validation split. Its declared data configuration is applied automatically; the modal highlights the input modalities it uses. The name belongs to this conversion, so different conversions of the same recording can have different names. The request includes both the stable adapter ID and its exact immutable version ID. An adapter edited while the modal is open must be selected again; the backend never silently substitutes another version.

**Data → Datasets** lists each prepared dataset directly. Open **View** for that dataset's provenance, conversion settings, manifest, preparation log and experiment references. There is no separate parent dataset to create or select before conversion. Internal source groups retain recording lineage; experiments select the exact published dataset version. Repeating the same conversion settings reuses its existing preparation; changing only the submitted name does not duplicate the data.

Each adapter version declares conversion under `train.data_requirements.recording_conversion`. This is the single source for required observations, action semantics, preprocessing, supported embodiments, source/runtime identity and the CPU loader validation command. There is no independent recipe catalog or requested output-format selector.

| Adapter | Data preset | Shared observations | Training preprocessing |
| --- | --- | --- | --- |
| ACT / ACT Native | Three scene views and joints | RGB front/left/right | 640 × 480, CHW |
| EgoVerse ACT / HPT | Recorded joints | RGB front/left/right | Native recorded-joint reader |

The scene cameras are fixed `third_person_camera`, `third_person_camera_left` and `third_person_camera_right`, named `scene_front`, `scene_left` and `scene_right` in shared observations. They are not head-mounted or wrist-mounted cameras. Point colors are floats in [0,1]. Shared point coordinates use ROS optical axes. FAAS wrist poses use the declared OpenGL camera frame.

FAAS-based conversion, used by HAT, requires verified exact-asset mappings. Shadow, Inspire RH56, Allegro V4, LEAP V1, WUJI2, WUJI1 and Sharpa are supported. The [WUJI1](wuji1-faas.md) and [Sharpa](sharpa-faas.md) mappings are Skynet extensions for their exact registered assets, retaining the original UniDex FAAS82 layout. Verification covers the mapping and reversible coordinate contract; it does not establish pretrained model performance or cross-hand generalization.

## Storage and work

All conversions produce `skynet.recording-dataset/v1`: a checksummed `manifest.json` referencing shared immutable arrays. Each dataset row identifies its own manifest and conversion settings; multiple rows can reference the same recording streams and observations without copying them. The builder writes native state/command streams once per recording, references verified archived capture arrays in place when suitable, and references reusable observation artifacts. It does not materialize separate ACT HDF5 or EgoVerse Zarr payload copies, resize images on disk, or generate a ZIP.

Opening Convert loads metadata only and displays the selected adapter's input modalities. Final submission performs source validation, then prepares only missing declared observations. GPU work is confined to observation generation. CPU preparation fixes the split and verifies source hashes, frame/timestamp alignment, command semantics and stream references. The selected adapter's frozen reader must validate the result before publication.

`Use in experiment` selects the exact adapter version, declared data settings, verified dataset version, pinned source and matching runtime. Readers verify referenced files, perform adapter-specific transforms in memory, and fit normalization using training episodes only. Original ACT model/loss behavior is retained while the shared reader uses the registered split and recorded command timing.

Dataset names and archive state apply to the selected dataset version. Archiving hides that row while retaining its files and experiment references; other datasets from the same recording remain available, and new conversions are allowed. Deletion and replacement must respect shared consumers and immutable experiment history. Removing an externally registered dataset retains its external source files. Old converted versions require explicit verified migration/retirement; this pipeline does not offer a local-copy or legacy-format compatibility path. Original PKLs must not be deleted as part of converted-data cleanup.

## Conversion time versus experiment time

This separation is **Skynet infrastructure**, not an original model requirement. It applies to every adapter using the shared recording dataset; externally imported native datasets keep their own reader contracts.

| Stage | Responsibility |
| --- | --- |
| Conversion | Generate only the selected adapter's required modalities, at every aligned source timestep. Retain source timing and all source-rate state, command and observation arrays. Reuse existing shared artifacts. |
| Experiment | Choose training frequency (`control_hz`) and action chunk (`action_steps`). A blank frequency uses the source frequency. |
| Preview / submission | Resolve the same sampling plan as the worker; report eligible windows and excluded episodes. Reject settings with no usable training windows, or no usable validation windows when the adapter requires validation. |
| Training / evaluation | Read the chosen aligned timesteps in memory. Pin the resolved frequency, chunk and per-episode stride in the run/checkpoint; evaluation follows that frequency. Changed settings require a new run rather than resuming the old temporal configuration. |

For a 60 Hz source, 30 Hz selects frames 0, 2, 4, …; 15 Hz selects frames 0, 4, 8, …. State, RGB, point cloud, timestamps and commands use the same selection before the adapter applies its declared action alignment. Only exact integer downsampling is supported: no interpolation or upsampling. Mixed source rates need an explicit common frequency that divides every source rate. Changing frequency or chunk does **not** rewrite datasets, rerender observations or duplicate artifacts.

Conversion checks source integrity, temporal alignment, action semantics and modality validity, not whether an episode is long enough for a particular training chunk. A valid short recording can therefore be converted and retained. Its usability depends on the experiment settings and adapter window rule. Eligible-window counts describe possible training starts, not the number of optimizer steps or samples drawn per epoch.

### Original models versus Skynet readers

Frequency selection above is a **Skynet experiment setting**. It does not silently change model losses, action-label meaning or tail handling:

| Adapter | Original-model setting / behavior | Skynet recording-reader behavior retained |
| --- | --- | --- |
| XPolicyLab ACT / ACT Native | Pinned recipe chunk default: 50; padded actions are masked by the ACT loss. | Shared-reader ACT aligns commands to the recorded observation timestep; ACT Native retains its explicit `max(0, observation step - 1)` action start. Both apply that rule on the sampled time axis. |
| EgoVerse ACT / HPT recorded joints | Pinned ACT `chunk_size`: 100; HPT EVA flow-head `action_horizon` and denoiser `act_seq`: 100. | Recorded-joint reader starts commands at the selected observation timestep and repeats the final command for short tails. One experiment chunk updates all corresponding model output dimensions. HPT's separate trunk token horizon is unchanged. |

Chunk defaults are model/recipe settings, not FAAS constraints. They are defined by the adapter's pinned source and experiment declaration. Conversion manifests carry no training-frequency or action-chunk requirement. Duplicate output-horizon settings in EgoVerse `model_overrides` are rejected in favor of the experiment's Action chunk field.

## Adding an adapter

Declare a recording data preset in its immutable adapter manifest, freeze its shared-manifest reader and conversion validation command in the capsule, and add tests covering observation/action semantics and loader behavior. A format label alone does not establish compatibility. External native datasets retain their adapter-specific contracts. Unsupported observations or hand mappings must fail explicitly before expensive work.
