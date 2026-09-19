# Adapter-selected recording datasets

**Data → Collect** retains original DexVerse PKL trajectories: T commands, T+1 scene states and the recording's available timing, controller, hand and scene metadata. Ending a session does not render observations or video. **View** displays recorded values and frozen 3D assets without starting a simulator. Raw recordings remain immutable.

## Convert

Select an **Adapter**, its declared **Data preset**, and a whole-episode train/validation split. The request includes both the stable adapter ID and its exact immutable version ID. An adapter edited while the modal is open must be selected again; the backend never silently substitutes another version.

Each adapter version declares conversion under `train.data_requirements.recording_conversion`. This is the single source for required observations, action semantics, preprocessing, supported embodiments, source/runtime identity and the CPU loader validation command. There is no independent recipe catalog or requested output-format selector.

| Adapter | Data preset | Shared observations | Training preprocessing |
| --- | --- | --- | --- |
| ACT / ACT Native | Three scene views and joints | RGB front/left/right | 640 × 480, CHW |
| EgoVerse ACT / HPT | Recorded joints | RGB front/left/right | Native recorded-joint reader |
| UniDex | Scene-front point cloud and FAAS | 1,024 front-camera XYZRGB points, absolute FAAS82 streams | Camera basis conversion and full-chunk anchoring in memory |

The scene cameras are fixed `third_person_camera`, `third_person_camera_left` and `third_person_camera_right`, named `scene_front`, `scene_left` and `scene_right` in shared observations. They are not head-mounted or wrist-mounted cameras. UniDex uses only the fixed front scene camera. Point colors are floats in [0,1]. Shared point coordinates use ROS optical axes; the UniDex reader changes the Y/Z signs to obtain its OpenGL camera basis. FAAS wrist poses use the declared OpenGL camera frame.

UniDex conversion requires separate validation episodes and verified exact-asset mappings. Shadow, Inspire RH56, Allegro V4 and LEAP V1 are currently verified. Sharpa and the two WUJI assets are blocked before GPU work. Conversion does not claim cross-hand evaluation readiness; simulator evaluation is not registered for this adapter.

## Storage and work

All conversions produce `skynet.recording-dataset/v1`: a checksummed `manifest.json` referencing shared immutable arrays. The builder writes native state/command streams once per recording, references verified archived capture arrays in place when suitable, and references reusable observation artifacts. It does not materialize separate ACT HDF5 or EgoVerse Zarr payload copies, resize images on disk, or generate a ZIP.

Opening Convert loads metadata only. The modal lists requirements and states that reuse/missing counts will be checked when conversion starts. Final submission performs source validation, then prepares only missing declared observations. GPU work is confined to observation generation. CPU preparation fixes the split and verifies source hashes, frame/timestamp alignment, command semantics and stream references. The selected adapter's frozen reader must validate the result before publication.

`Use in experiment` selects the exact adapter version, declared data settings, verified dataset version, pinned source and matching runtime. Readers verify referenced files, perform adapter-specific transforms in memory, and fit normalization using training episodes only. Original ACT model/loss behavior is retained while the shared reader uses the registered split and recorded command timing.

Dataset details expose the manifest and preparation log. Archiving hides a registry entry while retaining its files and references. Deletion and replacement must respect shared consumers and immutable experiment history. Old converted versions require explicit verified migration/retirement; this pipeline does not offer a local-copy or legacy-format compatibility path. Original PKLs must not be deleted as part of converted-data cleanup.

## Adding an adapter

Declare a recording data preset in its immutable adapter manifest, freeze its shared-manifest reader and conversion validation command in the capsule, and add tests covering observation/action semantics and loader behavior. A format label alone does not establish compatibility. External native datasets retain their adapter-specific contracts. Unsupported observations or hand mappings must fail explicitly before expensive work.
