# Teleoperation recordings → training dataset

1. Open **Data → Collection → Teleoperate**, choose the hand and task, and start a session.
2. In the headset, collect episodes and choose **End collection** when finished.
3. Open **Recordings → Review recordings** and choose an episode to review.
4. Choose **Convert for training**, name the dataset, optionally select a subset, then choose **Convert dataset**.
5. Once converted, download the HDF5 file or open its registered dataset. No terminal command is required.

Conversion uses the collection's original workstation, hand, task, source revision and recorded controls. It does not start training or evaluation. The old local hand/head tracking tools and offline training experiments are under **Setup**; JSONL motion files are not simulated task demonstrations.

## Output and compatibility

The format is `dexverse-demo-hdf5/v1`, using the pinned DexVerse `PerPickleH5Writer` schema. Each `data/demo_n` contains `actions`, `source_actions`, `obs/<group>/<term>`, `next_obs`, `initial_obs`, `final_obs`, and simulation timestamps. Metadata records the action joint order, observation shapes, task, robot, source recording indices and checksums. Actions are preserved as float32 in their original order. T actions correspond to T+1 saved states.

State observations are reconstructed from the saved simulator states without advancing physics. Contact impulses and images are excluded; scene snapshots do not contain recorded contact impulses. Tasks that need a goal require the saved goal. Missing state, checksum mismatches, incompatible hand/task data, or randomized multi-asset scenes fail explicitly.

Use a trainer supporting this DexVerse state HDF5 schema. The existing GR00T and OpenPI adapters require other formats; the old offline JSONL experiment uses a different Skynet state/action schema. This feature does not add a native HDF5 training adapter. A workstation dataset also needs transfer before a cluster job can consume it. Both frontend compatibility checks and backend submission reject an untransferred workstation path.

The original recordings remain unchanged. Successful conversion creates a dataset resource/version, a training-data bundle, and a derivation linking the output to a manifest of the original recordings. The HDF5 and manifest are retained on the workstation and cached on the Mac for download.

## Recovery and limits

- The app saves the request and converter source before starting the worker. Repeated identical requests recover the same conversion. Failed conversions can be retried; another selected subset creates a separate versioned result.
- A detached user service continues after the browser closes or the web app restarts. A local monitor resumes status checks and downloads. Lost SSH acknowledgements never trigger duplicate GPU work.
- Conversion waits for the existing GPU session lock for up to 30 minutes, then has a 60-minute runtime limit. It does not stop another collection. Each recording is limited to 100 MB and each output artifact to 1 GB; larger inputs fail explicitly.
- Progress, connection failures, worker errors and logs are accessible from the conversion dialog. Downloads are size/checksum verified before registry publication. Partial downloads are retried and cannot be marked ready.

## Verification

See `validation/collection-conversion.json` for browser coverage and limitations. Local tests cover data integrity, retries, deduplication, partial downloads, registration/lineage, form behavior and incompatible training inputs. `tests/collection_browser_fixture.py` starts a marked synthetic QA site on port 8091 with a disposable database and SSH disabled. It exercises the real API, download and registry code while substituting only the remote worker transport.

Actual GPU conversion of the 51 Shadow cube episodes is pending. Workstation access was deferred at the user's request; synthetic browser conversion is not evidence of Isaac Sim execution.
