# Historical state-only collection conversion

The primary UI now uses the unified policy-aware preparation workflow described in [policy-data-exports.md](policy-data-exports.md). The endpoints below remain for existing state-only conversion history and API compatibility. They do not make a dataset compatible with a visual policy.

# Teleoperation recordings → training dataset

This page describes the existing state-only Slurm conversion. For new synchronized RGB recordings and the XPolicyLab, DP, and ACT format choices in Resources, see [Policy data exports](policy-data-exports.md).

1. Open **Data → Collect**, choose the hand and task, and start a session.
2. In the headset, collect episodes and choose **End collection** when finished.
3. Open **Recordings → Review recordings** and choose an episode to review.
4. Choose **Convert for training**, name the dataset, then choose **Convert dataset**. Every recording in that session is included automatically.
5. Once converted, download the HDF5 file or open its registered dataset. No terminal command is required.

Conversion runs on Slurm through the configured cluster gateway, preserving the collection's original hand, task, source revision and recorded controls. The Mac uploads checksum-verified cached recordings; uncached originals must first be copied from the collection host. No conversion runs on that host. It does not start training or evaluation. The former local hand/head tracking and offline experiment sections have been removed from Collection. Their saved files and job records are preserved; JSONL motion files are not simulated task demonstrations.

## Output and compatibility

The format is `dexverse-demo-hdf5/v1`, using the pinned DexVerse `PerPickleH5Writer` schema. Each `data/demo_n` contains `actions`, `source_actions`, `obs/<group>/<term>`, `next_obs`, `initial_obs`, `final_obs`, and simulation timestamps. Metadata records the action joint order, observation shapes, task, robot, source recording indices and checksums. Actions are preserved as float32 in their original order. T actions correspond to T+1 saved states.

State observations are reconstructed from the saved simulator states without advancing physics. Contact impulses and images are excluded; scene snapshots do not contain recorded contact impulses. Tasks that need a goal require the saved goal. Missing state, checksum mismatches, incompatible hand/task data, or randomized multi-asset scenes fail explicitly.

Use a trainer supporting this DexVerse state HDF5 schema. The existing GR00T and OpenPI adapters require other formats; the old offline JSONL experiment uses a different Skynet state/action schema. This feature does not add a native HDF5 training adapter. New outputs are already on cluster storage. Older workstation outputs still require transfer; both frontend compatibility checks and backend submission reject an untransferred workstation path.

The original recordings remain unchanged. Successful conversion creates a dataset resource/version, a training-data bundle, and a derivation linking the output to a manifest of the original recordings. The HDF5 and manifest are retained on shared cluster storage and cached on the Mac for download. The conversion dialog and registry both display the exact path:

`/coc/flash7/ycho420/datasets/derivatives/dexverse-live/<session-id>/<conversion-id>/dataset.hdf5`

Job scripts, input staging and logs are separate, under `/coc/flash7/ycho420/jobs/runs/<conversion-id>/`.

## Existing Slurm implementation

The launcher reuses `ClusterClient.submit_script`, its durable submission receipts/token recovery, `job_statuses`, `write_capsule_file`, and the capture pipeline's verified upload. The Isaac Slurm template was extracted from `ProcessingService.compile` into `capture_processing/slurm.py` and is shared by both launchers. There is no alternate submission protocol.

`config/live_conversion.json` selects an existing profile in `config/capture_pipelines.json`. It currently uses sky2, account/partition `overcap`, and the configured `rtx_6000` GPU alias (verified against the real Slurm inventory), with 4 CPUs, 48 GB RAM and a 30-minute limit. Each job has a private temporary directory. The pinned converter is DexVerse revision `30cc673e27684b9f10186fa6bea731aed246bc9f`, using its documented `--set-state --obs-groups state` observation/HDF5 APIs.

## Recovery and limits

- The app saves the request and converter source before starting the worker. Repeated identical requests recover the same conversion. Failed conversions can be retried. New conversion requests always include every recording; older browser requests containing recording selections are rejected with a reload message. Earlier partial test datasets remain available in the registry and are labelled partially converted in Collection, with a full-session conversion action.
- The Slurm job continues after the browser closes or the web app restarts. The local monitor resumes status checks and downloads. Lost submission acknowledgements recover the same submission token rather than starting another job.
- Slurm manages GPU allocation independently of live collection. Each recording is limited to 100 MB and each output artifact to 1 GB; larger inputs fail explicitly. Unsupported GPU capability fails before simulator startup.
- Progress, queue reasons, job ID, connection failures, worker errors and logs are accessible from the conversion dialog. Publication requires both Slurm completion and verified output. Shared-storage visibility gets a bounded 120-second grace period; missing results never count as success. Downloads are size/checksum verified before registry publication.

## Verification

See `validation/collection-conversion.json` for browser coverage and limitations. Local tests cover data integrity, retries, deduplication, partial downloads, registration/lineage, form behavior and incompatible training inputs. `tests/collection_browser_fixture.py` starts a marked synthetic QA site on port 8091 with a disposable database and SSH disabled. It exercises the real API, download and registry code while simulating staging and scheduler/remote artifact responses.

Slurm job **3785684** completed on **oppy / Quadro RTX 6000** in 21 seconds. It converted recording 1 of the 51-recording Shadow cube session: 1 episode, 154 samples, 28 actions per sample. Every action and reconstructed joint-position observation matches the original; all initial/next/final observation alignment checks pass.

Verified output:
`/coc/flash7/ycho420/datasets/derivatives/dexverse-live/982ded94-3e0f-49de-8440-8dcd249f8443/6bee684e-4b10-4bc0-9eed-5bafca0620ed/dataset.hdf5`

SHA-256: `31a49ba77d871dedd41f3478b6c912a354e9fad2b54faa8f0ce6c408804a372e` (173,777 bytes).

The full 51-recording conversion is not complete: only recording 1 is cached on this Mac, and access to the source collection workstation remains deferred. No training or evaluation was submitted.
