# Lazy observation preparation

## Lifecycle

1. Collection saves immutable actions, scene states and CPU preview metadata. No post-session `render_images.py` or `create_demo_files_sequential.py` run is scheduled.
2. Recording View reads values and frozen hand/scene assets. It never requests training observation or video generation.
3. Convert resolves `observation_requirements` from the converter/adapter contract. Current DP, ACT and EgoVerse RGB recipes request the three fixed scene cameras at 256 × 256; the state-only recipe requests no camera artifacts.
4. The planner creates per-source, per-camera RGB or depth nodes, followed by derived point-cloud nodes when declared. The database claims only missing nodes.
5. GPU producers restore each saved pre-action state without stepping physics. CPU producers unproject verified depth, merge requested camera views, crop, then sample deterministically.
6. The existing native-format exporter consumes the verified observations and original joint commands. Its final v1 dataset remains self-contained and is verified by the current training loader.

This is observation infrastructure for adapter requirements. It does not add UniDex or OPFA policy implementations. Their observation contracts can use the same RGB/depth/XYZ/XYZRGB preparation primitives when their converters and training adapters are added.

## Storage and identity

Artifacts live under `WORK_ROOT/datasets/observations/<source-sha256>/<artifact-key>/`. Each contains `values.hdf5` and a checksummed manifest. HDF5 stores values, ordered frame IDs, control timestamps, and per-frame camera intrinsics and optical-to-world transforms. Depth is float32 metric image-plane distance with an explicit validity mask. Point-cloud padding also has a validity mask; repeated points are not silently counted as new geometry.

The artifact key hashes the source checksum, episode index, timing convention, requested camera recipe, shape/modality, pinned rendering code, source revision, scene asset inventory and frozen hand identity plus manifest checksum. Render workers disable loading and saving shared Kit user preferences before simulator startup, so another session’s persisted graphics settings do not alter the frozen recipe. Adapter name, dataset label, train/validation split and storage location do not change the key. A changed camera, source, renderer or asset identity cannot reuse stale values. Adding another view leaves existing view keys unchanged.

Point-cloud identity additionally includes camera order, XYZ/XYZRGB channels, coordinate frame, depth range, crop bounds in that coordinate frame, point count, sampler and seed. Multi-camera clouds merge all valid points before final sampling. RGB and depth must have matching source, scene realization, timing and calibration before color is attached to points. Requesting another crop or count reuses depth.

These are immutable data artifacts, not expiry caches. New source files or updated requirements create new identities. PostgreSQL notifications invalidate the existing frontend snapshots after transactions commit; no TTL delays publication.

## Ownership and recovery

Migration 014 adds artifacts, source aliases, dependency edges, preparation consumers, dataset-version consumers and independent producer jobs. A transaction claims missing artifacts before any remote submission. Producer leases prevent concurrent monitors from publishing competing results; expired owners cannot update or publish after another owner recovers the lease. Submission uncertainty retains the same attempt and submission token.

A separate parent process supervises native rendering. Initialization and episode setup have a 300-second progress deadline; capture, publication and shutdown have a 120-second progress deadline. The parent terminates only its renderer process group and writes an attempt-bound failure receipt if the native process crashes, hangs or exits without a result. Actual frame progress extends the deadline; an idle heartbeat does not. CPU derivation does not start this simulator supervisor.

Only a definite terminal failure permits a fresh producer attempt. Verified files already published by a partially successful worker can be reused after retry. Files are written in staging directories inside the producer capsule, on the same filesystem as their destination, and atomically published under their content identity; existing verified output is never overwritten. Publishing a dataset version and its observation references occurs in one database transaction.

Deleting the Convert that initiated work does not remove a producer needed by another consumer. Dataset deletion removes that dataset's references. Recording deletion checks active producers, shared aliases, dependencies and all consumers before removing remote files, then deletes exclusive outputs before their inputs. Pending deletion intents prevent new consumers from attaching after an interrupted deletion.

## Compatibility and limits

Existing dataset versions, trained checkpoints, native loaders and downloads keep their v1 behavior. Shared observations are converter inputs and provenance; prepared datasets contain the arrays required by their current native format. This retains per-format copies while eliminating repeated observation rendering. Existing saved image sidecars remain readable, but new sessions never create them automatically.

The renderer targets the pinned DexVerse v0/v1 sources and fixed asset layouts. Imported hand USD is regenerated from verified source files in a private producer directory; rendering never trusts or changes a shared generated-USD cache. Unknown revisions, changed hand/asset fingerprints, incompatible camera recipes, multi-asset random layouts, or missing state/goal data fail explicitly. Original recordings are preserved. Legacy native trajectories without recorded timestep metadata use the exact pinned simulator timestep and identify that source in capture provenance. Legacy 3D previews cannot reconstruct metadata that was never saved; they explain the missing fields without starting a simulator.

## GPU validation status — 2026-09-15

Real camera RGB/depth generation, point-cloud derivation from those frames, reuse and imported-hand rendering have not completed end-to-end GPU validation. Local geometry, converter, database, process-supervision and browser checks pass. The installed Isaac Sim 5.1 native RTX renderer segfaulted before `SimulationApp` returned on tested A40 nodes with driver 595.84. A minimal script without DexVerse, Pinocchio or physics reproduced the crash, including with isolated user settings and a clean library environment. This does not establish a specific driver defect.

The allocated L40S reported an uncorrectable ECC error; configured alternative nodes were occupied. No original recording or production dataset was changed, and all owned test jobs and private capsules were cleaned up. The next required check is a working native camera baseline, followed by one original episode through RGB, missing depth, CPU point clouds and reuse, then an imported-hand episode. Synthetic backend tests do not establish that result.

## Rollback

The migration is additive. Reverting application code does not require dropping observation tables or deleting shared files; v1 datasets remain readable. Reconcile or stop outstanding observation producers before removing their monitor code. Preserve metadata and artifacts until no recorded consumer references them. The migration file alone is not an automatic rollback mechanism; deleting referenced artifact tables or files would discard provenance and can orphan in-flight work.
