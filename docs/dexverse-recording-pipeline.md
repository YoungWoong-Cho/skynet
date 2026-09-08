# Offline local tracking experiment

For simulation demonstrations collected in the headset, use **Data → Collection → Recordings → Convert for training**. See [teleoperation dataset conversion](collection-conversion.md).

This older workflow is under **Data → Collection → Setup → Offline experiments & history**. Choose an imported local tracking recording, check the setup, and run the offline experiment. One cluster GPU is used. It automatically trains and evaluates; it is separate from converting a saved teleoperation dataset.

The implemented combination is **Vision Pro right-hand tracking → floating Shadow right hand → Dexverse-PickUpStick-v0 → Skynet state-based behavior cloning**. It runs with the existing visionOS 2.5 recorder. Saved-file processing does not use CloudXR.

This is a state-based policy, not GR00T or pi0.5. It learns from robot joint positions and the simulated object's position/orientation features. Using a different model, robot or simulator requires an explicit matching observation/action adapter. Unsupported combinations fail rather than substituting a different task or policy.

## What a completed cycle means

1. The original JSONL file is copied without rewriting it, and its SHA-256 is verified on the cluster.
2. Tracked right-hand joints are converted and replayed through DexVerse's DexPilot/relative-wrist retargeter. Actual physics produces observations, robot states and task outcomes.
3. An HDF5 dataset is saved and reopened by the training stage. A small MLP is trained for the requested epochs and its weights are saved.
4. Evaluation loads that saved checkpoint and runs the policy in newly seeded simulator scenes. It does not replay the original recorded actions as evaluation.
5. Output checksums are verified before the dataset is marked ready and linked to its original capture.

A cycle can complete with **zero successful task trials**. A hand-movement recording made without seeing the simulator is a motion test, not evidence of a successful task demonstration. Review the replay before using the data for task learning. One short episode does not establish generalization. Training loss measures fit to the training recording; evaluation reports actual simulator success.

## Setup and recovery

The **Check / configure DexVerse** button verifies the pinned source files, links the existing verified asset bundle, and checks the existing runtime packages. It never replaces a shared Python runtime or overwrites source files that differ from the pinned revision. Setup is safe to repeat.

The operator configuration is `config/capture_pipelines.json`:

- `local_source`: the local DexVerse Git checkout containing the pinned commit.
- `repository`: an isolated destination below the configured cluster workspace.
- `runtime`: the existing Isaac Sim 5.1.0 / Isaac Lab 2.3.2 / Python 3.11 environment.
- `asset_bundle`: a prepared asset bundle with a `READY` marker and the expected robot assets. The existing asset preparation script is `ops/datasets/download_dexverse_assets.sbatch`.
- `gateway`, `account`, `partition`: the explicit cluster route and queue. Processing uses this saved route; the dashboard's gateway selector does not rewrite an existing job.

The equivalent operator command is:

```bash
.venv/bin/python -m skynet_app.capture_processing.setup
```

Source is pinned to DexVerse commit `30cc673e27684b9f10186fa6bea731aed246bc9f`. The default runtime and asset paths refer to the already configured Skynet cluster. Another installation must supply its own compatible paths. Missing packages/assets produce a setup error; the button does not claim to install an NVIDIA GPU, Isaac Sim, or licensed components on the Mac.

The first launch on a GPU can spend several minutes compiling shaders. Subsequent launches reuse the simulator cache. Each job requests one GPU, four CPU cores, 48 GB RAM and a 30-minute limit. Use short recordings and modest evaluation counts for this first pipeline.

Repeated clicks with identical recording bytes, parameters and pipeline source reopen the same request. If the network drops during submission, the original submission token and exact script are used for recovery. An uncertain acknowledgement is not shown as a failed job and does not automatically create another job. Finished attempts and their artifacts are immutable; use another scene seed or an updated pipeline version for a new attempt.

## Dataset contract

Format: `skynet.dexverse-state-actions/v1`. HDF5 is parsed without unpickling uploaded data.

- `data/demo_0/obs/state`: float32 `[T, 35]`, ordered as 28 robot joint positions followed by seven task object-state features.
- `data/demo_0/next_obs/state`: float32 `[T, 35]`, observations after each action.
- `data/demo_0/actions`: float32 `[T, 28]`, exact commands passed to Isaac Lab's ordered translation, rotation and finger action terms.
- `robot_joint_positions`, `rewards`, `source_indices`, `success`, `outcomes/*`: recorded state, reward, source-frame mapping and actual termination conditions.
- `initial_scene` and `scene_states`: the simulator's structured numeric scene state, including the initial state and every post-step state. No arbitrary Python objects are serialized.
- Root `metadata`: task, embodiment, source checksum, converter/calibration, observation order, frequency, seed and outcome.

Actions use DexVerse's existing relative-wrist calibration and finger mapping. They are not asserted to be safe real-robot actuator commands. The dataset has no recorded RGB observation columns; the MP4 files are review artifacts. GR00T/pi0.5 require additional supported embodiment/image-data conversion and a corresponding simulator evaluator.

The checkpoint schema is `skynet-state-bc/v1`: network weights, training normalization, exact observation order, observed action ranges and dataset checksum. Evaluation reloads it with PyTorch's restricted weights loader. Predicted actions are bounded to the observed training range; the number of bounded actions is included in the evaluation result.

## Tracking conversion

The converter validates the recording schema and rigid transforms, requires all 21 joints used by DexPilot, and rejects untracked joints or gaps greater than 100 ms. It preserves both the original source timestamp and the receiving clock, resampling at the simulator's 60 Hz control rate with causal holding. There is no interpolation across tracking loss and no generated joint data.

ARKit's Y-up world is rotated into a right-handed Z-up simulator frame. The wrist basis is calibrated from the first frame's wrist/knuckles; later orientation comes from the tracked hand anchor. The basis follows the [OpenXR hand-joint convention](https://registry.khronos.org/OpenXR/specs/1.1/html/xrspec.html#convention-of-hand-joints): fingers extend along local -Z and the dorsal direction is +Y. Finger orientations are unused by DexPilot; the preserved finger positions determine retargeting.

## Extending collection

Raw recording providers remain behind `CaptureProvider` in `skynet_app/local_capture.py`. Keep original files immutable. Add another provider with its own native schema validator rather than converting unknown input to Vision Pro format.

For another processing combination, add a versioned converter and worker with explicit task, embodiment, observation and action contracts. Register its supported configuration in the processing service and catalog; publish its validation and simulator tests. Shared job storage, upload checksums, recovery, artifact delivery and dataset-lineage registration can be reused. Merely editing a catalog task/model name does not make a combination supported.

Live Vision Pro simulator teleoperation is a separate integration. The patched CloudXR 5.0.1 client has been installed and launched on the tested visionOS 2.5 headset; immersive streaming still requires a working network route. See [live setup](live-dexverse.md) for current tests and limitations.
