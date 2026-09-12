# Live DexVerse teleoperation

Data → Collect starts a bounded GPU session on the configured workstation or Slurm cluster, displays that host's actual address, and exposes status, logs and Stop session. The current target is the RTX 4090 workstation `rl2-bonjour` at `10.90.45.22`. One active session is allowed per installation. Requests and license consent are saved in the local database. Submission uses durable receipts: a lost SSH acknowledgement is recovered without launching a second session. Startup refreshes small remote status records at most once per two seconds, then once per ten seconds after the scene is ready; it does not repeat asset validation. The top-level Cluster gateway selector applies to cluster workflows; the live target is shown separately on this page.

The startup panel below **Start session** tracks the authenticated server connection, runtime check, hand files, streaming readiness and loaded hand/simulation. Completed milestones and the failed stage are saved with the session. A failure remains visible beside the controls after Start becomes available again, including after reloading the session link. Older sessions show their recorded error without invented milestones. Stream readiness comes from CloudXR's runtime markers; hand/simulation readiness requires the recorder's post-reset teleoperation marker. Ready means the scene can accept a headset connection, not that recording has started.

## Tested stack and limits

- DexVerse `30cc673e27684b9f10186fa6bea731aed246bc9f`, Isaac Sim 5.1.0, Isaac Lab 2.3.2.
- CloudXR 5.0.1 extracted unmodified from `nvcr.io/nvidia/cloudxr-runtime:5.0.1` and run as ordinary user processes. Server startup was tested on Ubuntu 24.04.4 / Quadro RTX 6000 and Ubuntu 22.04.5 / RTX 4090 (driver 580.173.02).
- NVIDIA's Apple client at `045bbbc3f960d301173b712c6ead4617f3f94eb0`, with CloudXRKit 5.0.1, builds and runs on the tested visionOS 2.5 headset. This is an observed installation result, not a claim of vendor certification or completed streaming validation.
- Thirteen hand variants are available across Pick up stick, Pick up cube, Stack cubes and Move sphere to target. Current simulator and headset evidence is recorded in `docs/validation/hand-following.json`. The UI distinguishes combinations without headset or saved-episode evidence.
- Previous cluster and `rl2-ws11` tests timed out on TCP 48010 from the headset, although the port responded from the cluster. On `rl2-bonjour`, both the Mac and the headset reached TCP 48010. The headset then established a streaming session, and the operator confirmed seeing the task and controlling the robot hand. Server logs also confirmed the Play command and right-wrist calibration.
- On September 6, 2026, session `6a335a76-3fc9-4e0a-a644-b5254be0d4a3` completed the task and saved one successful demonstration. Validation confirmed 233 actions of dimension 28, 234 scene states, and finite numeric values. The native file is 315,340 bytes, SHA-256 `b1d1de5a9eb8091e913a2ff3fcb683fa128e919f894e1df5d6e5c9dad4b2fcdc`. It remains in that session's `output/recordings/live/Dexverse-PickUpStick-v0/` folder, with a validated copy on the Mac. The service completed cleanly, TCP 48010 closed, and GPU use returned to idle. This verifies live capture; it does not establish native-demo training/evaluation support.

## Shared hand models

All new sessions use the [canonical Hands asset contract](hand-assets.md), including Shadow. Validation receipts below describe historical bundle digests; they do not establish simulator or headset validation for a different bundle.

## Imported hand adapters

Open a stored model in **Hands**, choose its side, then choose **Use in simulation**. This opens Collect with that exact hand selected. Choose one of the four tasks and start a session. Imported hands use their actual URDF visuals, collisions, masses, inertias, joint limits and mimic relationships; they do not load a Shadow model in their place.

| Model | Imported sides | Independent finger joints | Actions, including six wrist joints |
| --- | --- | ---: | ---: |
| Shadow | Right, left, both | 22 per hand | 28 per hand; 56 for both |
| WUJI Hand 1 | Right, left | 20 | 26 |
| WUJI Hand 2 (Beta 2) | Right, left | 20 | 26 |
| Sharpa Wave | Right, left | 22 | 28 |
| Allegro Hand V4 | Right | 16 | 22 |
| LEAP Hand V1 | Right | 16 | 22 |
| Inspire RH56 | Right, left | 6 | 12 |

Allegro left remains explicitly unsupported because its pinned source references a missing thumb mesh. LEAP's pinned repository supplies only a right hand. Imported bimanual combinations are not configured. The native Shadow right, left and bimanual choices remain available.

`config/hands.json` (`floating_hand`) records each family’s palm, ordered fingertips and side-specific alignment. Simulation bundles preserve the stored visuals, collisions, masses, joint limits and mimic relationships, and add a six-joint floating wrist. All selectable hands now use DexVerse’s `SimpleAbsoluteRetargeter` and vanilla DexPilot finger solver. The only wrist wrapper selects equivalent continuous angles to prevent backward spins at the ±180° boundary. See [hand tracking](hand-tracking.md) for the current mapping and Inspire model fixes.

WUJI Hand 1 also uses DexPilot, matching the other hands. Four-finger models map human thumb/index/middle/ring to their four fingertips. Inspire preserves its six mimic relationships. Joint mappings are explicit; incomplete mappings stop startup.

The bundle contains the URDFs, source meshes and license assets, retargeting configuration, source-name mapping, source revision, and hashes. Its identity includes the recipe, source model/assets and runtime adapter code. The first session uploads the bundle through the existing SSH transport and converts it to USD on the GPU workstation with Isaac Lab's URDF converter. Later sessions reuse the bundle and conversion cache. A small remote READY marker avoids repeated transfer; startup verifies bundle contents before loading. The runtime adds the selected robot to the pinned DexVerse process without modifying the upstream checkout. It checks simulator joints, bodies, action dimension and finite initial positions before enabling collection. Conversion, mapping or transport failures appear in session errors/logs; they never trigger another robot or backend.

Imported recordings include `skynet_hand` metadata with the exact source revision, bundle digest and ordered action joint names. The six wrist controls precede independent finger controls; mimic joints are not separate actions. Collection saves and reviews episodes; dataset conversion is a separate action in Recordings. No training or evaluation is started.

### Validation status

The [CPU validation receipt](validation/imported-hands-cpu.json) records all ten model identities and results. `ops/xr/hands/check_kinematics.py` uses the real dex-retargeting and Pinocchio libraries to check 42 gradually changing, reachable finger targets per hand (420 total), joint bounds and mimic relationships. Maximum fingertip-vector error was below 3.4 mm on this synthetic test. Independent checks also verify palm-down alignment and left/right thumb placement against the native Shadow frame. Four combined wrist poses per hand check the floating URDF's rotation/translation convention. These are kinematic checks against reachable model targets, not tests of human tracking, collision stability or successful grasping.

Run that check in an isolated environment with the dependency versions recorded in the receipt, passing one or more paths from `data/simulation-hands/<robot>/<digest>/`. Unit tests also exercise source/bundle corruption, atomic upload/cache reuse, incomplete preparation, exact session selection and the hand-library link. Browser checks cover the real model view, **Use in simulation**, selection persistence, task changes and unsupported-side explanation.

The latest [hand-following receipt](validation/hand-following.json) records simulator checks for all thirteen selectable hand variants. The operator confirmed Shadow left and Inspire right tracking in the headset. Automated model checks do not imply every hand/task pair has been headset-tested.

## Review saved demonstrations

Choose **Review recordings** beside a session, then select a recording in the modal. Each episode has an immutable file and checksum, so collecting the next episode does not invalidate an existing review. Skynet reads the original into bounded memory, validates its format, selected task/hand, successful episodes, numeric states/actions, and alignment, then stores the review remotely. Completed sessions are archived on sky2; active sessions remain on their collection workstation until they finish. Review payloads are not cached on the Mac. A recorded checksum, when available, must match. A restricted NumPy-only pickle reader rejects arbitrary classes; this endpoint accepts only files produced in a known completed session, never uploaded pickle files or arbitrary remote paths. Downloads are bounded to 100 MB and interrupted downloads can be retried without launching a GPU job.

The **Recorded values** section shows all scene frames for recordings up to 2,000 states, with Previous/Next, playback, seeking, field selection and downloads for the original, summary and JSON preview. Longer recordings explicitly show a sampled preview including the initial/final states; every state is validated and the original remains complete. Frame 0 is the initial scene and has no preceding action; action 0 produces frame 1. Time is simulation time at the pinned 60 Hz rate, not elapsed headset time. Field labels expose known pose and velocity components; joint/action names and units are not guessed because this native format does not include their mapping. Video is shown above the recorded values. Review and playback stream from the saved remote location and require that host to be reachable.

The September 6 validation receipt recorded: 233 actions × 28 values, 234 states, 3.883 seconds of simulation. Browser tests cover frame navigation, playback, alternate fields, selector persistence and downloads. See the validation receipts for later task and hand checks.

Native capture preserves DexVerse pickle demonstrations containing robot actions and scene states. These differ from local ARKit JSONL motion recordings. Completed episodes have validated task/hand identity, successful outcomes, finite numeric values, and T actions with T+1 states. Open **Collection → Recordings → Convert for training** to export a selected collection as state HDF5 and register its source lineage. Conversion is explicit and does not start training or evaluation. See [dataset conversion](collection-conversion.md) for compatibility and verification limits.

## Host configuration

`skynet_app/live_xr_catalog.py` holds the small, revision-pinned live hand/task catalog; page loads do not import Isaac Sim or repeat remote validation. The selected hand and task are frozen into each session request and passed to both the recorder and capture validator. Unsupported API requests fail before launch, and a request for a different combination while a session is active reports a conflict instead of reopening the wrong session.

`config/live_xr.json` selects the execution backend, SSH host, native runtime, session duration and existing pinned entry from `config/capture_pipelines.json`. The live worker expects `bin/cloudxr-service`, `lib/`, and `share/openxr/1/openxr_cloudxr.json` within the configured CloudXR directory. The existing DexVerse source/runtime/assets must already be prepared. Missing paths fail before launching a session.

With `execution: "workstation"`, provide `gateway` (the passwordless SSH alias), a dedicated absolute `work_root`, and `repository`, `runtime`, and `cloudxr_runtime` paths inside it. `memory_gb` bounds session RAM; `duration_minutes` must be 5–60. The host must have a working user systemd manager and NVIDIA graphics/compute drivers. Workstation mode never submits a Slurm job or falls back to the cluster. Existing session history retains its original backend and host.

The prepared Python environment must be relocated to its final path, including editable IsaacLab package paths. Validate Python imports, CUDA tensor execution, the pinned source-revision marker, robot models and task assets, then save a JSON validation receipt as `runtime/.skynet-runtime-ready.json`. A copied conda environment with paths still pointing to the original host is not ready. The current installation uses `conda-pack` 0.8.1 and its binary-prefix relocation helper, plus explicit relocation of the six IsaacLab editable packages. Package versions and assets remain pinned.

Each workstation session runs as `skynet-live-<session UUID>.service` through `systemd-run --user`. It has a dedicated directory, private file permissions, CPU/RAM limits and a duration limit with five minutes for cleanup. Automatic restart is disabled. Stop targets only that session's unit and gives the worker time to save logs and stop its children. A durable launch attempt and result distinguish failure from a lost acknowledgement; retries recover the same unit. A missing unit is only considered completed when its matching worker status records a clean terminal state.

On this cluster Docker access and GPU-node user namespaces are unavailable. The tested native runtime uses ordinary unprivileged processes; it does not require Docker, Apptainer execution on GPU nodes, namespace changes or altered host permissions. Preparation used Apptainer on the login host to obtain the official image and copy its unmodified `/opt/nvidia/cloudxr` payload. Runtime compatibility is host dependent: repeat a readiness test before enabling another platform.

The worker holds stdin open for CloudXR, uses a private per-job XDG runtime/TMPDIR, and checks both runtime readiness markers before starting Isaac. CloudXR's library path is isolated from Isaac's Python environment. This avoids the observed `/dev/null` epoll failure, conflicting shared temporary logs and incompatible bundled library injection. The address is derived from the current node's routing table, never reused from a previous job.

The NVIDIA CloudXR license must be accepted before starting. Acceptance is saved for the exact license URL; it is not assumed from an installed binary. Sessions stop automatically at the configured duration. Stop requests preserve completed recordings. In Slurm mode, pending jobs are cancelled through the scheduler and running jobs receive a graceful stop request. In workstation mode, the user service receives a graceful stop signal and systemd bounds cleanup time.

## Headset client setup

Use the official client repository:
https://github.com/NVIDIA/isaac-xr-teleop-sample-client-apple

Check out the exact revision above, then apply `ops/xr/headset-client-errors.patch` from this repository. Connect requests local-network permission automatically. A floating panel follows the user’s view and contains the task instruction, Start, Restart episode and End collection. The client shows one red-human / blue-robot legend. Connection failures immediately open a scrollable modal with the full error category, domain and code. Diagnostic buttons and eye-position controls are removed. The patch does not embed a server address, signing team or account.

Open the Xcode project, select your developer team and headset, and build with the appropriate Apple signing configuration. The tested personal-team build used an explicit empty entitlements file instead of the optional low-latency streaming entitlement; do not enable an entitlement your team cannot sign. Pairing and installation may require a private Wi-Fi network when campus Wi-Fi blocks device discovery. Do not delete existing headset recordings to troubleshoot streaming.

## Connection checks

1. Start a session and wait until its scene is loaded. Enter the current address shown in Skynet.
2. Choose Connect and allow local-network access if prompted.
3. If connection fails, the error modal opens automatically. Its full category, domain and code can be selected and copied.
4. For network errors, verify the active server and headset network route. SSH from a Mac is a separate connection and does not prove that the headset can stream.
5. Red points show the human hand; blue points show the robot joints. They remain visible before recording. No alignment or open-finger pose is required.
6. Tap **Start** in the headset. Fresh hand tracking and the control connection are required. The task instruction floats near the top of the user's view and follows the headset. Success must hold for ten simulation steps.
7. The instruction changes to **Episode ended. Saving…**. The simulator keeps rendering while a background worker writes and syncs a new episode file, then publishes its checksum receipt. After saving, the scene resets and waits for **Start** again without disconnecting CloudXR.
8. **Restart episode** discards only the unfinished attempt. **End collection** finishes any in-progress save, discards an unfinished attempt, acknowledges the saved count, and closes the session. The web **Stop session** also shuts down gracefully. Previously saved episodes remain reviewable after stop, timeout, or failure.

Tracking loss pauses movement and recording. A dropout under 0.5 seconds resumes the same episode; sustained loss discards the incomplete attempt and shows an interruption message before returning to the ready state. Out-of-bounds task termination also discards the attempt rather than splicing an automatic simulator reset into a recording. Saving errors stop the workflow with an explicit error; they do not silently reset into another attempt.

The session capsule freezes `ops/xr/collection.py` and the anatomical-frame helper alongside the existing worker. It launches the pinned DexVerse setup and recorder with a custom episode loop. Successful episodes are atomically saved separately under `output/recordings/live/episode-NNNNNN.pkl`. Parent validation checks each new receipt/checksum once and makes its review available during collection. The read-only status service exposes only `GET /collection` on workstation TCP 48011; controls and a liveness heartbeat use the existing CloudXR client-to-server channel. The headset reports unavailable/stale status or unacknowledged controls as an error. This port must be reachable in addition to the streaming ports.

For Pick up stick, lift the stick at least 20 cm above its default starting height and hold it within 30 degrees of vertical, with either end up. The collection runtime checks both directions of the stick's axis; DexVerse's original directed-axis check rejected an equivalent vertical pose, particularly when lifted with the left hand. The height threshold and ten consecutive successful steps are unchanged. Other tasks retain their original orientation conditions. After success, the episode saves and the scene resets and waits for the next Start click. Completed demonstrations remain on disk when a session stops.

CloudXRKit's pinned network guide lists server TCP 48010; UDP video 47998, 48005, 48008, 48012; input 47999; audio 48000; microphone 48002. Networks must permit the required return traffic as well. A network administrator may need to allow these ports between the headset network and the allocated GPU node. Skynet does not change campus firewalls or silently substitute a relay.

Reference setup: https://isaac-sim.github.io/IsaacLab/v2.3.2/source/how-to/cloudxr_teleoperation.html
Apple local network privacy: https://developer.apple.com/documentation/technotes/tn3179-understanding-local-network-privacy

Red points are the tracked human joints in the scene. Blue/cyan points are wrist-centered, rotation-normalized human points used by the finger retargeter, drawn back at the tracked wrist position. They are not measured robot fingertip positions. The legend appears in both the collection page and the headset teleoperation controls.

The [GPU validation receipt](validation/hands-gpu.json) records bounded checks on rl2-bonjour: imported-hand USD conversion, actual OpenXR-to-canonical conversion, 330 physics steps of hand input and neutral-pose recovery per imported variant. The `ops/xr/hands/check_simulation.py` and `check_native.py` utilities reproduce these checks with the pinned recorder and prepared bundles. Human fingertip goals may be limited by self-contact; the receipt retains those tracking residuals separately. These checks do not establish headset tracking quality or successful grasping for every hand/task pair.

The [automatic collection receipt](validation/collection-episodes.json) records a later WUJI 2 check with real Isaac physics and recorder code, synthetic tracking, and a synthetic success signal. It deliberately offsets wrist axes by 25°, interrupts a five-step attempt, then verifies two separate 20-step saves, automatic resets, and End collection. The interrupted attempt is absent from the saved files. `ops/xr/check_collection.py` reproduces this bounded check outside the user recording directory. Headset alignment, instructions, and the corrected finger mapping still need operator confirmation.

## Recording review

Each session has one **Review recordings** button. Choose a recording in the modal, then use the video controls to play, seek, or download it. The **Recorded values** section contains the corresponding saved scene states and actions.

The current CloudXR 5.0.1 / Isaac Sim 5.1 integration supports **scene replay**: a fixed camera renders the recorded hand and object states into a 960 × 540, 30 fps MP4. Rendering restores saved states directly without rerunning the policy, training, evaluation, or physics actions. Videos stay on remote storage and stream directly to the browser. Archived recordings use cluster storage and do not need the workstation.

Preparing an uncached replay uses a bounded Slurm GPU job when the session is archived on sky2; an unarchived active collection still uses its original workstation. The modal distinguishes queueing, confirmed rendering, final cleanup, verification, and browser loading. A workstation lock conflict stays in the waiting stage instead of alternating with rendering. GPU acquisition is bounded to two minutes, and the renderer has a ten-minute runtime limit. Cancellation targets only the selected preparation generation and waits for confirmed process/scheduler shutdown. Interrupted or uncertain preparation remains visible until cancellation is confirmed; retries cannot silently launch a second job. Closing review does not cancel preparation. Browser media loading has its own retryable timeout.

See [collection storage](collection-storage.md) for automatic verified archiving and cleanup.

Direct camera capture during live XR is unsupported in this integration. The XR renderer replaces the normal camera view with its stereo view and can return empty RGB buffers. Live collection therefore records the scene trajectory, and review labels its video **Scene replay**. If a recording contains a failed video-capture attempt, its error is shown alongside the replay.
