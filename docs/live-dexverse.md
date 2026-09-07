# Live DexVerse teleoperation

Collection → Live teleoperation starts a bounded GPU session on the configured workstation or Slurm cluster, displays that host's actual address, and exposes status, logs and Stop session. The current target is the RTX 4090 workstation `rl2-bonjour` at `10.88.3.52`. One active session is allowed per installation. Requests and license consent are saved in the local database. Submission uses durable receipts: a lost SSH acknowledgement is recovered without launching a second session. Refresh checks small status records at most once per ten seconds; it does not repeat asset validation. The top-level Cluster gateway selector applies to cluster workflows; the live target is shown separately on this page.

## Tested stack and limits

- DexVerse `30cc673e27684b9f10186fa6bea731aed246bc9f`, Isaac Sim 5.1.0, Isaac Lab 2.3.2.
- CloudXR 5.0.1 extracted unmodified from `nvcr.io/nvidia/cloudxr-runtime:5.0.1` and run as ordinary user processes. Server startup was tested on Ubuntu 24.04.4 / Quadro RTX 6000 and Ubuntu 22.04.5 / RTX 4090 (driver 580.173.02).
- NVIDIA's Apple client at `045bbbc3f960d301173b712c6ead4617f3f94eb0`, with CloudXRKit 5.0.1, builds and runs on the tested visionOS 2.5 headset. This is an observed installation result, not a claim of vendor certification or completed streaming validation.
- Headset-tested combination: `Dexverse-PickUpStick-v0` + `floating_shadow_right`, with relative DexPilot retargeting. The live selector also offers Shadow left and bimanual variants with Pick up stick, Pick up cube, Stack cubes and Relocate sphere from the same release. These additional combinations are source-supported but have not yet passed a headset test on this workstation. The UI states this explicitly. WUJI, Sharpa, Allegro, LEAP and Inspire models remain available in the hand library, but the pinned public DexVerse release does not ship their live simulation/retargeting adapters; their live choices are disabled with an explanation.
- Previous cluster and `rl2-ws11` tests timed out on TCP 48010 from the headset, although the port responded from the cluster. On `rl2-bonjour`, both the Mac and the headset reached TCP 48010. The headset then established a streaming session, and the operator confirmed seeing the task and controlling the robot hand. Server logs also confirmed the Play command and right-wrist calibration.
- On September 6, 2026, session `6a335a76-3fc9-4e0a-a644-b5254be0d4a3` completed the task and saved one successful demonstration. Validation confirmed 233 actions of dimension 28, 234 scene states, and finite numeric values. The native file is 315,340 bytes, SHA-256 `b1d1de5a9eb8091e913a2ff3fcb683fa128e919f894e1df5d6e5c9dad4b2fcdc`. It remains in that session's `output/recordings/live/Dexverse-PickUpStick-v0/` folder, with a validated copy on the Mac. The service completed cleanly, TCP 48010 closed, and GPU use returned to idle. This verifies live capture; it does not establish native-demo training/evaluation support.

## Review saved demonstrations

Choose **Review recording** in a completed live-session row. Skynet downloads the original to `data/live-reviews/<session>/<recording index>/` on the Mac, validates its format, selected task/hand, successful episodes, all numeric states/actions, and state/action alignment, then caches the review. A recorded checksum, when available, must match. A restricted NumPy-only pickle reader rejects arbitrary classes; this endpoint accepts only files produced in a known completed session, never uploaded pickle files or arbitrary remote paths. Downloads are bounded to 100 MB and interrupted downloads can be retried without launching a GPU job.

The review shows all scene frames for recordings up to 2,000 states, with Previous/Next, playback, seeking, field selection and downloads for the original, summary and JSON preview. Longer recordings explicitly show a sampled preview including the initial/final states; every state is validated and the original remains complete. Frame 0 is the initial scene and has no preceding action; action 0 produces frame 1. Time is simulation time at the pinned 60 Hz rate, not elapsed headset time. Field labels expose known pose and velocity components; joint/action names and units are not guessed because this native format does not include their mapping. Playback displays recorded values, not rendered scene video. Local review works without SSH once downloaded.

The real September 6 capture is cached locally: 233 actions × 28 values, 234 states, 3.883 seconds of simulation. Browser tests cover frame navigation, playback, alternate fields, selector persistence and downloads. Additional task/hand GPU validation is pending restoration of the workstation network route.

Native capture preserves DexVerse pickle demonstrations under the session output. These contain robot actions and scene states; they differ from the local ARKit JSONL recordings. Before reporting a completed capture, the worker validates the pinned format/task/robot, nonempty successful episodes, finite actions, and one initial state plus one state per action. This inspection is only for trusted files produced by the session, not arbitrary uploaded pickle files. The live collection workflow stops at capture and review; it does not start conversion, registry publication, training or evaluation. Existing cycle history and the separate saved-headset-recording workflow remain available.

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

Check out the exact revision above, then apply `ops/xr/headset-client-errors.patch` from this repository. It retains connection failures, makes long errors scrollable, adds a local-network usage description and an explicit permission/discovery action, and provides a bounded TCP 48010 test. Successful TCP testing does not prove UDP video/tracking connectivity. The patch does not embed a server address, signing team or account.

Open the Xcode project, select your developer team and headset, and build with the appropriate Apple signing configuration. The tested personal-team build used an explicit empty entitlements file instead of the optional low-latency streaming entitlement; do not enable an entitlement your team cannot sign. Pairing and installation may require a private Wi-Fi network when campus Wi-Fi blocks device discovery. Do not delete existing headset recordings to troubleshoot streaming.

## Connection checks

1. Start a session and wait until its scene is loaded. Enter the current address shown in Skynet.
2. In the client choose Enable local network and allow the system prompt. Then choose Test server connection.
3. If TCP 48010 is unreachable, verify the active node, headset network and the network administrator's permitted route. SSH from a Mac is a separate connection and does not prove that the headset can stream.
4. If the port is reachable, choose Connect. Report the complete category and error from Show full error if it fails.
5. When the scene opens, hold the selected hand(s) in front of you. Play calibrates and records, Stop pauses, and Reset discards the active attempt. The client sends these as START, STOP and RESET commands, respectively. A demonstration saves after the task success condition holds for ten steps.

For Pick up stick, lift the stick at least 20 cm above its default starting height and hold it vertical, within 30 degrees of pointing straight up. Keep recording until it saves and the session closes. Stop only pauses: it does not save the active attempt. Reset discards that attempt. Completed demonstrations remain on disk when a session stops.

CloudXRKit's pinned network guide lists server TCP 48010; UDP video 47998, 48005, 48008, 48012; input 47999; audio 48000; microphone 48002. Networks must permit the required return traffic as well. A network administrator may need to allow these ports between the headset network and the allocated GPU node. Skynet does not change campus firewalls or silently substitute a relay.

Reference setup: https://isaac-sim.github.io/IsaacLab/v2.3.2/source/how-to/cloudxr_teleoperation.html
Apple local network privacy: https://developer.apple.com/documentation/technotes/tn3179-understanding-local-network-privacy
