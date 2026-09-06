# Live DexVerse teleoperation

Collection → Live teleoperation starts a bounded Slurm GPU session, displays the allocated node's actual address, and exposes status, logs and Stop session. One active session is allowed per installation. Requests and license consent are saved in the local database. Submission uses durable receipts: a lost SSH acknowledgement is recovered without submitting a second job. Refresh checks small status records at most once per ten seconds; it does not repeat asset validation.

## Tested stack and limits

- DexVerse `30cc673e27684b9f10186fa6bea731aed246bc9f`, Isaac Sim 5.1.0, Isaac Lab 2.3.2.
- CloudXR 5.0.1 extracted unmodified from `nvcr.io/nvidia/cloudxr-runtime:5.0.1` and run as ordinary processes inside a Slurm GPU allocation. This was tested on Ubuntu 24.04.4, NVIDIA driver 580.178.04, Quadro RTX 6000.
- NVIDIA's Apple client at `045bbbc3f960d301173b712c6ead4617f3f94eb0`, with CloudXRKit 5.0.1, builds and runs on the tested visionOS 2.5 headset. This is an observed installation result, not a claim of vendor certification or completed streaming validation.
- Task: `Dexverse-PickUpStick-v0`; robot: `floating_shadow_right`; relative DexPilot retargeting. Other combinations are unsupported until they have matching control/retargeting adapters.
- The server and full simulator scene have started successfully. Headset connection testing exposed an initial-connection failure (`0x800B1004`). After the user enabled local-network permission, the headset's direct TCP 48010 test still timed out after eight seconds. The same active port responded from sky2 while the DexVerse scene was loaded. This identifies a network reachability blocker between campus Wi-Fi and the GPU node; immersive capture is not yet verified. Do not equate server readiness, completed recording or successful task manipulation.

Native capture preserves DexVerse pickle demonstrations under the session output. These contain robot actions and scene states; they differ from the local ARKit JSONL recordings. Automatic live-demo conversion, registry publication, training and evaluation are not yet supported. The existing saved-headset-recording cycle remains independently available.

## Host configuration

`config/live_xr.json` selects the native runtime, GPU type, session duration and existing pinned entry from `config/capture_pipelines.json`. The live worker expects `bin/cloudxr-service`, `lib/`, and `share/openxr/1/openxr_cloudxr.json` within the configured CloudXR directory. The existing DexVerse source/runtime/assets must already be configured using the saved-recording setup guide. Missing paths fail before a GPU job is submitted.

On this cluster Docker access and GPU-node user namespaces are unavailable. The tested native runtime uses ordinary unprivileged processes; it does not require Docker, Apptainer execution on GPU nodes, namespace changes or altered host permissions. Preparation used Apptainer on the login host to obtain the official image and copy its unmodified `/opt/nvidia/cloudxr` payload. Runtime compatibility is host dependent: repeat a readiness test before enabling another platform.

The worker holds stdin open for CloudXR, uses a private per-job XDG runtime/TMPDIR, and checks both runtime readiness markers before starting Isaac. CloudXR's library path is isolated from Isaac's Python environment. This avoids the observed `/dev/null` epoll failure, conflicting shared temporary logs and incompatible bundled library injection. The address is derived from the current node's routing table, never reused from a previous job.

The NVIDIA CloudXR license must be accepted before starting. Acceptance is saved for the exact license URL; it is not assumed from an installed binary. Jobs stop automatically at the configured duration. Stop requests preserve completed recordings. Pending jobs are cancelled through the scheduler; running jobs first receive a graceful stop request.

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
5. When the scene opens, START calibrates and records, STOP pauses, RESET discards the active attempt. A demonstration saves after the task success condition holds for ten steps.

CloudXRKit's pinned network guide lists server TCP 48010; UDP video 47998, 48005, 48008, 48012; input 47999; audio 48000; microphone 48002. Networks must permit the required return traffic as well. A network administrator may need to allow these ports between the headset network and the allocated GPU node. Skynet does not change campus firewalls or silently substitute a relay.

Reference setup: https://isaac-sim.github.io/IsaacLab/v2.3.2/source/how-to/cloudxr_teleoperation.html
Apple local network privacy: https://developer.apple.com/documentation/technotes/tn3179-understanding-local-network-privacy
