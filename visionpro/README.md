# Skynet Capture (visionOS 2.0+)

A native offline recorder for Apple Vision Pro. No NVIDIA GPU or CloudXR connection is needed. Hand joints and device poses come from ARKit; camera images, object poses, task success and robot actions are not synthesized.

## Install and test

1. Open `SkynetCapture.xcodeproj` in Xcode on a Mac with the visionOS SDK.
2. Select the app target, Signing & Capabilities, your Apple development team, and a unique bundle identifier. Keep automatic signing enabled.
3. Pair the physical Vision Pro in Xcode's Devices and Simulators window. Enable Developer Mode on the headset, select it as the run destination, and run the app. Developer provisioning may require Apple account authentication and headset confirmation.
4. Enter a task name, open tracking space, and allow hand tracking. Keep both hands visible. The Start button becomes available after tracking arrives.
5. Record approximately 10 seconds, move both hands and your head, then Stop and save. Confirm the saved frame count is nonzero.
6. Share the saved JSONL file to the Mac via AirDrop or Files. In the Skynet browser, Data → Collection → Import recording. Verify left/right/head counts, duration and warnings. Export the original and compare SHA-256 if checking data integrity.
7. Repeat the same import: it should report the existing recording. Close the immersive space during a second recording: it must be labeled interrupted and rejected as a completed dataset.

The visionOS simulator has no real hand-tracking provider. Its unsupported message is expected and does not verify headset capture. Recordings remain in the app's Documents/Skynet Recordings folder; sharing does not delete them. Uninstalling the app may remove its local data.

## Native schema

Each UTF-8 JSONL file has one `header`, sequential `frame` records, and one `footer`. Schema: `skynet.visionpro-tracking/v1`.

- Header: session UUID, task/operator labels, creation time, app/OS versions, stream declarations, frames and units.
- Frame: receive-clock timestamp and original ARKit timestamp (monotonic seconds), hand chirality and tracking flags, 4×4 origin-from-hand transform, named hand joints with anchor-from-joint transforms, and optional origin-from-head pose with tracking status. Matrices are column-major; translations are meters, in ARKit's right-handed, Y-up coordinate system. The session origin is not guaranteed to match another recording's origin.
- One frame is generated per hand update; left/right records interleave. Head pose is queried at that update's timestamp. This is event-driven sampling, not a promised fixed rate. No pose is substituted when a head anchor is unavailable. Consumers must honor tracking flags, including each joint's flag.
- Footer: frame count, completion flag, stop reason, duration and tracked-hand counts. Interrupted or incomplete files remain available for recovery but are not accepted as completed datasets.

Capture streams to disk and synchronizes periodically. Memory use does not grow with recording duration. The importer validates every record and preserves the source bytes. Its 512 MB limit bounds upload/validation cost. Imports are content-addressed and repeated imports are idempotent. A changed file cannot replace an existing session identity.

## Storage and future providers

The server saves files under its database directory, `local-captures/`, and registers them with format `visionpro_tracking_jsonl_v1` and status `LOCAL`. They are not automatically available on the training cluster. Export the original for storage elsewhere; robot training requires an explicitly implemented retargeting/conversion pipeline and transfer to accessible storage.

`skynet_app/local_capture.py` defines the `CaptureProvider` protocol and provider registry. Providers inspect native files; shared storage, identity, import and export logic remains independent of their device. Cluster collection adapters remain the separate execution path for simulators such as DexVerse.

DexVerse's Isaac Sim/CloudXR workflow requires an NVIDIA GPU host. Its raw robot-action pickle format is distinct from local headset tracking. The pinned CloudXR sample client requires a newer visionOS release than this local app. Do not relabel one format as the other.

## Build verification

`xcodebuild -project visionpro/SkynetCapture.xcodeproj -scheme SkynetCapture -sdk xros -destination 'generic/platform=visionOS' -derivedDataPath /tmp/skynet-vision-build CODE_SIGNING_ALLOWED=NO build`

This verifies compilation for physical hardware. It does not provision or install the app. Real headset recording, permissions, sharing and re-import must be tested separately.
