# Skynet Capture (visionOS 2.0+)

A standalone native offline recorder for Apple Vision Pro. The former Skynet backend importer and offline experiment APIs have been retired; this app exports native files through the system share control. No NVIDIA GPU or CloudXR connection is needed. Hand joints and device poses come from ARKit; camera images, object poses, task success and robot actions are not synthesized.

## Install and test

1. Open `SkynetCapture.xcodeproj` in Xcode on a Mac with the visionOS SDK.
2. Select the app target, Signing & Capabilities, your Apple development team, and a unique bundle identifier. Keep automatic signing enabled.
3. Wear and unlock the physical Vision Pro. Connect it and the Mac to the same private Wi-Fi network. Leave Settings → General → Remote Devices open on the headset, then select the headset in Xcode's Devices and Simulators window and click Pair. Enter the current code shown on the headset. If neither device appears, try another private Wi-Fi network; some networks prevent local device discovery.
4. Enable Settings → Privacy & Security → Developer Mode on the headset after pairing. Complete any restart and confirmation, select the headset as Xcode's run destination, and run the app. Developer provisioning may require Apple account authentication and headset confirmation. The first connection can take several minutes while Xcode prepares the headset; keep it worn, unlocked, and on the same network. If launch is blocked by an untrusted developer certificate, open Settings → General → VPN & Device Management on the headset, select your Developer App certificate, and trust it, then run again from Xcode.
5. Enter a task name and choose Start recording. Allow hand tracking and keep both hands visible. Tracking opens automatically and recording begins when a hand and the head are tracked. Cancel is available while preparing; failures or a 30-second readiness timeout show a specific message.
6. Record approximately 10 seconds, move both hands and your head, then Stop and save. Tracking closes automatically. Confirm the saved frame and tracked-head counts are nonzero.
7. Choose Share beside a saved recording to open AirDrop / Save to Files. Share becomes available after tracking closes and uses the native system share control with a supplied preview. Preserve the exported JSONL file as the original recording; Skynet no longer exposes the old import UI or API.
8. Interrupt tracking during a second recording: the saved file must be labeled interrupted. Consumers must not treat it as a completed demonstration.

Temporary tracking pauses while preparing can recover when the app becomes active. A pause during recording finalizes that file as interrupted and ends tracking; choose Start recording for a new capture. The UI handles the tracking-space lifecycle automatically.

The visionOS simulator has no real hand-tracking provider. Its unsupported message is expected and does not verify headset capture. Recordings remain in the app's Documents/Skynet Recordings folder; sharing does not delete them. Uninstalling the app may remove its local data.

## Native schema

Each UTF-8 JSONL file has one `header`, sequential `frame` records, and one `footer`. Schema: `skynet.visionpro-tracking/v1`.

- Header: session UUID, task/operator labels, creation time, app/OS versions, stream declarations, frames and units.
- Frame: receive-clock timestamp and original ARKit timestamp (monotonic seconds), hand chirality and tracking flags, 4×4 origin-from-hand transform, named hand joints with anchor-from-joint transforms, and optional origin-from-head pose with tracking status. Matrices are column-major; translations are meters, in ARKit's right-handed, Y-up coordinate system. The session origin is not guaranteed to match another recording's origin.
- One frame is generated per hand update; left/right records interleave. Head pose is queried at the current `CACurrentMediaTime`, stored as `head_timestamp`. ARKit hand-update timestamps can have a different clock origin and must not be used as head-query times. Starting with recorder 1.1, the header names each timestamp clock explicitly; preserve these separate clocks when processing data. This is event-driven sampling, not a promised fixed rate. No pose is substituted when a head anchor is unavailable. Consumers must honor tracking flags, including each joint's flag.
- Footer: frame count, completion flag, stop reason, duration and tracked-hand counts. Interrupted or incomplete files remain available for recovery but are not accepted as completed datasets.

Capture streams to disk and synchronizes periodically. Memory use does not grow with recording duration. Consumers of exported files must validate every record, preserve the source bytes, and respect interrupted or incomplete recordings.

## Storage and backend support

Recordings remain in the native app's Documents/Skynet Recordings folder until explicitly removed. Export originals with the system share control. Skynet's former `/api/collection/local` and `/api/collection/processing` APIs and their provider implementation have been removed; existing server-side files and database records were not deleted by that code cleanup.

For live simulator collection, use the supported Isaac Sim/CloudXR workflow in Skynet. Its robot-action trajectories have a different schema from this recorder's headset tracking.

## Build verification

`xcodebuild -project visionpro/SkynetCapture.xcodeproj -scheme SkynetCapture -sdk xros -destination 'generic/platform=visionOS' -derivedDataPath /tmp/skynet-vision-build CODE_SIGNING_ALLOWED=NO build`

This verifies compilation for physical hardware. It does not provision or install the app. Real headset recording, permissions and sharing must be tested separately.
