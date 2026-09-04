import ARKit
import Foundation
import Observation
import QuartzCore
import simd

/// Native tracking is preserved; robot retargeting is a separate, versioned operation.
@MainActor @Observable
final class Recorder {
    var taskName = "Hand movement test"
    var operatorName = ""
    var message = "Open tracking space to check hand-tracking permission."
    var ready = false
    var recording = false
    var frameCount = 0
    var trackedHands = "Waiting for hands"
    var recordings: [URL] = []
    private var session: ARKitSession?
    private var hands: HandTrackingProvider?
    private var world: WorldTrackingProvider?
    private var updatesTask: Task<Void, Never>?
    private var eventsTask: Task<Void, Never>?
    private var file: FileHandle?
    private var visibleHands: Set<String> = []
    private var lastTime: Double = 0
    private var startTime: Double = 0
    private var trackedCounts = ["left": 0, "right": 0]

    var folder: URL {
        URL.documentsDirectory.appending(path: "Skynet Recordings", directoryHint: .isDirectory)
    }

    init() { reloadRecordings() }

    func reloadRecordings() {
        do {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            recordings = try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)
                .filter { $0.pathExtension == "jsonl" }.sorted { $0.lastPathComponent > $1.lastPathComponent }
        } catch { message = "Cannot read saved recordings: \(error.localizedDescription)" }
    }

    func openTracking() async {
        guard session == nil else { return }
        guard HandTrackingProvider.isSupported && WorldTrackingProvider.isSupported else {
            message = "Unsupported here. Hand and head tracking require a physical Apple Vision Pro; the simulator cannot record them."
            return
        }
        let next = ARKitSession()
        let authorization = await next.requestAuthorization(for: [.handTracking])
        guard authorization[.handTracking] == .allowed else {
            message = "Hand tracking permission is denied. Enable it for Skynet Capture in Settings → Privacy & Security, then reopen tracking."
            return
        }
        let handProvider = HandTrackingProvider()
        let worldProvider = WorldTrackingProvider()
        do {
            try await next.run([handProvider, worldProvider])
            session = next; hands = handProvider; world = worldProvider
            message = "Waiting for tracked hands. Keep both hands visible."
            updatesTask = Task { [weak self] in
                for await update in handProvider.anchorUpdates {
                    guard !Task.isCancelled, let self else { break }
                    self.receive(update.anchor, timestamp: update.timestamp, removed: update.event == .removed)
                }
            }
            eventsTask = Task { [weak self] in
                for await event in next.events {
                    guard !Task.isCancelled, let self else { break }
                    switch event {
                    case .authorizationChanged(let type, let status):
                        if type == .handTracking && status != .allowed {
                            self.closeTracking(reason: "Hand tracking permission was revoked.")
                        }
                    case .dataProviderStateChanged(_, let state, let error):
                        if state == .paused || state == .stopped || error != nil {
                            self.closeTracking(reason: "Tracking interrupted (\(state)): \(error?.localizedDescription ?? "Space closed or app inactive").")
                        }
                    default: break
                    }
                }
            }
        } catch {
            next.stop()
            message = "Tracking could not start: \(error.localizedDescription)"
        }
    }

    func startRecording() {
        guard ready, !recording else { return }
        let id = UUID().uuidString.lowercased()
        let url = folder.appending(path: "\(ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-"))-\(id).jsonl")
        do {
            guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
                throw CocoaError(.fileWriteUnknown)
            }
            file = try FileHandle(forWritingTo: url)
            startTime = CACurrentMediaTime(); lastTime = 0; frameCount = 0
            trackedCounts = ["left": 0, "right": 0]
            try write([
                "type": "header", "schema": "skynet.visionpro-tracking/v1", "session_id": id,
                "created_at": ISO8601DateFormatter().string(from: Date()),
                "task": taskName, "operator": operatorName, "device": "Apple Vision Pro",
                "app_version": "1.0", "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
                "clock": "arkit-monotonic-seconds", "units": "meters", "matrix_order": "column-major",
                "coordinate_frame": "ARKit session origin; right-handed; Y up",
                "joint_frame": "hand-anchor", "head_frame": "ARKit session origin",
                "sampling": "one record per hand anchor update; head queried at the same timestamp",
                "streams": ["hand_joints", "hand_origin", "head_pose"],
                "robot_actions": false, "images_recorded": false
            ])
            try file?.synchronize()
            recording = true; message = "Recording locally. Stop and save when finished."
        } catch { failWrite(error) }
    }

    private func matrix(_ value: simd_float4x4) -> [Float] {
        (0..<4).flatMap { column in (0..<4).map { row in value[column][row] } }
    }

    private func receive(_ anchor: HandAnchor, timestamp: Double, removed: Bool) {
        let side = anchor.chirality == .left ? "left" : "right"
        let tracked = anchor.isTracked && !removed
        let wasReady = ready
        if tracked { visibleHands.insert(side) } else { visibleHands.remove(side) }
        ready = !visibleHands.isEmpty
        if ready != wasReady && !recording {
            message = ready ? "Tracking is ready. Start recording when you are ready."
                : "No hands are tracked. Keep your hands visible before starting a recording."
        }
        trackedHands = ["left", "right"].map { "\($0.capitalized): \(visibleHands.contains($0) ? "tracked" : "not tracked")" }.joined(separator: " · ")
        guard recording else { return }
        // Per-anchor timestamps may interleave across hands. Preserve the source time and use
        // a monotonic receive clock for ordering; never replace a missing pose with an identity.
        let received = CACurrentMediaTime()
        let head = world?.queryDeviceAnchor(atTimestamp: timestamp)
        let joints: [[String: Any]] = (anchor.handSkeleton?.allJoints ?? []).map {
            ["name": String(describing: $0.name), "tracked": $0.isTracked,
             "anchor_from_joint": matrix($0.anchorFromJointTransform)]
        }
        do {
            try write([
                "type": "frame", "index": frameCount, "timestamp": received,
                "source_timestamp": timestamp, "hand": side, "tracked": tracked,
                "origin_from_hand": matrix(anchor.originFromAnchorTransform), "joints": joints,
                "head_tracked": head?.isTracked ?? false,
                "origin_from_head": head.map { matrix($0.originFromAnchorTransform) } as Any? ?? NSNull()
            ])
            frameCount += 1; lastTime = received
            if tracked { trackedCounts[side, default: 0] += 1 }
            if frameCount % 120 == 0 { try file?.synchronize() }
        } catch { failWrite(error) }
    }

    func stopRecording(reason: String = "operator", completed: Bool = true) {
        guard file != nil else { return }
        do {
            try write(["type": "footer", "frames": frameCount,
                       "duration_seconds": max(0, lastTime - startTime),
                       "completed": completed, "stop_reason": reason,
                       "tracked_frames": trackedCounts])
            try file?.synchronize(); try file?.close()
            file = nil; recording = false
            message = frameCount > 0 ? "Saved \(frameCount) frames locally. Share the JSONL file to your Mac and import it in Skynet → Data → Collection."
                : "Saved an empty recording. Keep your hands visible and record again; empty captures cannot be imported."
        } catch { failWrite(error) }
        reloadRecordings()
    }

    func closeTracking(reason: String = "Tracking space closed.") {
        if recording { stopRecording(reason: reason, completed: false) }
        ready = false
        visibleHands.removeAll()
        trackedHands = "Tracking closed"
        updatesTask?.cancel(); eventsTask?.cancel()
        updatesTask = nil; eventsTask = nil
        let old = session; session = nil; hands = nil; world = nil; old?.stop()
        message = reason + " Reopen tracking space to continue. Saved files remain on the headset."
    }

    private func write(_ object: [String: Any]) throws {
        guard let file else { throw CocoaError(.fileWriteUnknown) }
        var bytes = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        bytes.append(10)
        try file.write(contentsOf: bytes)
    }

    private func failWrite(_ error: Error) {
        try? file?.close(); file = nil; recording = false
        message = "Recording stopped because saving failed: \(error.localizedDescription). Any partial file is retained for recovery; it is not a completed dataset."
        reloadRecordings()
    }
}
