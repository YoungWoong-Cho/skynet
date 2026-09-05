import SwiftUI
import RealityKit
import UIKit

@main
struct SkynetCaptureApp: App {
    @State private var recorder = Recorder()
    var body: some SwiftUI.Scene {
        WindowGroup {
            CaptureView(recorder: recorder)
        }.defaultSize(width: 650, height: 700)
        ImmersiveSpace(id: "tracking") {
            RealityView { _ in }
                .task { recorder.trackingSpaceOpen = true; await recorder.openTracking() }
                .onDisappear {
                    recorder.trackingSpaceOpen = false
                    recorder.closeTracking(preservingMessage: true)
                }
        }.immersionStyle(selection: .constant(.mixed), in: .mixed)
    }
}

private struct SharedRecording: Identifiable {
    let url: URL
    var id: URL { url }
}

private struct RecordingShareSheet: UIViewControllerRepresentable {
    let url: URL
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: [url], applicationActivities: nil)
    }
    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}

struct CaptureView: View {
    @Bindable var recorder: Recorder
    @Environment(\.openImmersiveSpace) private var openSpace
    @Environment(\.dismissImmersiveSpace) private var dismissSpace
    @State private var preparing = false
    @State private var finishing = false
    @State private var preparation: Task<Void, Never>?
    @State private var sharedRecording: SharedRecording?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text("Skynet Capture").font(.largeTitle)
                Text("Record hand joints and head movement on this Vision Pro. No GPU computer or network is needed while recording.")
                Text("Tracking only: no camera images, object states, or robot actions. Keep the headset’s view of your hands clear.")
                    .font(.callout).foregroundStyle(.secondary)
                TextField("Task or demonstration name", text: $recorder.taskName).disabled(preparing || recorder.recording)
                TextField("Operator label (optional)", text: $recorder.operatorName).disabled(preparing || recorder.recording)
                captureControls
                if preparing || recorder.recording {
                    Label(recorder.trackedHands, systemImage: "hand.raised")
                    Label(recorder.headTracked ? "Head: tracked" : "Head: not tracked", systemImage: "visionpro")
                }
                Text(recorder.message).accessibilityAddTraits(.updatesFrequently)
                Divider()
                savedRecordings
            }.padding(32).textFieldStyle(.roundedBorder)
        }
        .sheet(item: $sharedRecording) { RecordingShareSheet(url: $0.url) }
        .onChange(of: recorder.recording) { wasRecording, isRecording in
            if wasRecording && !isRecording && !finishing {
                finishing = true
                Task { await endTracking(); finishing = false }
            }
        }
    }

    private var captureControls: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                if recorder.recording {
                    Button("Stop and save") {
                        finishing = true
                        recorder.stopRecording()
                        Task { await endTracking(); finishing = false }
                    }.buttonStyle(.borderedProminent)
                } else {
                    Button(preparing ? "Preparing tracking…" : finishing ? "Finishing…" : "Start recording") {
                        beginRecording()
                    }.buttonStyle(.borderedProminent)
                        .disabled(preparing || finishing || recorder.taskName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
                if preparing {
                    Button("Cancel") { preparation?.cancel() }
                }
                if recorder.recording { Text("\(recorder.frameCount) frames").monospacedDigit() }
            }
            Text("Start sets up tracking automatically. Recording begins when a hand and your head are tracked; Stop and save finishes tracking so you can share.")
                .font(.callout).foregroundStyle(.secondary)
        }
    }

    private var savedRecordings: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Saved on this headset").font(.title2)
            Text("Choose Share, then AirDrop or Save to Files. In Skynet on your Mac, open Data → Collection → Import recording. The original stays here.").font(.callout)
            if recorder.recordings.isEmpty { Text("No recordings yet.").foregroundStyle(.secondary) }
            ForEach(recorder.recordings, id: \.self) { url in
                HStack {
                    Text(recorder.savedRecordingLabels[url] ?? url.lastPathComponent)
                        .font(.callout).lineLimit(2)
                    Spacer()
                    Button {
                        finishing = true
                        Task {
                            await endTracking()
                            finishing = false
                            sharedRecording = SharedRecording(url: url)
                        }
                    } label: { Label("Share", systemImage: "square.and.arrow.up") }
                    .disabled(preparing || finishing || recorder.recording)
                }
            }
        }
    }

    @MainActor private func endTracking() async {
        recorder.closeTracking(preservingMessage: true)
        if recorder.trackingSpaceOpen {
            await dismissSpace()
            recorder.trackingSpaceOpen = false
        }
    }

    @MainActor private func beginRecording() {
        preparing = true
        preparation = Task { @MainActor in
            defer { preparing = false; preparation = nil }
            await endTracking()
            recorder.trackingError = nil
            recorder.message = "Preparing tracking. Allow hand tracking if asked, and keep your hands visible."
            switch await openSpace(id: "tracking") {
            case .opened: break
            case .userCancelled:
                recorder.message = "Cancelled. No recording started."
                return
            case .error:
                recorder.message = "Tracking could not open. Close other immersive apps and try Start recording again."
                return
            @unknown default:
                recorder.message = "Tracking returned an unsupported result. No recording started."
                return
            }
            for _ in 0..<300 {
                if Task.isCancelled {
                    await endTracking()
                    recorder.message = "Cancelled. No recording started."
                    return
                }
                if let error = recorder.trackingError {
                    await endTracking()
                    recorder.message = error
                    return
                }
                if recorder.ready {
                    recorder.startRecording()
                    if !recorder.recording { await endTracking() }
                    return
                }
                try? await Task.sleep(for: .milliseconds(100))
            }
            await endTracking()
            recorder.message = "Tracking was not ready within 30 seconds. Keep your hands visible in a well-lit space, close system panels, and try Start recording again."
        }
    }
}
