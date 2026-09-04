import SwiftUI
import RealityKit

@main
struct SkynetCaptureApp: App {
    @State private var recorder = Recorder()
    var body: some SwiftUI.Scene {
        WindowGroup {
            CaptureView(recorder: recorder)
        }.defaultSize(width: 650, height: 700)
        ImmersiveSpace(id: "tracking") {
            RealityView { _ in }
                .task { await recorder.openTracking() }
                .onDisappear { recorder.closeTracking() }
        }.immersionStyle(selection: .constant(.mixed), in: .mixed)
    }
}

struct CaptureView: View {
    @Bindable var recorder: Recorder
    @Environment(\.openImmersiveSpace) private var openSpace
    @Environment(\.dismissImmersiveSpace) private var dismissSpace
    @State private var spaceOpen = false
    @State private var opening = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text("Skynet Capture").font(.largeTitle)
                Text("Record hand joints and head movement on this Vision Pro. No GPU computer or network is needed while recording.")
                Text("This records tracking only, without camera images, object states, or robot actions. Keep the headset’s view of your hands clear.").font(.callout).foregroundStyle(.secondary)
                TextField("Task or demonstration name", text: $recorder.taskName).disabled(recorder.recording)
                TextField("Operator label (optional)", text: $recorder.operatorName).disabled(recorder.recording)
                HStack {
                    Button(opening ? "Opening…" : "Open tracking space") {
                        opening = true
                        Task {
                            if spaceOpen { await dismissSpace(); spaceOpen = false }
                            switch await openSpace(id: "tracking") {
                            case .opened: spaceOpen = true
                            case .userCancelled: recorder.message = "Opening was cancelled. No recording started."
                            case .error: recorder.message = "Tracking space could not open. Close other immersive apps and try again."
                            @unknown default: recorder.message = "Tracking space returned an unsupported result."
                            }
                            opening = false
                        }
                    }.disabled(opening || recorder.recording)
                    Button("Close tracking") {
                        recorder.closeTracking()
                        Task { await dismissSpace(); spaceOpen = false }
                    }.disabled(!spaceOpen)
                }
                Label(recorder.trackedHands, systemImage: "hand.raised")
                Text(recorder.message).accessibilityAddTraits(.updatesFrequently)
                HStack {
                    Button("Start recording") { recorder.startRecording() }
                        .disabled(!recorder.ready || recorder.recording || recorder.taskName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    Button("Stop and save") { recorder.stopRecording() }.disabled(!recorder.recording)
                    Text("\(recorder.frameCount) frames").monospacedDigit()
                }
                Divider()
                Text("Saved on this headset").font(.title2)
                Text("Share a recording to your Mac using AirDrop or Files. In Skynet, open Data → Collection → Import recording. Sharing keeps the original here.").font(.callout)
                if recorder.recordings.isEmpty { Text("No recordings yet.").foregroundStyle(.secondary) }
                ForEach(recorder.recordings, id: \.self) { url in
                    ShareLink(item: url) { Label(url.lastPathComponent, systemImage: "square.and.arrow.up") }
                }
            }.padding(32).textFieldStyle(.roundedBorder)
        }
    }
}
