# Episode viewer

Recordings and evaluation rollouts reuse the existing application dialog and the same viewer. Camera tabs crop the saved observations used by training or inference; interactive mode shows the recorded joint skeleton in 3D. It does not replay physics or reconstruct unrecorded scene meshes. The sidebar reads the Hands catalog and identifies legacy DexVerse recordings separately.

Rollout layers are independent:

- **Prediction**: the model command converted to joint targets with the frozen action order, scale and offset.
- **Actual**: the simulator hand observed before applying that command.
- **Demonstration**: the original recording, only when the evaluation explicitly pins one checksum-verified source. It is aligned by elapsed simulation time and ends with that recording. This is a reference trajectory, not ground truth for a diverged rollout.

Camera overlays use the decoded video frame timestamp, rather than the potentially newer playback clock. New recorded-simulator evaluations save a `.review.json` alongside each video, at the video sample rate. Existing rollouts without that trace retain video playback and show unavailable layers as disabled; missing predictions are never synthesized. Other evaluator implementations require the same trace schema to expose interactive layers.

Collection previews are CPU-generated next to the cluster review artifacts from checksum-verified image sidecars and original scene states. They contain a camera video and lightweight replay JSON. New image sidecars also embed the checksum of the kinematic tree used by that capture. The database does not contain videos or per-frame points; browser requests stream these artifacts without a local payload cache.

Validation: CPU geometry tests cover named joint order, root transforms, command conversion, reference expiry and camera layout. UI regression tests cover decoded-frame synchronization, independent toggles, stale responses and callback cleanup. Real collection playback was checked in the browser; the new rollout trace writer has not been exercised by a new GPU evaluation in this change.
