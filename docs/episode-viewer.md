# Episode viewer

Recordings and evaluation rollouts reuse the existing application dialog and the same viewer. Camera tabs crop the saved observations used by training or inference; interactive mode shows the recorded joint skeleton in 3D. It does not replay physics or reconstruct unrecorded scene meshes. The sidebar uses the Hands catalog and follows the displayed frame with finger joints only. Its palm, hardware wrist, root position and root orientation remain fixed; orbit controls remain interactive. Recorded joint names are mapped to the catalog palm subtree, including legacy Shadow names and canonical/bimanual names.

Rollout layers are independent:

- **Prediction**: the model command converted to joint targets with the frozen action order, scale and offset.
- **Actual**: the simulator hand observed before applying that command.
- **Demonstration**: the original recording, only when the evaluation explicitly pins one checksum-verified source. It is aligned by elapsed simulation time and ends with that recording. This is a reference trajectory, not ground truth for a diverged rollout.

Camera overlays and the sidebar finger pose use the same decoded video frame timestamp, rather than the potentially newer playback clock. Seeking and interactive playback use the requested timeline frame. For rollouts, the sidebar Pose selector defaults to Actual when recorded, otherwise explicitly selects the first available pose layer. A missing or expired pose restores the sidebar hand's default finger pose while keeping its wrist, position, orientation and orbit controls unchanged. The static preview is labeled Default pose; its Pose selector is hidden when no joint playback is available. Keypoints alone are not treated as joint angles. New recorded-simulator evaluations save a `.review.json` alongside each video, at the video sample rate. Existing rollouts without that trace retain video playback and show unavailable layers as disabled; missing predictions are never synthesized. Other evaluator implementations require the same trace schema to expose interactive layers.

Collection previews are CPU-generated next to the cluster review artifacts from checksum-verified image sidecars and original scene states. They contain a camera video and lightweight replay JSON. New image sidecars also embed the checksum of the kinematic tree used by that capture. The database does not contain videos or per-frame points; browser requests stream these artifacts without a local payload cache.

Validation: CPU geometry tests cover named joint order, root transforms, command conversion, reference expiry and camera layout. UI regression tests cover decoded-frame synchronization, independent toggles, stale responses and callback cleanup. Real collection playback was checked in the browser; the new rollout trace writer has not been exercised by a new GPU evaluation in this change.
