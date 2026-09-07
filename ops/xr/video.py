"""Bounded MP4 encoding and a fixed camera for headless scene replay."""

import hashlib
from pathlib import Path
import queue
import threading
import uuid

import numpy as np

FPS = 30
RESOLUTION = (960, 540)
MAX_VIDEO_BYTES = 256 * 1024 * 1024


class VideoEncoder:
    def __init__(self, path):
        self.path = Path(path)
        self.partial = self.path.with_suffix("." + uuid.uuid4().hex + ".partial.mp4")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError("Refusing to replace a saved video")
        self.frames = 0
        self.error = None
        self.queue = queue.Queue(maxsize=8)
        self.thread = threading.Thread(target=self._encode, daemon=True)
        self.thread.start()

    def _encode(self):
        writer = None
        try:
            import imageio_ffmpeg

            writer = imageio_ffmpeg.write_frames(
                str(self.partial),
                RESOLUTION,
                fps=FPS,
                codec="libx264",
                pix_fmt_in="rgb24",
                pix_fmt_out="yuv420p",
                macro_block_size=1,
                output_params=[
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-threads",
                    "2",
                    "-movflags",
                    "+faststart",
                ],
                ffmpeg_timeout=20,
            )
            writer.send(None)
            while True:
                frame = self.queue.get()
                if frame is None:
                    break
                writer.send(frame)
                if (
                    self.partial.exists()
                    and self.partial.stat().st_size > MAX_VIDEO_BYTES
                ):
                    raise ValueError("Video exceeds the 256 MB limit")
        except Exception as exc:
            self.error = str(exc)
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception as exc:
                    self.error = str(exc)

    def append(self, frame):
        if self.error:
            raise RuntimeError(self.error)
        if frame.shape != (RESOLUTION[1], RESOLUTION[0], 3) or frame.dtype != np.uint8:
            raise ValueError("Scene camera returned an invalid video frame")
        try:
            self.queue.put_nowait(np.ascontiguousarray(frame).copy())
        except queue.Full as exc:
            raise RuntimeError("Video encoding cannot keep up with collection") from exc
        self.frames += 1

    def finish(self, *, discard=False):
        if self.thread.is_alive():
            try:
                self.queue.put(None, timeout=20)
            except queue.Full:
                self.error = "Video encoder stopped responding"
            self.thread.join(timeout=25)
        if self.thread.is_alive():
            self.error = "Video encoder did not finish within 25 seconds"
        if discard or self.error or not self.frames:
            self.partial.unlink(missing_ok=True)
            if self.error and not discard:
                raise RuntimeError(self.error)
            return None
        if (
            not self.partial.is_file()
            or not 0 < self.partial.stat().st_size <= MAX_VIDEO_BYTES
        ):
            raise ValueError("Video encoder produced an empty or oversized file")
        self.partial.replace(self.path)
        return dict(
            path=str(self.path),
            sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            size_bytes=self.path.stat().st_size,
            frames=self.frames,
            fps=FPS,
            width=RESOLUTION[0],
            height=RESOLUTION[1],
            viewpoint="scene",
        )


class SceneCamera:
    def __init__(self, env):
        import carb
        import omni.replicator.core as rep

        if carb.settings.get_settings().get("/app/xr/enabled"):
            raise RuntimeError(
                "Direct camera capture is unsupported in this XR runtime. Use scene replay after collection."
            )
        self.env = env
        self.camera = rep.create.camera(
            name="SkynetReviewCamera",
            position=(1.15, -1.2, 1.55),
            look_at=(-0.05, 0.0, 0.70),
            focal_length=32,
            clipping_range=(0.01, 100),
        )
        self.product = rep.create.render_product(self.camera, RESOLUTION)
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb.attach(self.product)
        # Warm shader and readback pipelines before the first recorded frame.
        for _ in range(10):
            env.sim.render()

    def frame(self):
        # Two render-only updates flush the asynchronous RGB readback without
        # advancing physics or changing the saved scene state.
        self.env.sim.render()
        self.env.sim.render()
        frame = self.rgb.get_data()
        if not isinstance(frame, np.ndarray) or frame.shape != (
            RESOLUTION[1],
            RESOLUTION[0],
            4,
        ):
            raise RuntimeError(
                f"The simulation camera did not produce a video frame (shape {getattr(frame, 'shape', None)})"
            )
        return frame[:, :, :3].copy()

    def close(self):
        self.rgb.detach(self.product)
        self.product.destroy()
