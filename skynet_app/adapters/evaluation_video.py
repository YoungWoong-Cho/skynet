"""Compose policy camera observations into one labeled evaluation video frame."""

import math

import cv2
import numpy as np


def rgb_frame(value, *, channel_first=False):
    """Convert one raw camera observation without changing its source buffer."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    while value.ndim > 3:
        value = value[0]
    if channel_first:
        value = np.moveaxis(value, 0, -1)
    if value.ndim != 3 or value.shape[-1] not in (3, 4):
        raise ValueError(f"Expected an RGB camera image, got {value.shape}")
    value = value[..., :3]
    if np.issubdtype(value.dtype, np.floating):
        value = np.rint(np.clip(value, 0, 1) * 255)
    return np.array(value, dtype=np.uint8, order="C", copy=True)


def camera_label(key):
    name = key.rsplit(".", 1)[-1]
    return {
        "scene_front": "Front", "scene_left": "Left", "scene_right": "Right",
        "front_img_1": "Front", "left_wrist_img": "Left wrist",
        "right_wrist_img": "Right wrist", "agentview_image": "Front",
        "robot0_eye_in_hand_image": "Wrist",
    }.get(name, name.replace("_", " ").capitalize())


def compose_camera_views(views, *, channel_first=False, captions=()):
    """Show every supplied view in order, retaining aspect ratios and RGB colors.

    Up to three views share a row. Larger camera sets wrap into additional rows.
    Padding, rather than resizing, preserves differently sized camera images.
    """
    if not views:
        raise ValueError("Evaluation video requires at least one camera view")
    frames = [(camera_label(key), rgb_frame(value, channel_first=channel_first))
              for key, value in views.items()]
    columns = min(3, len(frames))
    rows = math.ceil(len(frames) / columns)
    width = max(frame.shape[1] for _, frame in frames)
    height = max(frame.shape[0] for _, frame in frames)
    label_height = 32
    caption_height = 24 * len(captions)
    # Keep encoder-friendly dimensions without letting ffmpeg resize the frames.
    canvas_width = math.ceil(columns * width / 16) * 16
    canvas_height = math.ceil((rows * (height + label_height) + caption_height) / 16) * 16
    canvas = np.full((canvas_height, canvas_width, 3), 24, dtype=np.uint8)
    for index, (label, frame) in enumerate(frames):
        x = (index % columns) * width
        y = (index // columns) * (height + label_height)
        cv2.putText(canvas, label, (x + 8, y + 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (240, 240, 240), 1, cv2.LINE_AA)
        left = x + (width - frame.shape[1]) // 2
        top = y + label_height + (height - frame.shape[0]) // 2
        canvas[top:top + frame.shape[0], left:left + frame.shape[1]] = frame
    for index, caption in enumerate(captions):
        y = rows * (height + label_height) + 18 + 24 * index
        cv2.putText(canvas, caption, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas
