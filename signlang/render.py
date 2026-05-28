from pathlib import Path
import json

import cv2
import numpy as np

from signlang.topology import FRAME_DIM, HOLISTIC_CONNECTIONS


def load_frames_from_json(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    frames = np.asarray(data["frames"], dtype=np.float32)
    if frames.ndim != 2 or frames.shape[1] != FRAME_DIM:
        raise ValueError(f"{path} must contain frames shaped [n, {FRAME_DIM}]")
    return frames


def smooth_keypoints(frames, window_length=5, polyorder=2):
    frames = np.asarray(frames, dtype=np.float32)
    if len(frames) < 3:
        return frames

    window = min(window_length, len(frames))
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        return frames

    try:
        from scipy.signal import savgol_filter
    except ImportError:
        return frames

    return savgol_filter(frames, window_length=window, polyorder=polyorder, axis=0).astype(np.float32)


def render_keypoints_video(
    frames,
    output_path,
    fps=20,
    canvas_size=512,
    smooth=True,
    point_radius=4,
):
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2 or frames.shape[1] != FRAME_DIM:
        raise ValueError(f"frames must have shape [n, {FRAME_DIM}]")
    if len(frames) == 0:
        raise ValueError("frames must contain at least one frame")
    if smooth:
        frames = smooth_keypoints(frames)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (canvas_size, canvas_size),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    written = 0
    try:
        for frame in frames:
            image = draw_skeleton_frame(frame, canvas_size=canvas_size, point_radius=point_radius)
            writer.write(image)
            written += 1
    finally:
        writer.release()

    if written == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"No frames were written to {output_path}")

    return output_path


def draw_skeleton_frame(frame, canvas_size=512, point_radius=4):
    points = np.asarray(frame, dtype=np.float32).reshape(-1, 2)
    image = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    pixel_points = np.clip(points, 0.0, 1.0) * (canvas_size - 1)

    for start, end in HOLISTIC_CONNECTIONS:
        start_point = pixel_points[start]
        end_point = pixel_points[end]
        if not (_is_visible(points[start]) and _is_visible(points[end])):
            continue
        cv2.line(
            image,
            tuple(start_point.astype(int)),
            tuple(end_point.astype(int)),
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

    for original_point, pixel_point in zip(points, pixel_points):
        if not _is_visible(original_point):
            continue
        cv2.circle(
            image,
            tuple(pixel_point.astype(int)),
            point_radius,
            (0, 0, 255),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    return image


def _is_visible(point):
    if not np.isfinite(point).all():
        return False
    return not np.allclose(point, 0.0)
