from io import BytesIO
import urllib.parse
import urllib.request

import numpy as np

from signlang.topology import FRAME_DIM, LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET


SIGNMT_ENDPOINT = "https://us-central1-sign-mt.cloudfunctions.net/spoken_text_to_signed_pose"

POSE_POINT_TO_MEDIAPIPE_INDEX = {
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
}

FACE_TO_MEDIAPIPE_INDEX = {
    "33": 2,   # right eye
    "263": 5,  # left eye
    "61": 9,   # mouth right-ish
    "291": 10, # mouth left-ish
}


def fetch_signmt_pose_bytes(text, spoken_language="en", signed_language="ase", timeout=45):
    params = urllib.parse.urlencode(
        {
            "text": text,
            "spoken": spoken_language,
            "signed": signed_language,
        }
    )
    url = f"{SIGNMT_ENDPOINT}?{params}"
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("content-type", "")
        payload = response.read()
    if "application/pose" not in content_type.lower():
        raise RuntimeError(f"Unexpected sign.mt response type: {content_type or 'unknown'}")
    if not payload:
        raise RuntimeError("Empty sign.mt response.")
    return payload


def signmt_pose_to_frames(pose_bytes):
    try:
        from pose_format import Pose
    except Exception as exc:
        raise RuntimeError("Missing dependency 'pose-format'. Install with: pip install pose-format") from exc

    pose = Pose.read(BytesIO(pose_bytes))
    width = float(getattr(pose.header.dimensions, "width", 0) or 0)
    height = float(getattr(pose.header.dimensions, "height", 0) or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid dimensions in sign.mt pose response.")

    data = pose.body.data.filled(np.nan)
    if data.ndim != 4 or data.shape[1] < 1:
        raise RuntimeError("Unexpected sign.mt pose tensor shape.")
    sequence = data[:, 0, :, :]

    components = _split_components(pose, sequence)
    frames = np.zeros((sequence.shape[0], FRAME_DIM), dtype=np.float32)

    pose_xy = components.get("POSE_LANDMARKS")
    if pose_xy is not None:
        _map_body_points(frames, pose_xy, pose.header.components, width, height)
    hand_left_xy = components.get("LEFT_HAND_LANDMARKS")
    if hand_left_xy is not None:
        _map_hand_points(frames, hand_left_xy, LEFT_HAND_OFFSET, width, height)
    hand_right_xy = components.get("RIGHT_HAND_LANDMARKS")
    if hand_right_xy is not None:
        _map_hand_points(frames, hand_right_xy, RIGHT_HAND_OFFSET, width, height)
    face_xy = components.get("FACE_LANDMARKS")
    if face_xy is not None:
        _map_face_points(frames, face_xy, pose.header.components, width, height)

    _infer_missing_core_points(frames)
    frames = np.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return frames, float(getattr(pose.body, "fps", 25.0) or 25.0)


def _split_components(pose, sequence):
    parts = {}
    offset = 0
    for component in pose.header.components:
        length = len(component.points)
        parts[component.name] = sequence[:, offset : offset + length, :2]
        offset += length
    return parts


def _map_body_points(frames, pose_xy, components, width, height):
    pose_component = next(component for component in components if component.name == "POSE_LANDMARKS")
    points = pose_component.points
    for point_index, point_name in enumerate(points):
        body_index = POSE_POINT_TO_MEDIAPIPE_INDEX.get(point_name)
        if body_index is None:
            continue
        coords = pose_xy[:, point_index, :]
        _set_joint(frames, body_index, coords[:, 0] / width, coords[:, 1] / height)


def _map_hand_points(frames, hand_xy, hand_offset, width, height):
    for hand_index in range(min(21, hand_xy.shape[1])):
        coords = hand_xy[:, hand_index, :]
        _set_joint(frames, hand_offset + hand_index, coords[:, 0] / width, coords[:, 1] / height)


def _map_face_points(frames, face_xy, components, width, height):
    face_component = next(component for component in components if component.name == "FACE_LANDMARKS")
    name_to_index = {str(name): idx for idx, name in enumerate(face_component.points)}
    for face_name, body_index in FACE_TO_MEDIAPIPE_INDEX.items():
        face_index = name_to_index.get(face_name)
        if face_index is None:
            continue
        coords = face_xy[:, face_index, :]
        _set_joint(frames, body_index, coords[:, 0] / width, coords[:, 1] / height)


def _set_joint(frames, joint_index, x_values, y_values):
    base = joint_index * 2
    frames[:, base] = x_values
    frames[:, base + 1] = y_values


def _infer_missing_core_points(frames):
    left_shoulder = _joint_xy(frames, 11)
    right_shoulder = _joint_xy(frames, 12)
    left_hip = _joint_xy(frames, 23)
    right_hip = _joint_xy(frames, 24)
    nose = _joint_xy(frames, 0)

    shoulders_valid = _joint_valid(left_shoulder) & _joint_valid(right_shoulder)
    if np.any(shoulders_valid):
        mid_shoulder = (left_shoulder + right_shoulder) / 2.0
        shoulder_span = np.linalg.norm(left_shoulder - right_shoulder, axis=1)
        est_nose = mid_shoulder.copy()
        est_nose[:, 1] -= np.maximum(shoulder_span * 0.65, 0.08)
        _set_when_invalid(frames, 0, est_nose, shoulders_valid)
        _set_when_invalid(frames, 1, mid_shoulder + np.array([-0.04, -0.02], dtype=np.float32), shoulders_valid)
        _set_when_invalid(frames, 4, mid_shoulder + np.array([0.04, -0.02], dtype=np.float32), shoulders_valid)
        _set_when_invalid(frames, 7, mid_shoulder + np.array([-0.07, 0.0], dtype=np.float32), shoulders_valid)
        _set_when_invalid(frames, 8, mid_shoulder + np.array([0.07, 0.0], dtype=np.float32), shoulders_valid)

    hips_valid = _joint_valid(left_hip) & _joint_valid(right_hip)
    if np.any(hips_valid):
        mid_hip = (left_hip + right_hip) / 2.0
        _set_when_invalid(frames, 25, left_hip + np.array([0.0, 0.16], dtype=np.float32), hips_valid)
        _set_when_invalid(frames, 26, right_hip + np.array([0.0, 0.16], dtype=np.float32), hips_valid)
        _set_when_invalid(frames, 27, left_hip + np.array([0.0, 0.30], dtype=np.float32), hips_valid)
        _set_when_invalid(frames, 28, right_hip + np.array([0.0, 0.30], dtype=np.float32), hips_valid)
        _set_when_invalid(frames, 29, left_hip + np.array([-0.03, 0.33], dtype=np.float32), hips_valid)
        _set_when_invalid(frames, 30, right_hip + np.array([0.03, 0.33], dtype=np.float32), hips_valid)
        _set_when_invalid(frames, 31, left_hip + np.array([0.03, 0.34], dtype=np.float32), hips_valid)
        _set_when_invalid(frames, 32, right_hip + np.array([-0.03, 0.34], dtype=np.float32), hips_valid)
        _set_when_invalid(frames, 9, mid_hip + np.array([-0.05, -0.20], dtype=np.float32), hips_valid)
        _set_when_invalid(frames, 10, mid_hip + np.array([0.05, -0.20], dtype=np.float32), hips_valid)

    _clamp_frames(frames)


def _joint_xy(frames, joint_index):
    base = joint_index * 2
    return frames[:, base : base + 2]


def _joint_valid(joint_xy):
    return np.isfinite(joint_xy).all(axis=1) & (joint_xy[:, 0] > 0) & (joint_xy[:, 1] > 0)


def _set_when_invalid(frames, joint_index, replacement_xy, condition):
    existing = _joint_xy(frames, joint_index)
    valid = _joint_valid(existing)
    update_mask = condition & (~valid)
    if not np.any(update_mask):
        return
    base = joint_index * 2
    frames[update_mask, base] = replacement_xy[update_mask, 0]
    frames[update_mask, base + 1] = replacement_xy[update_mask, 1]


def _clamp_frames(frames):
    np.clip(frames, 0.0, 1.0, out=frames)
