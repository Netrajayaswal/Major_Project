import math
import re

import numpy as np

from signlang.topology import FRAME_DIM, LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET


LETTER_SHAPES = {
    "A": [0.8, 0.0, 0.0, 0.0, 0.0],
    "B": [0.0, 1.0, 1.0, 1.0, 1.0],
    "C": [0.7, 0.55, 0.55, 0.55, 0.55],
    "D": [0.6, 1.0, 0.0, 0.0, 0.0],
    "E": [0.25, 0.25, 0.25, 0.25, 0.25],
    "F": [1.0, 0.25, 1.0, 1.0, 1.0],
    "G": [0.85, 0.75, 0.0, 0.0, 0.0],
    "H": [0.6, 0.85, 0.85, 0.0, 0.0],
    "I": [0.0, 0.0, 0.0, 0.0, 1.0],
    "J": [0.0, 0.0, 0.0, 0.0, 1.0],
    "K": [0.85, 1.0, 1.0, 0.0, 0.0],
    "L": [1.0, 1.0, 0.0, 0.0, 0.0],
    "M": [0.15, 0.0, 0.0, 0.0, 0.0],
    "N": [0.3, 0.0, 0.0, 0.0, 0.0],
    "O": [0.45, 0.45, 0.45, 0.45, 0.45],
    "P": [0.85, 1.0, 1.0, 0.0, 0.0],
    "Q": [0.85, 0.75, 0.0, 0.0, 0.0],
    "R": [0.4, 1.0, 1.0, 0.0, 0.0],
    "S": [0.0, 0.0, 0.0, 0.0, 0.0],
    "T": [0.75, 0.0, 0.0, 0.0, 0.0],
    "U": [0.25, 1.0, 1.0, 0.0, 0.0],
    "V": [0.25, 1.0, 1.0, 0.0, 0.0],
    "W": [0.25, 1.0, 1.0, 1.0, 0.0],
    "X": [0.35, 0.65, 0.0, 0.0, 0.0],
    "Y": [1.0, 0.0, 0.0, 0.0, 1.0],
    "Z": [0.3, 1.0, 0.0, 0.0, 0.0],
    "0": [0.45, 0.45, 0.45, 0.45, 0.45],
    "1": [0.2, 1.0, 0.0, 0.0, 0.0],
    "2": [0.2, 1.0, 1.0, 0.0, 0.0],
    "3": [1.0, 1.0, 1.0, 0.0, 0.0],
    "4": [0.0, 1.0, 1.0, 1.0, 1.0],
    "5": [1.0, 1.0, 1.0, 1.0, 1.0],
    "6": [1.0, 0.0, 1.0, 1.0, 1.0],
    "7": [1.0, 1.0, 0.0, 1.0, 1.0],
    "8": [1.0, 1.0, 1.0, 0.0, 1.0],
    "9": [1.0, 1.0, 1.0, 1.0, 0.0],
}


def make_fingerspell_frames(text, fps=20, frames_per_symbol=None):
    symbols = re.findall(r"[A-Za-z0-9]", str(text).upper())
    if not symbols:
        symbols = ["?"]

    frames_per_symbol = frames_per_symbol or max(7, int(fps * 0.34))
    frames = []
    for symbol_index, symbol in enumerate(symbols):
        shape = LETTER_SHAPES.get(symbol, [0.5, 0.5, 0.5, 0.5, 0.5])
        for frame_index in range(frames_per_symbol):
            phase = frame_index / max(frames_per_symbol - 1, 1)
            frames.append(_make_fingerspell_frame(shape, symbol, symbol_index, phase))
        frames.extend(_hold_frames(frames[-1], max(1, fps // 16)))

    return np.asarray(frames, dtype=np.float32).reshape(-1, FRAME_DIM)


def make_separator_frames(frame_count=4):
    return np.zeros((max(1, frame_count), FRAME_DIM), dtype=np.float32)


def _hold_frames(frame, count):
    return [np.asarray(frame, dtype=np.float32).copy() for _ in range(count)]


def _make_fingerspell_frame(shape, symbol, symbol_index, phase):
    joints = _base_pose(phase)
    right_wrist = np.array(
        [
            0.58 + 0.025 * math.sin((symbol_index * 0.31 + phase) * math.tau),
            0.42 + 0.025 * math.sin((phase + 0.25) * math.tau),
        ],
        dtype=np.float32,
    )
    left_wrist = np.array([0.38, 0.54 + 0.01 * math.sin(phase * math.tau)], dtype=np.float32)
    joints[15] = left_wrist
    joints[16] = right_wrist
    _neutral_hand(joints, LEFT_HAND_OFFSET, left_wrist, side=-1)
    _dominant_hand(joints, RIGHT_HAND_OFFSET, right_wrist, shape, symbol, phase)
    return joints.reshape(-1)


def _base_pose(phase):
    joints = np.zeros((75, 2), dtype=np.float32)
    bounce = math.sin(phase * math.tau) * 0.006

    joints[0] = [0.50, 0.18 + bounce]
    joints[1] = [0.48, 0.16 + bounce]
    joints[2] = [0.47, 0.16 + bounce]
    joints[3] = [0.46, 0.16 + bounce]
    joints[4] = [0.52, 0.16 + bounce]
    joints[5] = [0.53, 0.16 + bounce]
    joints[6] = [0.54, 0.16 + bounce]
    joints[7] = [0.44, 0.18 + bounce]
    joints[8] = [0.56, 0.18 + bounce]
    joints[9] = [0.47, 0.23 + bounce]
    joints[10] = [0.53, 0.23 + bounce]
    joints[11] = [0.36, 0.34]
    joints[12] = [0.64, 0.34]
    joints[13] = [0.35, 0.43]
    joints[14] = [0.63, 0.39]
    joints[15] = [0.38, 0.54]
    joints[16] = [0.58, 0.42]
    joints[17] = [0.36, 0.55]
    joints[18] = [0.60, 0.41]
    joints[19] = [0.38, 0.56]
    joints[20] = [0.59, 0.43]
    joints[21] = [0.40, 0.55]
    joints[22] = [0.57, 0.44]
    joints[23] = [0.42, 0.62]
    joints[24] = [0.58, 0.62]
    joints[25] = [0.42, 0.80]
    joints[26] = [0.58, 0.80]
    joints[27] = [0.42, 0.94]
    joints[28] = [0.58, 0.94]
    joints[29] = [0.39, 0.96]
    joints[30] = [0.61, 0.96]
    joints[31] = [0.45, 0.97]
    joints[32] = [0.55, 0.97]
    return joints


def _neutral_hand(joints, offset, wrist, side):
    joints[offset] = wrist
    for finger_index in range(5):
        base_x = (finger_index - 2) * 0.012 * side
        base = wrist + np.array([base_x, 0.008], dtype=np.float32)
        for joint_index in range(4):
            joints[offset + 1 + finger_index * 4 + joint_index] = base + np.array(
                [base_x * 0.2, 0.012 + joint_index * 0.011],
                dtype=np.float32,
            )


def _dominant_hand(joints, offset, wrist, shape, symbol, phase):
    joints[offset] = wrist
    spread = _symbol_spread(symbol, phase)
    thumb_angle = -2.5 + shape[0] * 1.4
    finger_roots = [
        (-0.035, 0.002, thumb_angle),
        (-0.024 - spread, -0.014, -1.58 - spread * 3.0),
        (0.000, -0.020, -1.57),
        (0.024 + spread, -0.014, -1.56 + spread * 3.0),
        (0.046 + spread, 0.000, -1.45 + spread * 3.4),
    ]

    for finger_index, (root_x, root_y, base_angle) in enumerate(finger_roots):
        extension = float(shape[finger_index])
        _finger_points(
            joints,
            offset + 1 + finger_index * 4,
            wrist + np.array([root_x, root_y], dtype=np.float32),
            base_angle,
            extension,
            symbol,
            finger_index,
            phase,
        )


def _finger_points(joints, start_index, root, base_angle, extension, symbol, finger_index, phase):
    curl = (1.0 - extension) * 1.15
    length_scale = 0.78 if finger_index in {0, 4} else 1.0
    if symbol in {"G", "H", "P", "Q"} and finger_index in {1, 2}:
        base_angle = -0.05 if symbol in {"G", "H"} else 1.45
    if symbol == "X" and finger_index == 1:
        curl = 0.55
    if symbol == "R" and finger_index == 1:
        root = root + np.array([0.012, 0.0], dtype=np.float32)
    if symbol == "J" and finger_index == 4:
        root = root + np.array(
            [0.025 * math.sin(phase * math.tau), 0.02 * phase],
            dtype=np.float32,
        )
    if symbol == "Z" and finger_index == 1:
        root = root + np.array(
            [0.04 * (phase - 0.5), 0.018 * math.sin(phase * math.tau)],
            dtype=np.float32,
        )

    point = root
    joints[start_index] = point
    for joint_index in range(1, 4):
        angle = base_angle + curl * joint_index * 0.42
        segment = 0.022 * length_scale * (0.9 + 0.1 * extension)
        point = point + np.array(
            [math.cos(angle) * segment, math.sin(angle) * segment],
            dtype=np.float32,
        )
        joints[start_index + joint_index] = point


def _symbol_spread(symbol, phase):
    if symbol in {"V", "W"}:
        return 0.011
    if symbol == "Y":
        return 0.014
    if symbol == "Z":
        return 0.006 * math.sin(phase * math.tau)
    return 0.003
