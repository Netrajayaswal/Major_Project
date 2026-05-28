import math

import numpy as np

from signlang.topology import FRAME_DIM, LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET


def make_demo_frames(gloss_sequence, fps=20):
    tokens = [token for token in gloss_sequence.split() if token]
    if not tokens:
        tokens = ["SIGN"]

    frames = []
    frames_per_token = max(12, fps)
    for token_index, _token in enumerate(tokens):
        for frame_index in range(frames_per_token):
            phase = frame_index / frames_per_token
            frames.append(_make_pose_frame(token_index, phase))

    return np.asarray(frames, dtype=np.float32).reshape(-1, FRAME_DIM)


def _make_pose_frame(token_index, phase):
    joints = np.zeros((75, 2), dtype=np.float32)
    bounce = math.sin(phase * math.tau) * 0.025
    wave = math.sin((phase + token_index * 0.17) * math.tau)

    joints[0] = [0.50, 0.18 + bounce * 0.25]
    joints[1] = [0.48, 0.16]
    joints[2] = [0.47, 0.16]
    joints[3] = [0.46, 0.16]
    joints[4] = [0.52, 0.16]
    joints[5] = [0.53, 0.16]
    joints[6] = [0.54, 0.16]
    joints[7] = [0.44, 0.18]
    joints[8] = [0.56, 0.18]
    joints[9] = [0.47, 0.23]
    joints[10] = [0.53, 0.23]
    joints[11] = [0.36, 0.34]
    joints[12] = [0.64, 0.34]
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

    left_wrist = np.array([0.30 + 0.08 * wave, 0.46 - 0.08 * abs(wave)], dtype=np.float32)
    right_wrist = np.array([0.70 - 0.08 * wave, 0.46 - 0.08 * abs(wave)], dtype=np.float32)
    joints[13] = [0.31, 0.39]
    joints[14] = [0.69, 0.39]
    joints[15] = left_wrist
    joints[16] = right_wrist
    joints[17] = left_wrist + [-0.02, -0.01]
    joints[18] = right_wrist + [0.02, -0.01]
    joints[19] = left_wrist + [-0.01, 0.01]
    joints[20] = right_wrist + [0.01, 0.01]
    joints[21] = left_wrist + [0.01, 0.00]
    joints[22] = right_wrist + [-0.01, 0.00]

    _fill_hand(joints, LEFT_HAND_OFFSET, left_wrist, side=-1, phase=phase)
    _fill_hand(joints, RIGHT_HAND_OFFSET, right_wrist, side=1, phase=phase + 0.25)
    return joints.reshape(-1)


def _fill_hand(joints, offset, wrist, side, phase):
    joints[offset] = wrist
    finger_roots = [
        np.array([0.020 * side, -0.005], dtype=np.float32),
        np.array([0.010 * side, -0.025], dtype=np.float32),
        np.array([0.000, -0.030], dtype=np.float32),
        np.array([-0.010 * side, -0.025], dtype=np.float32),
        np.array([-0.020 * side, -0.010], dtype=np.float32),
    ]
    bend = math.sin(phase * math.tau) * 0.008
    index = 1
    for finger_index, root in enumerate(finger_roots):
        direction = root / max(float(np.linalg.norm(root)), 1e-6)
        for joint_index in range(4):
            length = 0.018 + 0.004 * joint_index
            curl = np.array([side * bend * (finger_index - 2), abs(bend) * joint_index], dtype=np.float32)
            joints[offset + index] = wrist + root + direction * length * joint_index + curl
            index += 1

