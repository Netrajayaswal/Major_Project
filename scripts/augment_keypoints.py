from pathlib import Path
import argparse
import json
import random
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signlang.topology import FRAME_DIM, JOINT_COUNT, LEFT_RIGHT_POSE_PAIRS


def main():
    parser = argparse.ArgumentParser(description="Augment normalized keypoint JSON clips.")
    parser.add_argument("--input-dir", default="data/keypoints/train")
    parser.add_argument("--output-dir", default="data/keypoints/train")
    parser.add_argument("--copies", type=int, default=2)
    parser.add_argument("--noise-std", type=float, default=0.005)
    parser.add_argument("--max-rotation-deg", type=float, default=5.0)
    parser.add_argument("--time-stretch", type=float, default=0.2)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(path for path in input_dir.rglob("*.json") if "_aug" not in path.stem)
    if not files:
        raise SystemExit(
            f"No source JSON files found in {input_dir}. "
            "Run prepare/setup-data first, then retry augmentation."
        )

    written = 0
    for path in files:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        frames = np.asarray(data["frames"], dtype=np.float32).reshape(-1, JOINT_COUNT, 2)
        for copy_index in range(args.copies):
            augmented = augment_clip(
                frames,
                noise_std=args.noise_std,
                max_rotation_deg=args.max_rotation_deg,
                time_stretch=args.time_stretch,
            )
            output = dict(data)
            output["frames"] = augmented.reshape(-1, FRAME_DIM).tolist()
            output["augmentation"] = {
                "source": str(path.relative_to(input_dir)),
                "copy_index": copy_index,
            }
            relative_parent = path.parent.relative_to(input_dir)
            target_dir = output_dir / relative_parent
            target_dir.mkdir(parents=True, exist_ok=True)
            output_path = target_dir / f"{path.stem}_aug{copy_index + 1}.json"
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(output, file)
            written += 1
    print(f"wrote {written} augmented clips to {output_dir}")


def augment_clip(frames, noise_std=0.005, max_rotation_deg=5.0, time_stretch=0.2):
    augmented = np.asarray(frames, dtype=np.float32).copy()

    if random.random() < 0.5:
        augmented = horizontal_flip(augmented)

    factor = random.uniform(1.0 - time_stretch, 1.0 + time_stretch)
    augmented = stretch_time(augmented, factor)

    degrees = random.uniform(-max_rotation_deg, max_rotation_deg)
    augmented = rotate_clip(augmented, degrees)

    valid = ~np.all(np.isclose(augmented, 0.0), axis=-1)
    augmented[valid] += np.random.normal(0.0, noise_std, size=augmented[valid].shape)
    augmented = np.clip(augmented, 0.0, 1.0)
    augmented[~valid] = 0.0
    return augmented.astype(np.float32)


def horizontal_flip(frames):
    flipped = frames.copy()
    flipped[..., 0] = 1.0 - flipped[..., 0]

    for left, right in LEFT_RIGHT_POSE_PAIRS:
        flipped[:, [left, right], :] = flipped[:, [right, left], :]
    flipped[:, 33:54, :], flipped[:, 54:75, :] = (
        flipped[:, 54:75, :].copy(),
        flipped[:, 33:54, :].copy(),
    )
    return flipped


def stretch_time(frames, factor):
    old_length = len(frames)
    new_length = max(2, int(round(old_length * factor)))
    old_positions = np.linspace(0.0, 1.0, old_length)
    new_positions = np.linspace(0.0, 1.0, new_length)
    stretched = np.empty((new_length, JOINT_COUNT, 2), dtype=np.float32)
    flat = frames.reshape(old_length, -1)
    for dim_index in range(flat.shape[1]):
        stretched.reshape(new_length, -1)[:, dim_index] = np.interp(
            new_positions,
            old_positions,
            flat[:, dim_index],
        )
    return stretched


def rotate_clip(frames, degrees):
    radians = np.deg2rad(degrees)
    rotation = np.array(
        [
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)],
        ],
        dtype=np.float32,
    )
    center = body_center(frames)
    valid = ~np.all(np.isclose(frames, 0.0), axis=-1)
    rotated = frames.copy()
    for frame_index in range(len(frames)):
        points = rotated[frame_index]
        points[valid[frame_index]] = (
            (points[valid[frame_index]] - center[frame_index]) @ rotation.T
            + center[frame_index]
        )
    return rotated


def body_center(frames):
    hips = frames[:, [23, 24], :]
    valid_hips = ~np.all(np.isclose(hips, 0.0), axis=-1)
    centers = np.zeros((len(frames), 2), dtype=np.float32)
    for index in range(len(frames)):
        if valid_hips[index].any():
            centers[index] = hips[index][valid_hips[index]].mean(axis=0)
        else:
            valid = ~np.all(np.isclose(frames[index], 0.0), axis=-1)
            centers[index] = frames[index][valid].mean(axis=0) if valid.any() else [0.5, 0.5]
    return centers


if __name__ == "__main__":
    main()
