from pathlib import Path
import argparse
import csv
import json
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signlang.topology import FRAME_DIM, JOINT_COUNT
from signlang.dataset import is_video_file, iter_labeled_videos


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe-style keypoint JSON files.")
    parser.add_argument("--mode", choices=["raw", "rendered"], required=True)
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--labels-csv")
    parser.add_argument("--videos-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--expected-joints", type=int, default=JOINT_COUNT)
    parser.add_argument("--auto-label-from-folder", action="store_true")
    parser.add_argument("--clean-output", action="store_true", help="Delete existing JSON files in output-dir before extraction.")
    parser.add_argument("--skip-existing", action="store_true", help="Do not re-extract videos whose output JSON already exists.")
    args = parser.parse_args()

    labels_csv = Path(args.labels_csv or f"data/labels/{args.split}_labels.csv")
    videos_dir = Path(args.videos_dir or _default_video_dir(args.mode, args.split))
    output_dir = Path(args.output_dir or f"data/keypoints/{args.split}")
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = read_labels(labels_csv)
    if not labels and (args.auto_label_from_folder or has_labeled_video_folders(videos_dir)):
        labels = labels_from_folder_names(videos_dir)
    if not labels:
        raise SystemExit(
            f"No labels found in {labels_csv}. Add video_filename,gloss rows first, "
            "or pass --auto-label-from-folder when videos are inside gloss-named folders."
        )

    if args.clean_output:
        deleted = clean_json_files(output_dir)
        print(f"cleaned {deleted} existing JSON files from {output_dir}")

    print(f"labels: {len(labels)}")
    print(f"videos dir: {videos_dir}")

    written = 0
    skipped_existing = 0
    for video_name, gloss in labels.items():
        video_path = find_video(videos_dir, video_name)
        if video_path is None:
            print(f"[skip] missing video: {video_name}")
            continue
        output_path = output_json_path(output_dir, videos_dir, video_path)
        if args.skip_existing and output_path.exists():
            skipped_existing += 1
            continue
        if args.mode == "raw":
            frames = extract_from_raw_video(video_path)
        else:
            frames = extract_from_rendered_video(video_path, expected_joints=args.expected_joints)
        if len(frames) == 0:
            print(f"[skip] no frames extracted: {video_path}")
            continue
        normalized_frames = normalize_clip(frames)
        write_clip_json(output_path, gloss, video_path, normalized_frames, args.mode)
        written += 1
        print(f"[ok] {video_path.name} -> {output_path}")

    print(f"wrote {written} keypoint clips to {output_dir}")
    if skipped_existing:
        print(f"skipped {skipped_existing} existing keypoint clips")


def _default_video_dir(mode, split):
    if mode == "raw":
        return f"data/raw_videos/{split}"
    return f"data/skeleton_videos/{split}"


def read_labels(path):
    labels = {}
    if not Path(path).exists():
        return labels
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            filename = (row.get("video_filename") or row.get("filename") or row.get("video") or "").strip()
            gloss = (row.get("gloss") or row.get("label") or "").strip().upper()
            if filename and gloss:
                labels[filename] = gloss
    return labels


def find_video(videos_dir, video_name):
    normalized_name = str(video_name).replace("\\", "/")
    direct = videos_dir / normalized_name
    if direct.exists():
        return direct
    stem = Path(video_name).stem
    # Search top-level directory
    for path in videos_dir.iterdir():
        if is_video_file(path) and path.stem == stem:
            return path
    # Search subdirectories recursively
    for path in videos_dir.rglob("*"):
        if is_video_file(path) and path.stem == stem:
            return path
    return None


def labels_from_folder_names(videos_dir):
    labels = {}
    for record in iter_labeled_videos(videos_dir):
        labels[record["relative_path"]] = record["gloss"]
    return labels


def has_labeled_video_folders(videos_dir):
    return any(True for _record in iter_labeled_videos(videos_dir))


def clean_json_files(output_dir):
    deleted = 0
    for path in Path(output_dir).rglob("*.json"):
        path.unlink()
        deleted += 1
    return deleted


def output_json_path(output_dir, videos_dir, video_path):
    try:
        relative_path = Path(video_path).relative_to(Path(videos_dir))
    except ValueError:
        relative_path = Path(video_path).name
    return (Path(output_dir) / relative_path).with_suffix(".json")


def extract_from_raw_video(video_path):
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit("mediapipe is required for --mode raw. Install requirements.txt first.") from exc

    cap = cv2.VideoCapture(str(video_path))
    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
    )
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)
            joints = np.zeros((JOINT_COUNT, 2), dtype=np.float32)
            _copy_landmarks(result.pose_landmarks, joints, 0, 33)
            _copy_landmarks(result.left_hand_landmarks, joints, 33, 21)
            _copy_landmarks(result.right_hand_landmarks, joints, 54, 21)
            frames.append(joints)
    finally:
        holistic.close()
        cap.release()
    return np.asarray(frames, dtype=np.float32)


def _copy_landmarks(landmarks, joints, offset, count):
    if landmarks is None:
        return
    for index, landmark in enumerate(landmarks.landmark[:count]):
        joints[offset + index] = [landmark.x, landmark.y]


def extract_from_rendered_video(video_path, expected_joints=JOINT_COUNT):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    previous = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            points = detect_red_centroids(frame)
            ordered = order_points(points, previous, expected_joints=expected_joints)
            previous = ordered
            height, width = frame.shape[:2]
            normalized = ordered.copy()
            normalized[:, 0] = normalized[:, 0] / max(width - 1, 1)
            normalized[:, 1] = normalized[:, 1] / max(height - 1, 1)
            frames.append(normalized)
    finally:
        cap.release()
    return np.asarray(frames, dtype=np.float32)


def detect_red_centroids(frame):
    red = frame[:, :, 2]
    green = frame[:, :, 1]
    blue = frame[:, :, 0]
    mask = ((red > 150) & (green < 80) & (blue < 80)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for contour in contours:
        if cv2.contourArea(contour) < 2:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        centroids.append([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
    return np.asarray(centroids, dtype=np.float32)


def order_points(points, previous, expected_joints=JOINT_COUNT):
    ordered = np.zeros((expected_joints, 2), dtype=np.float32)
    if len(points) == 0:
        return previous.copy() if previous is not None else ordered

    if previous is None:
        points = points[np.lexsort((points[:, 0], points[:, 1]))]
        count = min(expected_joints, len(points))
        ordered[:count] = points[:count]
        return ordered

    try:
        from scipy.optimize import linear_sum_assignment

        distances = np.linalg.norm(previous[:, None, :] - points[None, :, :], axis=-1)
        row_indices, col_indices = linear_sum_assignment(distances)
        ordered[:] = previous
        for row, col in zip(row_indices, col_indices):
            ordered[row] = points[col]
        return ordered
    except ImportError:
        remaining = list(range(len(points)))
        ordered[:] = previous
        for joint_index, joint in enumerate(previous):
            if not remaining:
                break
            distances = np.linalg.norm(points[remaining] - joint, axis=-1)
            best_local = int(np.argmin(distances))
            best_point_index = remaining.pop(best_local)
            ordered[joint_index] = points[best_point_index]
        return ordered


def normalize_clip(frames):
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[1:] != (JOINT_COUNT, 2):
        raise ValueError(f"expected frames shaped [n, {JOINT_COUNT}, 2]")

    valid = np.isfinite(frames).all(axis=-1) & ~np.all(np.isclose(frames, 0.0), axis=-1)
    if not valid.any():
        return np.zeros((len(frames), FRAME_DIM), dtype=np.float32)

    valid_points = frames[valid]
    minimum = valid_points.min(axis=0)
    maximum = valid_points.max(axis=0)
    scale = np.maximum(maximum - minimum, 1e-6)
    normalized = (frames - minimum) / scale
    normalized[~valid] = 0.0
    return np.clip(normalized, 0.0, 1.0).reshape(len(frames), FRAME_DIM).astype(np.float32)


def write_clip_json(path, gloss, video_path, frames, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "gloss": gloss,
        "source_video": str(video_path),
        "extractor": mode,
        "joint_count": JOINT_COUNT,
        "frame_dim": FRAME_DIM,
        "coordinate_space": "normalized_bbox",
        "frames": frames.tolist(),
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file)


if __name__ == "__main__":
    main()
