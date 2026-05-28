from pathlib import Path
import csv
import re


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

IGNORED_LABEL_FOLDERS = {
    "clip",
    "clips",
    "extra",
    "extras",
    "sample",
    "samples",
    "video",
    "videos",
}


def is_video_file(path):
    path = Path(path)
    return path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES


def normalize_gloss_name(name):
    text = str(name).strip()
    text = re.sub(r"^\s*\d+\s*[\.\)\]\-_]*\s*", "", text)
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.upper()


def infer_gloss_from_video_path(video_path, videos_dir):
    root = Path(videos_dir).resolve()
    current = Path(video_path).resolve().parent
    while current != root and current != current.parent:
        raw_name = current.name.strip().lower()
        gloss = normalize_gloss_name(current.name)
        if gloss and raw_name not in IGNORED_LABEL_FOLDERS and gloss.lower() not in IGNORED_LABEL_FOLDERS:
            return gloss
        current = current.parent
    return ""


def iter_labeled_videos(videos_dir):
    root = Path(videos_dir)
    if not root.exists():
        return
    video_paths = [path for path in root.rglob("*") if is_video_file(path)]
    for path in sorted(video_paths, key=lambda item: str(item).lower()):
        gloss = infer_gloss_from_video_path(path, root)
        if not gloss:
            continue
        yield {
            "path": path,
            "relative_path": str(path.relative_to(root)),
            "gloss": gloss,
        }


def group_records_by_gloss(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record["gloss"], []).append(record["relative_path"])
    return grouped


def write_label_csv(records, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["video_filename", "gloss"])
        for record in sorted(records, key=lambda item: (item["gloss"], item["relative_path"].lower())):
            writer.writerow([record["relative_path"], record["gloss"]])
