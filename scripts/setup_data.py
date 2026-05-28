"""Auto-populate labels and text-gloss CSVs from skeleton_videos folder structure.

Usage:
    python scripts/setup_data.py

Scans data/skeleton_videos/train/ for sub-folders.  Each sub-folder name
becomes the gloss label for every video inside it.
Generates:
    data/labels/train_labels.csv
    data/text_gloss/train.csv
    data/text_gloss/valid.csv
Also copies ~20% of keypoints to data/keypoints/valid after extraction.
"""

from pathlib import Path
import argparse
import csv
import random
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signlang.dataset import group_records_by_gloss, iter_labeled_videos, write_label_csv

SENTENCE_TEMPLATES = {
    "HOME": [
        "I am going home",
        "Let's go home",
        "This is my home",
        "Home sweet home",
        "I want to go home",
        "Where is your home",
        "She is at home",
        "He went home",
        "We are home",
        "Take me home",
        "I love my home",
        "Home is where the heart is",
        "I will be home soon",
        "Are you home",
        "They are not home",
        "Go home now",
        "I miss home",
        "Welcome home",
        "Home",
        "I need to go home",
        "Is anyone home",
        "Nobody is home",
        "Come home with me",
        "My home is far away",
        "Our home is beautiful",
    ],
}

# Fallback templates for any gloss not in the dict above
GENERIC_TEMPLATES = [
    "{gloss_lower}",
    "Please sign {gloss_lower}",
    "Show me {gloss_lower}",
    "I want to say {gloss_lower}",
    "The word is {gloss_lower}",
    "Can you sign {gloss_lower}",
    "Sign {gloss_lower} for me",
    "How do you sign {gloss_lower}",
]


def scan_skeleton_videos(base_dir):
    """Return dict mapping gloss -> list of relative video paths."""
    records = list(iter_labeled_videos(base_dir))
    warn_unlabeled_videos(base_dir)
    return group_records_by_gloss(records)


def write_train_labels(gloss_videos, output_path):
    """Write train_labels.csv from gloss_videos mapping."""
    records = []
    for gloss, videos in sorted(gloss_videos.items()):
        for video_rel in videos:
            records.append({"relative_path": str(video_rel), "gloss": gloss})
    write_label_csv(records, output_path)
    count = len(records)
    print(f"Wrote {count} rows to {output_path}")
    return count


def generate_text_gloss_pairs(glosses, train_ratio=0.8):
    """Generate sentence/gloss pairs for text-to-gloss training."""
    train_pairs = []
    valid_pairs = []

    for gloss in glosses:
        if gloss in SENTENCE_TEMPLATES:
            sentences = SENTENCE_TEMPLATES[gloss]
        else:
            gloss_lower = gloss.lower()
            sentences = [t.format(gloss_lower=gloss_lower) for t in GENERIC_TEMPLATES]

        random.shuffle(sentences)
        split_idx = max(1, int(len(sentences) * train_ratio))
        train_pairs.extend((s, gloss) for s in sentences[:split_idx])
        valid_pairs.extend((s, gloss) for s in sentences[split_idx:])

    # Ensure at least some validation data
    if not valid_pairs and train_pairs:
        random.shuffle(train_pairs)
        split = max(1, len(train_pairs) // 5)
        valid_pairs = train_pairs[-split:]
        train_pairs = train_pairs[:-split]

    return train_pairs, valid_pairs


def write_text_gloss_csv(pairs, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence", "gloss_sequence"])
        for sentence, gloss in pairs:
            writer.writerow([sentence, gloss])
    print(f"Wrote {len(pairs)} rows to {output_path}")


def split_keypoints_for_validation(train_dir, valid_dir, ratio=0.2, allowed_glosses=None):
    """Copy a fraction of train keypoints to valid for pose training."""
    train_path = Path(train_dir)
    valid_path = Path(valid_dir)
    valid_path.mkdir(parents=True, exist_ok=True)

    all_json_files = sorted(train_path.rglob("*.json"))
    json_files = all_json_files
    if allowed_glosses:
        json_files = [path for path in json_files if keypoint_gloss(path) in allowed_glosses]
    if not json_files:
        if all_json_files:
            print(f"No keypoint JSON files in {train_dir} match the current video-folder glosses.")
            print("Run prepare with --clean-output to rebuild keypoints for the current labels, then re-run setup-data.")
        else:
            print(f"No keypoint JSON files in {train_dir} yet. Run 'prepare' first, then re-run this.")
        return 0

    random.shuffle(json_files)
    split_count = max(1, int(len(json_files) * ratio))
    copied = 0
    for f in json_files[:split_count]:
        dest = valid_path / f.relative_to(train_path)
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
            copied += 1
    print(f"Copied {copied} keypoint files to {valid_path} for validation")
    return copied


def keypoint_gloss(path):
    try:
        import json

        with open(path, "r", encoding="utf-8") as file:
            return str(json.load(file).get("gloss", "")).strip().upper()
    except (OSError, ValueError):
        return ""


def warn_unlabeled_videos(base_dir):
    base = Path(base_dir)
    if not base.exists():
        return
    direct_videos = [
        path for path in base.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ]
    if direct_videos:
        print(f"[warn] Found {len(direct_videos)} videos directly in {base_dir} without a gloss folder.")
        print("       Move them into sub-folders named by their gloss (e.g. Home/, Water/).")


def main():
    parser = argparse.ArgumentParser(description="Create labels and text-gloss CSVs from labeled video folders.")
    parser.add_argument("--videos-dir", default=str(PROJECT_ROOT / "data" / "skeleton_videos" / "train"))
    parser.add_argument("--labels-csv", default=str(PROJECT_ROOT / "data" / "labels" / "train_labels.csv"))
    parser.add_argument("--train-csv", default=str(PROJECT_ROOT / "data" / "text_gloss" / "train.csv"))
    parser.add_argument("--valid-csv", default=str(PROJECT_ROOT / "data" / "text_gloss" / "valid.csv"))
    parser.add_argument("--keypoints-train-dir", default=str(PROJECT_ROOT / "data" / "keypoints" / "train"))
    parser.add_argument("--keypoints-valid-dir", default=str(PROJECT_ROOT / "data" / "keypoints" / "valid"))
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    skeleton_base = Path(args.videos_dir)
    labels_csv = Path(args.labels_csv)
    train_csv = Path(args.train_csv)
    valid_csv = Path(args.valid_csv)

    print("=" * 60)
    print("STEP 1: Scanning skeleton videos for gloss folders...")
    print("=" * 60)
    gloss_videos = scan_skeleton_videos(skeleton_base)

    if not gloss_videos:
        print("ERROR: No gloss sub-folders found in", skeleton_base)
        print("Place videos in sub-folders named by gloss, e.g.:")
        print("  data/skeleton_videos/train/Home/video1.mp4")
        print("  data/skeleton_videos/train/Water/video2.mp4")
        return

    print(f"\nFound {len(gloss_videos)} gloss(es):")
    for gloss, videos in gloss_videos.items():
        print(f"  {gloss}: {len(videos)} videos")

    print("\n" + "=" * 60)
    print("STEP 2: Writing train_labels.csv...")
    print("=" * 60)
    write_train_labels(gloss_videos, labels_csv)

    print("\n" + "=" * 60)
    print("STEP 3: Generating text-gloss pairs...")
    print("=" * 60)
    glosses = sorted(gloss_videos.keys())
    train_pairs, valid_pairs = generate_text_gloss_pairs(glosses)
    write_text_gloss_csv(train_pairs, train_csv)
    write_text_gloss_csv(valid_pairs, valid_csv)

    print("\n" + "=" * 60)
    print("STEP 4: Splitting keypoints for validation...")
    print("=" * 60)
    split_keypoints_for_validation(
        Path(args.keypoints_train_dir),
        Path(args.keypoints_valid_dir),
        ratio=args.valid_ratio,
        allowed_glosses=set(glosses),
    )

    print("\n" + "=" * 60)
    print("DATA SETUP COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. python main.py prepare --mode rendered --split train --auto-label-from-folder --clean-output")
    print("  2. python main.py setup-data      (re-run to split keypoints)")
    print("  3. python main.py train-text")
    print("  4. python main.py train-pose")
    example_text = glosses[0].lower() if glosses else "your gloss"
    print(f"  5. python main.py run --text \"{example_text}\"")


if __name__ == "__main__":
    main()
