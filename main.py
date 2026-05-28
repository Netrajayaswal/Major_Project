from pathlib import Path
import argparse
import csv
import subprocess
import sys
from datetime import datetime

from signlang.pipeline import IncompleteTranslationError, TextToSignPipeline, UnsupportedGlossError
from signlang.web_app import run_web_app


def main():
    parser = argparse.ArgumentParser(description="Text-to-sign skeleton video pipeline.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Generate a skeleton video from text.")
    run_parser.add_argument("--text", default=None)
    run_parser.add_argument("--output", default=None)
    run_parser.add_argument("--text-model", default="outputs/checkpoints/text2gloss")
    run_parser.add_argument("--pose-model", default="outputs/checkpoints/pose_transformer.pt")
    run_parser.add_argument("--pose-source", choices=["auto", "model", "clips"], default="auto")
    run_parser.add_argument("--fps", type=int, default=20)
    run_parser.add_argument("--canvas-size", type=int, default=512)
    run_parser.add_argument("--fallback", choices=["strict", "fingerspell", "pretrained", "remote"], default="remote")
    run_parser.add_argument("--pretrained-dir", default="outputs/checkpoints/pretrained_text2sign")
    run_parser.add_argument("--remote-spoken", default="en")
    run_parser.add_argument("--remote-signed", default="ase")
    run_parser.add_argument("--allow-partial", action="store_true", help="Allow videos that skip untrained important words.")
    run_parser.add_argument("--strict", action="store_true", help="Fail if trained checkpoints are missing.")
    run_parser.add_argument("--allow-preview", action="store_true", help="Allow untrained preview video when no trained sign is available.")

    prepare_parser = subparsers.add_parser("prepare", help="Extract keypoints from videos.")
    prepare_parser.add_argument("--mode", choices=["raw", "rendered"], required=True)
    prepare_parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    prepare_parser.add_argument("--labels-csv", default=None)
    prepare_parser.add_argument("--videos-dir", default=None)
    prepare_parser.add_argument("--output-dir", default=None)
    prepare_parser.add_argument("--auto-label-from-folder", action="store_true")
    prepare_parser.add_argument("--clean-output", action="store_true")
    prepare_parser.add_argument("--skip-existing", action="store_true")

    setup_parser = subparsers.add_parser("setup-data", help="Build labels/text-gloss CSVs from labeled video folders.")
    setup_parser.add_argument("--videos-dir", default=None)
    setup_parser.add_argument("--labels-csv", default=None)
    setup_parser.add_argument("--train-csv", default=None)
    setup_parser.add_argument("--valid-csv", default=None)
    setup_parser.add_argument("--keypoints-train-dir", default=None)
    setup_parser.add_argument("--keypoints-valid-dir", default=None)
    setup_parser.add_argument("--valid-ratio", type=float, default=None)
    setup_parser.add_argument("--seed", type=int, default=None)

    subparsers.add_parser("train-text", help="Fine-tune mT5-small text-to-gloss.")
    subparsers.add_parser("train-pose", help="Train lightweight gloss-to-pose transformer.")
    subparsers.add_parser("status", help="Show dataset and checkpoint status.")

    ui_parser = subparsers.add_parser("ui", help="Start the local two-panel translator web UI.")
    ui_parser.add_argument("--host", default="127.0.0.1")
    ui_parser.add_argument("--port", type=int, default=7860)
    ui_parser.add_argument("--text-model", default="outputs/checkpoints/text2gloss")
    ui_parser.add_argument("--pose-model", default="outputs/checkpoints/pose_transformer.pt")
    ui_parser.add_argument("--pose-source", choices=["auto", "model", "clips"], default="auto")
    ui_parser.add_argument("--fps", type=int, default=20)
    ui_parser.add_argument("--canvas-size", type=int, default=512)
    ui_parser.add_argument("--output-dir", default="outputs/videos/ui")
    ui_parser.add_argument("--fallback", choices=["strict", "fingerspell", "pretrained", "remote"], default="remote")
    ui_parser.add_argument("--pretrained-dir", default="outputs/checkpoints/pretrained_text2sign")
    ui_parser.add_argument("--remote-spoken", default="en")
    ui_parser.add_argument("--remote-signed", default="ase")
    ui_parser.add_argument("--allow-partial", action="store_true", help="Allow videos that skip untrained important words.")
    ui_parser.add_argument("--no-browser", action="store_true", help="Start the server without opening a browser tab.")

    download_parser = subparsers.add_parser("download-pretrained", help="Download optional Hugging Face Text2Sign model.")
    download_parser.add_argument("--repo-id", default="xiaruize/text2sign")
    download_parser.add_argument("--output-dir", default="outputs/checkpoints/pretrained_text2sign")

    render_parser = subparsers.add_parser("render", help="Render one keypoint JSON file.")
    render_parser.add_argument("json_path")
    render_parser.add_argument("--output", default=None)

    argv = sys.argv[1:] or ["ui"]
    args, passthrough = parser.parse_known_args(argv)

    if args.command == "run":
        run_command(args)
    elif args.command == "prepare":
        call_script("scripts/extract_keypoints.py", _prepare_args(args))
    elif args.command == "setup-data":
        call_script("scripts/setup_data.py", _setup_args(args) + passthrough)
    elif args.command == "train-text":
        call_script("scripts/train_text2gloss.py", passthrough)
    elif args.command == "train-pose":
        call_script("scripts/train_pose.py", passthrough)
    elif args.command == "render":
        render_args = [args.json_path]
        if args.output:
            render_args.extend(["--output", args.output])
        call_script("scripts/render_keypoints.py", render_args + passthrough)
    elif args.command == "status":
        status_command()
    elif args.command == "ui":
        run_web_app(
            host=args.host,
            port=args.port,
            text_model_dir=args.text_model,
            pose_checkpoint_path=args.pose_model,
            output_dir=args.output_dir,
            pose_source=args.pose_source,
            fps=args.fps,
            canvas_size=args.canvas_size,
            fallback_mode=args.fallback,
            pretrained_model_dir=args.pretrained_dir,
            remote_spoken_language=args.remote_spoken,
            remote_signed_language=args.remote_signed,
            allow_partial=args.allow_partial,
            open_browser=not args.no_browser,
        )
    elif args.command == "download-pretrained":
        download_pretrained_command(args)
    else:
        parser.error(f"unknown command: {args.command}")


def run_command(args):
    text = args.text or input("Enter input text: ").strip()
    if not text:
        raise SystemExit("No input text provided.")

    output = Path(args.output or default_output_path())
    pipeline = TextToSignPipeline(
        args.text_model,
        args.pose_model,
        pretrained_model_dir=args.pretrained_dir,
        remote_spoken_language=args.remote_spoken,
        remote_signed_language=args.remote_signed,
    )
    try:
        result = pipeline.generate(
            text,
            output,
            fps=args.fps,
            canvas_size=args.canvas_size,
            demo_if_missing=args.allow_preview and not args.strict,
            pose_source=args.pose_source,
            fallback_mode="strict" if args.strict else args.fallback,
            require_complete=not args.allow_partial and args.fallback == "strict",
        )
    except IncompleteTranslationError as exc:
        print(str(exc))
        print("No video generated because the full sentence would be incomplete/inaccurate.")
        print("Add labeled clips for the missing words, run prepare/setup-data, then train-pose again.")
        print("For broader text coverage, use --fallback remote or --fallback pretrained.")
        raise SystemExit(2) from exc
    except UnsupportedGlossError as exc:
        print(str(exc))
        print("No video generated because this would require an untrained/random sign.")
        print("Add labeled clips for the missing glosses, run prepare, then train-pose/train-text again.")
        print("For broader text coverage, use --fallback remote or --fallback pretrained.")
        raise SystemExit(2) from exc
    except RuntimeError as exc:
        if args.fallback == "pretrained":
            print(str(exc))
            print("No pretrained video generated.")
            print("Install the optional model with: python main.py download-pretrained")
            print("Then run: python main.py ui --fallback pretrained")
            raise SystemExit(2) from exc
        if args.fallback == "remote":
            print(str(exc))
            print("No remote sign.mt video generated.")
            print("Check your internet connection and remote language codes.")
            print("Example: python main.py run --fallback remote --remote-spoken en --remote-signed ase --text \"hello world\"")
            raise SystemExit(2) from exc
        raise

    print(f"gloss: {result['gloss']}")
    if result.get("attempted_gloss") and result["attempted_gloss"] != result["gloss"]:
        print(f"attempted gloss: {result['attempted_gloss']}")
    print(f"text source: {result.get('text_source', 'unknown')}")
    print(f"mode: {result['mode']}")
    if result["missing"]:
        print("missing checkpoints: " + ", ".join(result["missing"]))
    if result.get("unsupported_glosses"):
        print("unsupported glosses skipped: " + ", ".join(result["unsupported_glosses"]))
    if result.get("ignored_text_words"):
        print("untrained/ignored text words: " + ", ".join(result["ignored_text_words"]))
    if result.get("fallback_text_words"):
        print("fingerspelled fallback words: " + ", ".join(result["fallback_text_words"]))
    if result["mode"] == "untrained_preview":
        print("this video is an untrained renderer preview, not real sign output")
    print(f"video: {result['output_path']}")


def _prepare_args(args):
    command_args = ["--mode", args.mode, "--split", args.split]
    if args.labels_csv:
        command_args.extend(["--labels-csv", args.labels_csv])
    if args.videos_dir:
        command_args.extend(["--videos-dir", args.videos_dir])
    if args.output_dir:
        command_args.extend(["--output-dir", args.output_dir])
    if args.auto_label_from_folder:
        command_args.append("--auto-label-from-folder")
    if args.clean_output:
        command_args.append("--clean-output")
    if args.skip_existing:
        command_args.append("--skip-existing")
    return command_args


def _setup_args(args):
    command_args = []
    for option in [
        "videos_dir",
        "labels_csv",
        "train_csv",
        "valid_csv",
        "keypoints_train_dir",
        "keypoints_valid_dir",
    ]:
        value = getattr(args, option)
        if value:
            command_args.extend([f"--{option.replace('_', '-')}", value])
    if args.valid_ratio is not None:
        command_args.extend(["--valid-ratio", str(args.valid_ratio)])
    if args.seed is not None:
        command_args.extend(["--seed", str(args.seed)])
    return command_args


def call_script(script, args):
    command = [sys.executable, script] + list(args)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def download_pretrained_command(args):
    from huggingface_hub import snapshot_download

    print("Downloading optional pretrained Text2Sign model.")
    print("Note: this external checkpoint is large and is only used when you pass --fallback pretrained.")
    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.output_dir,
        allow_patterns=[
            "README.md",
            "*.py",
            "models/**",
            "schedulers/**",
            "checkpoint_epoch_70.pt",
        ],
    )
    print(f"pretrained model downloaded to: {path}")


def status_command():
    checks = [
        ("train labels", count_csv_rows("data/labels/train_labels.csv")),
        ("test labels", count_csv_rows("data/labels/test_labels.csv")),
        ("raw train videos", count_files("data/raw_videos/train", VIDEO_SUFFIXES)),
        ("skeleton train videos", count_files("data/skeleton_videos/train", VIDEO_SUFFIXES)),
        ("train keypoint clips", count_files("data/keypoints/train", {".json"})),
        ("valid keypoint clips", count_files("data/keypoints/valid", {".json"})),
        ("test keypoint clips", count_files("data/keypoints/test", {".json"})),
        ("text gloss train pairs", count_csv_rows("data/text_gloss/train.csv")),
        ("text gloss valid pairs", count_csv_rows("data/text_gloss/valid.csv")),
        ("generated videos", count_files("outputs/videos", {".mp4"})),
    ]
    for label, value in checks:
        print(f"{label}: {value}")

    print(f"text checkpoint: {exists_text('outputs/checkpoints/text2gloss')}")
    print(f"pose checkpoint: {Path('outputs/checkpoints/pose_transformer.pt').exists()}")
    glosses = checkpoint_glosses("outputs/checkpoints/pose_transformer.pt")
    if glosses:
        print("trained pose glosses: " + ", ".join(glosses))
    clip_glosses = keypoint_glosses(["data/keypoints/train", "data/keypoints/valid"])
    if clip_glosses:
        print("available keypoint glosses: " + ", ".join(clip_glosses))


def count_files(directory, suffixes):
    path = Path(directory)
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in suffixes)


def count_csv_rows(path):
    csv_path = Path(path)
    if not csv_path.exists():
        return 0
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
        return sum(1 for row in csv.DictReader(file) if any(value.strip() for value in row.values() if value))


def exists_text(path):
    model_path = Path(path)
    if not model_path.exists():
        return False
    if (model_path / "config.json").exists():
        return True
    return any(
        checkpoint.is_dir() and (checkpoint / "config.json").exists()
        for checkpoint in model_path.glob("checkpoint-*")
    )


def checkpoint_glosses(path):
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return []
    try:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return []
    token_to_id = checkpoint.get("vocab", {}).get("token_to_id", {})
    return sorted(token for token in token_to_id if not token.startswith("<"))


def keypoint_glosses(directories):
    glosses = set()
    for directory in directories:
        root = Path(directory)
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            try:
                import json

                with open(path, "r", encoding="utf-8") as file:
                    gloss = str(json.load(file).get("gloss", "")).strip().upper()
            except Exception:
                continue
            if gloss:
                glosses.add(gloss)
    return sorted(glosses)


def default_output_path():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/videos") / f"sign_{timestamp}.mp4"


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


if __name__ == "__main__":
    main()
