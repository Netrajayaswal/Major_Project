from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signlang.render import load_frames_from_json, render_keypoints_video


def main():
    parser = argparse.ArgumentParser(description="Render a keypoint JSON clip to MP4.")
    parser.add_argument("json_path")
    parser.add_argument("--output", default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--canvas-size", type=int, default=512)
    parser.add_argument("--no-smooth", action="store_true")
    args = parser.parse_args()

    json_path = Path(args.json_path)
    output = Path(args.output) if args.output else Path("outputs/videos") / f"{json_path.stem}.mp4"
    frames = load_frames_from_json(json_path)
    render_keypoints_video(
        frames,
        output,
        fps=args.fps,
        canvas_size=args.canvas_size,
        smooth=not args.no_smooth,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

