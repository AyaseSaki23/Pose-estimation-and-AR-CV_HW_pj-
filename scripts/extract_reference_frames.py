import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract evenly spaced reference frames from a video."
    )
    parser.add_argument(
        "--video",
        default="data/raw/videos/input_rotation.mp4",
        help="Input video path, relative to project root or absolute.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/objects/object_01/reference_extracted",
        help="Output directory for extracted jpg frames.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of frames to extract.",
    )
    parser.add_argument(
        "--prefix",
        default="ref_candidate",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete old jpg files in output directory before extraction.",
    )
    return parser.parse_args()


def resolve_path(project_root, path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return project_root / path


def main():
    project_root = Path(__file__).resolve().parents[1]
    args = parse_args()

    video_path = resolve_path(project_root, args.video)
    output_dir = resolve_path(project_root, args.output_dir)

    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for jpg_path in output_dir.glob("*.jpg"):
            jpg_path.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if frame_count <= 0:
        raise SystemExit("Could not read video frame count")

    if args.count == 1:
        indices = [0]
    else:
        margin = max(0, min(frame_count // 20, 15))
        start = margin
        end = max(start, frame_count - 1 - margin)
        indices = [
            round(start + i * (end - start) / (args.count - 1))
            for i in range(args.count)
        ]

    saved = 0
    for out_index, frame_index in enumerate(indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"skip frame {frame_index}: read failed")
            continue

        output_path = output_dir / f"{args.prefix}_{out_index:02d}_frame{frame_index:06d}.jpg"
        if not cv2.imwrite(str(output_path), frame):
            print(f"skip frame {frame_index}: write failed")
            continue

        timestamp = frame_index / fps if fps > 0 else 0.0
        print(f"saved {output_path}  frame={frame_index}  time={timestamp:.2f}s")
        saved += 1

    cap.release()
    print(f"Done. Saved {saved}/{args.count} frames to {output_dir}")


if __name__ == "__main__":
    main()
