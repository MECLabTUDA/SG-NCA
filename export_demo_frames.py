"""Create a deterministic multi-position Cholec demo clip for Android."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image


TEMPORAL_OFFSETS = [-25, -21, -18, -14, -11, -7, -4, 0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-id", default="VID01")
    parser.add_argument("--start-frame", type=int, default=500)
    parser.add_argument("--target-count", type=int, default=20)
    parser.add_argument("--target-step", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_frames = [
        args.start_frame + index * args.target_step
        for index in range(args.target_count)
    ]
    required_frames = sorted({
        target + offset
        for target in target_frames
        for offset in TEMPORAL_OFFSETS
    })
    if required_frames[0] < 0:
        raise ValueError("The first target does not have one second of history.")

    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(args.video)
    try:
        for frame_index in required_frames:
            frame = Image.fromarray(reader.get_data(frame_index)).resize((256, 256))
            frame.save(frames_dir / f"{frame_index:08d}.png")
    finally:
        reader.close()

    metadata = {
        "name": f"Cholec {args.video_id} — frames {target_frames[0]}–{target_frames[-1]}",
        "video_id": args.video_id,
        "fps": 25,
        "target_step": args.target_step,
        "target_frames": target_frames,
        "temporal_offsets": TEMPORAL_OFFSETS,
        "available_frames": required_frames,
    }
    (args.output_dir / "demo_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(
        f"Exported {len(required_frames)} unique images for "
        f"{len(target_frames)} navigable targets to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
