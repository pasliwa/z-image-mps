#!/usr/bin/env python3
import argparse
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio


FILENAME_RE = re.compile(r"(.+)__([0-9]+)_([0-9]+)\.jpg$", re.IGNORECASE)


def find_images(paths):
    """Yield Path objects for all .jpg files under the given paths."""
    result = []
    for p in paths:
        p = Path(p).expanduser()
        if p.is_dir():
            result.extend(sorted(p.rglob("*.jpg")))
        elif p.is_file() and p.suffix.lower() == ".jpg":
            result.append(p)
        else:
            print(f"Skipping non-existent or non-jpg path: {p}")
    return result


def parse_filename(path: Path):
    """
    Parse filename like:

        1764797370915__000000050_0.jpg
        ^^^^^^^^^^^^^    ^^^^^^^ ^- sample_id
        prefix           step

    Returns (sample_id, step_int, step_str) or None if not matching.
    """
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    step_str = m.group(2)
    sample_id = m.group(3)
    step_int = int(step_str)
    return sample_id, step_int, step_str


def group_by_sample(image_paths):
    """
    Group images by sample id.

    Returns dict:
        sample_id -> list of (step_int, step_str, path)
    """
    groups = defaultdict(list)
    for p in image_paths:
        parsed = parse_filename(p)
        if parsed is None:
            print(f"Warning: filename does not match pattern, skipping: {p}")
            continue
        sample_id, step_int, step_str = parsed
        groups[sample_id].append((step_int, step_str, p))
    return groups


def add_step_text(image: Image.Image, step_str: str, corner: str = "top_left"):
    """
    Overlay the step text on the image in the chosen corner.

    corner: 'top_left', 'top_right', 'bottom_left', 'bottom_right'
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    text = f"step {int(step_str)}"

    # Use textbbox to measure text size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    margin = 10
    if corner == "top_left":
        x, y = margin, margin
    elif corner == "top_right":
        x, y = img.width - text_w - margin, margin
    elif corner == "bottom_left":
        x, y = margin, img.height - text_h - margin
    else:  # bottom_right
        x, y = img.width - text_w - margin, img.height - text_h - margin

    # Background rectangle behind the text for readability
    rect_pad = 4
    rect = [
        x - rect_pad,
        y - rect_pad,
        x + text_w + rect_pad,
        y + text_h + rect_pad,
    ]
    draw.rectangle(rect, fill=(0, 0, 0))  # solid black background

    # Draw the text in white
    draw.text((x, y), text, fill="white", font=font)
    return img

def make_gif_for_sample(sample_id, frames_info, output_dir, duration_ms=500, corner="top_left"):
    """
    Create a GIF for a single sample.

    frames_info: list of (step_int, step_str, path)
    """
    frames_info = sorted(frames_info, key=lambda x: x[0])  # sort by step_int
    frames = []

    for step_int, step_str, path in frames_info:
        im = Image.open(path).convert("RGB")
        im = add_step_text(im, step_str, corner=corner)
        frames.append(im)

    if not frames:
        print(f"No frames for sample {sample_id}, skipping GIF.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"sample_{sample_id}.gif"

    # Save GIF
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"Saved GIF for sample {sample_id}: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create per-sample GIFs from LORA training samples with step overlay."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Input directories and/or individual .jpg files (e.g. path/to/samples/*.jpg/png)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="gifs",
        help="Directory where GIFs will be saved (default: ./gifs)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=500,
        help="Frame duration in milliseconds (default: 500)",
    )
    parser.add_argument(
        "--corner",
        type=str,
        default="top_left",
        choices=["top_left", "top_right", "bottom_left", "bottom_right"],
        help="Corner in which to draw the step text (default: top_left)",
    )

    args = parser.parse_args()

    images = find_images(args.paths)
    if not images:
        print("No .jpg images found. Check your paths.")
        return

    groups = group_by_sample(images)
    if not groups:
        print("No valid images matching the naming pattern were found.")
        return

    output_dir = Path(args.output_dir).expanduser()

    for sample_id, frames_info in groups.items():
        make_gif_for_sample(
            sample_id=sample_id,
            frames_info=frames_info,
            output_dir=output_dir,
            duration_ms=args.duration,
            corner=args.corner,
        )


if __name__ == "__main__":
    main()
