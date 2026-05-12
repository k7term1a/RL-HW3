"""
Assemble per-epoch snapshot GIFs into a single animated overview GIF.

Each epoch's full episode (all steps) is played through completely,
then the next epoch's episode begins — so you see the agent's behaviour
evolving across training checkpoints.

Usage:
    uv run python make_training_gif.py --dir outputs/hw3_1 --out outputs/hw3_1/training_progress.gif
    uv run python make_training_gif.py --dir outputs/hw3_2/double_dqn --out outputs/hw3_2/double_progress.gif
"""
import argparse
import glob
import os
from PIL import Image, ImageDraw, ImageFont


def extract_all_frames(gif_path: str) -> list[Image.Image]:
    """Extract every frame from a GIF file."""
    frames = []
    with Image.open(gif_path) as img:
        try:
            while True:
                frames.append(img.copy().convert("RGBA"))
                img.seek(img.tell() + 1)
        except EOFError:
            pass
    return frames


def add_epoch_banner(img: Image.Image, epoch: int, frame_idx: int,
                     total_frames: int) -> Image.Image:
    """Add a top banner showing epoch number and step progress."""
    banner_h = 24
    canvas = Image.new("RGBA", (img.width, img.height + banner_h), (25, 25, 35, 255))
    canvas.paste(img, (0, banner_h))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    label = f"Episode {epoch}   Step {frame_idx}/{total_frames - 1}"
    draw.text((8, 5), label, fill=(220, 220, 255, 255), font=font)

    # small progress bar
    bar_x, bar_w, bar_h = img.width - 80, 70, 6
    bar_y = (banner_h - bar_h) // 2
    draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
                   fill=(60, 60, 80, 255))
    filled = int(bar_w * frame_idx / max(total_frames - 1, 1))
    if filled > 0:
        draw.rectangle([bar_x, bar_y, bar_x + filled, bar_y + bar_h],
                       fill=(100, 180, 255, 255))

    return canvas


def add_epoch_divider(img: Image.Image, epoch: int) -> Image.Image:
    """A brief title card shown between epochs."""
    canvas = Image.new("RGBA", img.size, (15, 15, 25, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font_big = ImageFont.truetype("arial.ttf", 22)
        font_sm  = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font_big = ImageFont.load_default()
        font_sm  = font_big

    cx, cy = img.width // 2, img.height // 2
    text = f"Episode {epoch}"
    bbox = draw.textbbox((0, 0), text, font=font_big)
    tw = bbox[2] - bbox[0]
    draw.text((cx - tw // 2, cy - 18), text, fill=(180, 220, 255, 255), font=font_big)
    sub = "greedy policy snapshot"
    bbox2 = draw.textbbox((0, 0), sub, font=font_sm)
    sw = bbox2[2] - bbox2[0]
    draw.text((cx - sw // 2, cy + 12), sub, fill=(120, 140, 160, 255), font=font_sm)
    return canvas


def build_progress_gif(snapshot_dir: str, out_path: str,
                       step_ms: int = 200, divider_ms: int = 600):
    """
    For each ep*.gif in snapshot_dir (sorted by episode number):
      1. Show a title card  (divider_ms)
      2. Play every step of that episode  (step_ms each)
    """
    pattern = os.path.join(snapshot_dir, "ep*.gif")
    paths   = sorted(glob.glob(pattern))
    if not paths:
        print(f"No ep*.gif files found in {snapshot_dir}")
        return

    all_frames:    list[Image.Image] = []
    all_durations: list[int]         = []

    for path in paths:
        ep_num = int(os.path.splitext(os.path.basename(path))[0].replace("ep", ""))
        frames = extract_all_frames(path)
        if not frames:
            continue

        # reference size from first frame
        ref_size = frames[0].size

        # title card
        dummy = Image.new("RGBA", (ref_size[0], ref_size[1] + 24), (25, 25, 35, 255))
        divider = add_epoch_divider(dummy, ep_num).convert("RGB")
        all_frames.append(divider)
        all_durations.append(divider_ms)

        # every step of this episode
        n = len(frames)
        for i, frame in enumerate(frames):
            annotated = add_epoch_banner(frame, ep_num, i, n).convert("RGB")
            all_frames.append(annotated)
            all_durations.append(step_ms)

    if not all_frames:
        print("No frames collected.")
        return

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    all_frames[0].save(
        out_path,
        save_all=True,
        append_images=all_frames[1:],
        duration=all_durations,
        loop=0,
        optimize=False,
    )
    print(f"Saved: {out_path}  ({len(all_frames)} frames from {len(paths)} episodes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",       required=True, help="Directory containing ep*.gif snapshots")
    parser.add_argument("--out",       default="training_progress.gif")
    parser.add_argument("--step_ms",   type=int, default=200, help="ms per step frame")
    parser.add_argument("--divider_ms",type=int, default=600, help="ms for epoch title card")
    args = parser.parse_args()
    build_progress_gif(args.dir, args.out, args.step_ms, args.divider_ms)
