import os

import cairosvg
from PIL import Image

import ipdb


def prompt_requires_image(struct) -> bool:
    """Return whether the prompt structure contains an image component."""
    if isinstance(struct, dict):
        if struct.get("type") == "image":
            return True

        return any(
            prompt_requires_image(value)
            for value in struct.values()
        )

    if isinstance(struct, list):
        return any(
            prompt_requires_image(item)
            for item in struct
        )

    return False


def print_image_info(path: str):
    """Print basic metadata about an image file."""
    im = Image.open(path)

    width, height = im.size
    image_format = getattr(im, "format", None) or "unknown"
    mode = im.mode
    n_frames = getattr(im, "n_frames", 1)

    print(path)
    print(f"format: {image_format}")
    print(f"mode:   {mode}")
    print(
        f"size:   {width}×{height} "
        f"({width * height / 1_000_000:.2f} MP)"
    )
    print(f"frames: {n_frames}")

    return width, height, image_format, mode, n_frames


def preprocess_img(path, max_side=1024):
    """
    Preprocess an image before model inference.

    SVG images are converted to PNG.
    Images are downscaled to avoid memory spikes while preserving
    their aspect ratio, and are converted to RGB.
    """
    image_path = path

    if path.lower().endswith((".djvu", ".djv", ".pdf")):
        return None

    if path.lower().endswith(".svg"):
        new_path = os.path.splitext(path)[0] + ".png"

        cairosvg.svg2png(
            url=os.path.abspath(path),
            write_to=new_path,
            output_width=max_side,
            unsafe=True,
        )

        image_path = new_path

    im = Image.open(image_path)

    if getattr(im, "is_animated", False):
        im.seek(0)

    width, height = im.size
    longest_side = max(width, height)

    im = im.convert("RGB")

    if longest_side > max_side:
        scale_factor = longest_side / max_side

        im = im.resize(
            (
                max(1, int(width / scale_factor)),
                max(1, int(height / scale_factor)),
            ),
            Image.BOX,
        )

    return im
