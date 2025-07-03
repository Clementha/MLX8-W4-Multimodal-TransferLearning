#!/usr/bin/env python3
"""
Down-sample GoPro images for captioning.

Copies every .jpg/.jpeg/.png from SRC_DIR (recursively) into DEST_DIR,
resizing so that max(width, height) == TARGET_PX while preserving aspect
ratio.  Filenames are copied unchanged except that whitespace is removed.

Usage:
  python3 downsample_images.py
"""

from __future__ import annotations

import concurrent.futures as cf
import random
import sys
from pathlib import Path
from typing import Iterable, List

from PIL import Image  # pip install pillow  (pillow-simd is even faster)

# -------- GLOBALS ----------------------------------------------------
SRC_DIR = Path("/media/bwilliams/New Volume1/mars_hackathon")
DEST_DIR = SRC_DIR / "dwn_sampld"

TARGET_PX = 800          # longest edge after resizing
JPEG_QUALITY = 85        # output quality (1-95), 85 is a good balance

# Set to None to process every file, or an int to sample that many
SAMPLE_SIZE: int | None = None
# ---------------------------------------------------------------------


def find_images(root: Path) -> List[Path]:
  """Return a list of image paths (.jpg, .jpeg, .png) under *root*."""
  exts = {".jpg", ".jpeg", ".png"}
  return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def resize_one(src: Path, dst: Path) -> None:
  """
  Resize *src* to TARGET_PX and save to *dst*.
  Keeps original format; converts PNG with alpha to RGB PNG.
  """
  try:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
      im = im.convert("RGB")  # strips EXIF orientation but safe for GoPros
      w, h = im.size
      scale = TARGET_PX / max(w, h)
      if scale < 1:  # only shrink, never enlarge
        new_size = (int(w * scale), int(h * scale))
        im = im.resize(new_size, Image.LANCZOS)
      # Write in same format; JPEG gets quality setting
      save_kwargs = {"quality": JPEG_QUALITY} if dst.suffix.lower() in {".jpg", ".jpeg"} else {}
      im.save(dst, **save_kwargs)
  except Exception as exc:
    print(f"⚠️  Failed on {src}: {exc}", file=sys.stderr)


def strip_spaces(p: Path) -> Path:
  """Return *p* with spaces removed from the stem."""
  return p.with_stem(p.stem.replace(" ", ""))


def main() -> None:
  DEST_DIR.mkdir(parents=True, exist_ok=True)

  imgs = find_images(SRC_DIR)
  if SAMPLE_SIZE and SAMPLE_SIZE < len(imgs):
    random.shuffle(imgs)
    imgs = imgs[:SAMPLE_SIZE]

  # Map each source to destination path in DEST_DIR
  tasks = [(img, strip_spaces(DEST_DIR / img.name)) for img in imgs]

  print(f"Processing {len(tasks)} images → {DEST_DIR}")
  with cf.ThreadPoolExecutor() as pool:
    for src, dst in tasks:
      pool.submit(resize_one, src, dst)


if __name__ == "__main__":
  main()
