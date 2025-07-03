#!/usr/bin/env python3
"""
Create ~1 GB zip archives from down-sampled images for easy upload.

This script:
  1. Finds all .jpg/.jpeg/.png files under a source directory.
  2. Groups them into chunks whose total size does not exceed CHUNK_SIZE.
  3. Writes each chunk into a separate ZIP archive, preserving relative paths.
  4. Names archives sequentially as dataset_part001.zip, dataset_part002.zip, etc.

Usage:
  python3 chunk_and_zip.py

Dependencies:
  - Python 3.8+
  - No external libraries beyond the standard library.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import List, Iterable

# ------------ CONFIGURATION ------------

# Directory containing down-sampled images
SRC_DIR = Path("/media/bwilliams/New Volume1/mars_hackathon/dwn_sampld")

# Where to write the zip archives
OUTPUT_DIR = Path("/media/bwilliams/New Volume1/mars_hackathon/zipped")

# Maximum bytes per archive (~1 GiB)
CHUNK_SIZE = 1 * 1024 ** 3  # 1 GiB

# File extensions to include
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# ------------ FUNCTIONS ------------

def chunk_files(
    files: Iterable[Path], 
    chunk_size: int
) -> List[List[Path]]:
    """
    Group files into lists where each group's cumulative size ≤ chunk_size.

    Args:
        files: Iterable of file paths.
        chunk_size: Maximum total bytes per group.

    Returns:
        A list of file-path lists.
    """
    chunks: List[List[Path]] = []
    current: List[Path] = []
    total = 0

    for f in files:
        size = f.stat().st_size
        # If single file > chunk, put it alone
        if size > chunk_size:
            if current:
                chunks.append(current)
                current = []
                total = 0
            chunks.append([f])
            continue
        # Start new chunk if exceeding
        if total + size > chunk_size:
            chunks.append(current)
            current = [f]
            total = size
        else:
            current.append(f)
            total += size

    if current:
        chunks.append(current)

    return chunks


def create_archives(chunks: List[List[Path]]) -> None:
    """
    Write each chunk of files into a ZIP archive.

    Archives are named dataset_partXXX.zip in numeric order.

    Args:
        chunks: List of file-path lists to archive.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for idx, chunk in enumerate(chunks, start=1):
        zip_path = OUTPUT_DIR / f"dataset_part{idx:03d}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for f in chunk:
                # Preserve the filename at top level in the archive
                arcname = f.name
                zf.write(f, arcname)
        print(f"Created {zip_path}")


def main() -> None:
    """Entry point: find images, chunk them, and create zip archives."""
    # Gather all target images
    images = [
        p for p in SRC_DIR.rglob("*") 
        if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    ]

    if not images:
        print(f"No images found in {SRC_DIR}")
        return

    # Group into ~1 GB chunks
    chunks = chunk_files(images, CHUNK_SIZE)
    print(f"Splitting {len(images)} images into {len(chunks)} archives.")

    # Create the zip files
    create_archives(chunks)


if __name__ == "__main__":
    main()


