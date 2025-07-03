#!/usr/bin/env python3
"""
Move images from subfolders into the parent directory,
adding a prefix to avoid name clashes.
"""

import shutil
from pathlib import Path

# === CONFIGURATION ===
PARENT_DIR = Path("/media/bwilliams/New Volume1/mars_hackathon")
TARGET_DIR = PARENT_DIR  # where all renamed images go
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}  # add more if needed

def make_prefix_mapping(base_dir: Path, prefix_char: str, match_prefix: str):
    """
    Returns a dict mapping each matching subfolder to its prefix,
    enumerated in alphabetical order.
    """
    folders = sorted(
        [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith(match_prefix)]
    )
    return {folder: f"{prefix_char}{i+1}" for i, folder in enumerate(folders)}

def move_images(mapping: dict[str, str]):
    """
    For each folder in mapping, move its images to TARGET_DIR,
    renaming them with <prefix>_<original_name_without_spaces><ext>.
    """
    for folder, prefix in mapping.items():
        for img in folder.rglob("*"):
            if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
                cleaned = img.stem.replace(" ", "")
                new_name = f"{prefix}_{cleaned}{img.suffix}"
                dest = TARGET_DIR / new_name
                print(f"Moving {img} → {dest}")
                shutil.move(str(img), str(dest))

def main():
    # 1) healthy & degraded in the main parent dir
    healthy_map = make_prefix_mapping(PARENT_DIR, "h", "healthy_")
    degraded_map = make_prefix_mapping(PARENT_DIR, "d", "degraded_")

    # 2) only year3_restored in the restored dir
    restored_map = make_prefix_mapping(PARENT_DIR, "r", "year3_restored_")

    # 3) do the moves
    move_images(healthy_map)
    move_images(degraded_map)
    move_images(restored_map)

if __name__ == "__main__":
    main()
