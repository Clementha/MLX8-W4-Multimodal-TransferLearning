#!/usr/bin/env python3
"""
count_reef_images.py

Count how many reef images exist in a given directory.
"""
import sys
from pathlib import Path

def count_images(directory: Path) -> int:
    """
    Count image files in the directory matching common extensions.

    Args:
        directory: Path to the images folder.
    Returns:
        Number of image files found.
    """
    # common image extensions
    patterns = ["*.jpg", "*.jpeg", "*.png"]
    total = 0
    for pattern in patterns:
        total += len(list(directory.glob(pattern)))
    return total


def main() -> None:
    """
    Entry point: reads a directory path from argv or uses the default reef image dir.
    Prints the count of image files.
    """
    # default path if none provided
    default_dir = (
        Path(__file__).resolve().parent.parent
        / "data" / "reef_data" / "images" / "images"
    )
    dir_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir

    if not dir_path.is_dir():
        print(f"Error: not a directory: {dir_path}", file=sys.stderr)
        sys.exit(1)

    total = count_images(dir_path)
    print(total)

if __name__ == "__main__":
    main()
