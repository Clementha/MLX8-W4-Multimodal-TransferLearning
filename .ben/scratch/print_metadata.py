#!/usr/bin/env python3
"""
Print the first 10 rows of the Flickr30k metadata Parquet.
"""

from pathlib import Path
import pandas as pd


def main() -> None:
    """
    Load metadata.parquet and print the first 10 rows in full.
    """
    # adjust this path if you move the script
    metadata_path = Path(__file__).resolve().parent.parent / "data" / "flickr30k" / "metadata.parquet"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    # read parquet (requires pandas + pyarrow)
    df = pd.read_parquet(metadata_path)
    # print all columns for the first 10 entries
    with pd.option_context("display.max_columns", None):
        print(df.head(10))


if __name__ == "__main__":
    main()
