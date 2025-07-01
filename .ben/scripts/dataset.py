# scripts/dataset.py
"""
Defines Flickr30kDataset for loading images and metadata as a PyTorch Dataset.
"""
from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple


class Flickr30kDataset(Dataset):
    """
    PyTorch Dataset for Flickr30k images and captions.

    Attributes:
        df (pd.DataFrame): Metadata containing filename, caption, split.
        image_dir (Path): Directory where images are stored.
        transform (Optional[Callable[[Image.Image], Image.Image]]): Optional image transform.
    """
    def __init__(
        self,
        metadata_path: Path,
        image_dir: Path,
        transform: Optional[Callable[[Image.Image], Image.Image]] = None,
    ) -> None:
        """
        Initialize dataset.

        Args:
            metadata_path: Path to metadata.parquet.
            image_dir: Path to directory with images.
            transform: Optional PIL image transform.
        """
        self.df: pd.DataFrame = pd.read_parquet(metadata_path)
        self.image_dir: Path = Path(image_dir)
        self.transform = transform

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, str, str]:
        """
        Get the image, filename, and caption at index.

        Returns:
            Tuple of (PIL.Image RGB, filename, caption string).
        """
        row = self.df.iloc[idx]
        img_path = self.image_dir / row["filename"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, row["filename"], row["caption"]
