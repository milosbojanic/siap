import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple
import albumentations as A

from src.config import Config
from src.preprocessing import remove_hairs
from src.features import extract_image_features


class MelanomaDataset(Dataset):
    """
    Unified dataset for melanoma classification.
    Returns (image_tensor, features_tensor, label) for each sample.

    Works with both custom CNN and EfficientNet models.
    """

    def __init__(
        self,
        df,
        img_dir: str,
        config: Config,
        transforms: Optional[A.Compose] = None,
        metadata_columns: Optional[List[str]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.config = config
        self.transforms = transforms
        self.metadata_columns = metadata_columns or []

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        # Load image
        img_path = os.path.join(self.img_dir, f"{row['image_name']}.jpg")
        img_bgr = cv2.imread(img_path)

        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Resize FIRST — avoids expensive preprocessing on full-res images
        # (ISIC images are ~4000x3000, resize to target size before any processing)
        img_bgr = cv2.resize(img_bgr, (self.config.image_size, self.config.image_size))

        # Hair removal (now on small image — much faster)
        if self.config.apply_hair_removal:
            img_bgr = remove_hairs(img_bgr)

        # Extract image features (Hu moments)
        image_features = extract_image_features(img_bgr, n_hu=4)

        # Get metadata features (pre-encoded columns from df)
        if self.metadata_columns:
            meta = row[self.metadata_columns].values.astype(np.float32)
            features = np.concatenate([image_features, meta])
        else:
            features = image_features

        # Convert BGR -> RGB for transforms
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Apply albumentations transforms
        if self.transforms:
            transformed = self.transforms(image=img_rgb)
            img_tensor = transformed["image"]
        else:
            img_tensor = torch.from_numpy(
                img_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
            )

        features_tensor = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(float(row["target"]), dtype=torch.float32)

        return img_tensor, features_tensor, label


class CachedMelanomaDataset(Dataset):
    """
    Ultra-fast dataset that reads from preprocessed .npy cache.
    Use preprocess_and_cache() from src.preprocessing_cache to create the cache first.

    Much faster than MelanomaDataset because:
    - No cv2.imread() of large JPEGs
    - No resize / hair removal / Hu moments per access
    - numpy .npy files load ~10x faster than JPEG decoding
    """

    def __init__(
        self,
        df,
        cache_dir: str,
        config: Config,
        transforms: Optional[A.Compose] = None,
        metadata_columns: Optional[List[str]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.config = config
        self.transforms = transforms
        self.metadata_columns = metadata_columns or []

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_name = row["image_name"]

        # Read from cache — fast numpy load
        img_bgr = np.load(os.path.join(self.cache_dir, f"{img_name}_img.npy"))
        hu_features = np.load(os.path.join(self.cache_dir, f"{img_name}_feat.npy"))

        # Metadata features
        if self.metadata_columns:
            meta = row[self.metadata_columns].values.astype(np.float32)
            features = np.concatenate([hu_features, meta])
        else:
            features = hu_features

        # BGR -> RGB for transforms
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Apply albumentations transforms
        if self.transforms:
            transformed = self.transforms(image=img_rgb)
            img_tensor = transformed["image"]
        else:
            img_tensor = torch.from_numpy(
                img_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
            )

        features_tensor = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(float(row["target"]), dtype=torch.float32)

        return img_tensor, features_tensor, label
