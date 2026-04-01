from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import os

# Project root = parent of src/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    """Central configuration for the melanoma classification pipeline."""

    # Data paths (relative to project root, resolved in __post_init__)
    train_csv: str = "help/subset_train.csv"
    test_csv: str = "help/subset_test.csv"
    train_dir: str = "help/subset_train"
    test_dir: str = "help/subset_test"
    tone_mapping_csv: Optional[str] = "help/tone_mapping.csv"

    # Quick-test: limit dataset size (None = use all)
    subset_size: Optional[int] = None

    # Image
    image_size: int = 224

    # Training
    batch_size: int = 16
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_folds: int = 5
    num_workers: int = 0  # 0 for Windows compatibility
    patience: int = 5  # early stopping patience

    # Model
    model_type: str = "cnn"  # "cnn" or "efficientnet"
    use_metadata: bool = True
    apply_hair_removal: bool = True

    # Loss function: "bce" (weighted BCE) or "focal" (Focal Loss)
    loss_type: str = "bce"

    # Augmentation
    augment_train: bool = True

    # Paths
    model_save_dir: str = "models"
    cache_dir: Optional[str] = None  # None = no cache, preprocess on-the-fly

    # Device
    device: str = "auto"  # "auto", "cuda", "cpu"

    # Computed at runtime (do not set manually)
    feature_dim: int = 0
    metadata_columns: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Resolve all relative paths to absolute paths based on project root."""
        path_attrs = ("train_csv", "test_csv", "train_dir", "test_dir",
                      "tone_mapping_csv", "model_save_dir")
        for attr in path_attrs:
            val = getattr(self, attr)
            if val is not None and not os.path.isabs(val):
                setattr(self, attr, str(_PROJECT_ROOT / val))

    @classmethod
    def quick_test(cls) -> "Config":
        """50 images, 2 epochs, 2 folds -- runs in ~1-2 minutes on CPU."""
        return cls(
            subset_size=50,
            epochs=2,
            num_folds=2,
            batch_size=8,
            image_size=64,
            patience=2,
        )

    @classmethod
    def small_run(cls) -> "Config":
        """All 1,482 images, 5 epochs, 3 folds."""
        return cls(
            epochs=5,
            num_folds=3,
            batch_size=16,
        )

    @classmethod
    def full_run(cls) -> "Config":
        """All data, 20 epochs, 5-fold CV."""
        return cls(
            epochs=20,
            num_folds=5,
            batch_size=16,
        )

    @classmethod
    def colab(cls) -> "Config":
        """Full training on Google Colab with GPU and Kaggle dataset."""
        cfg = cls(
            epochs=20,
            num_folds=5,
            batch_size=32,
            num_workers=4,
            device="cuda",
            train_dir="/content/data/train",
            train_csv="/content/data/train.csv",
            test_dir="/content/data/test",
            test_csv="/content/data/test.csv",
            tone_mapping_csv=None,
            model_save_dir="/content/drive/MyDrive/melanoma_results/models",
        )
        return cfg

    def get_device(self):
        import torch
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def ensure_dirs(self):
        """Create output directories if they don't exist."""
        os.makedirs(self.model_save_dir, exist_ok=True)
