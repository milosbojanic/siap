import pandas as pd
import numpy as np
import torch
from typing import List, Tuple, Optional
from sklearn.model_selection import StratifiedGroupKFold

from src.config import Config
from src.features import encode_metadata


def load_and_prepare_data(config: Config) -> pd.DataFrame:
    """
    Load training CSV, merge skin tone data from tone_mapping.csv,
    encode metadata, and update config with feature dimensions.

    Returns prepared DataFrame with all encoded columns.
    """
    df = pd.read_csv(config.train_csv)

    # Merge skin tone from tone_mapping.csv (subset_train.csv has empty skin_tone_index)
    if config.tone_mapping_csv:
        tone_df = pd.read_csv(config.tone_mapping_csv)
        # Drop the existing empty skin_tone_index column
        if "skin_tone_index" in df.columns:
            df = df.drop(columns=["skin_tone_index"])
        df = df.merge(
            tone_df[["image_name", "skin_tone_index"]],
            on="image_name",
            how="left",
        )
    # Fill missing skin tones with -1
    if "skin_tone_index" in df.columns:
        df["skin_tone_index"] = df["skin_tone_index"].fillna(-1).astype(int)

    # Encode metadata (one-hot sex, site; normalize age)
    if config.use_metadata:
        df, metadata_cols, metadata_dim = encode_metadata(df)
        config.metadata_columns = metadata_cols
        config.feature_dim = 4 + metadata_dim  # 4 Hu moments + metadata
    else:
        config.metadata_columns = []
        config.feature_dim = 4  # Only Hu moments

    return df


def load_test_data(config: Config, train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load test CSV and encode metadata consistently with training data.
    Uses the same columns that were created during training encoding.
    """
    df = pd.read_csv(config.test_csv)

    # Merge skin tone
    if config.tone_mapping_csv:
        tone_df = pd.read_csv(config.tone_mapping_csv)
        if "skin_tone_index" in df.columns:
            df = df.drop(columns=["skin_tone_index"])
        df = df.merge(
            tone_df[["image_name", "skin_tone_index"]],
            on="image_name",
            how="left",
        )
    if "skin_tone_index" in df.columns:
        df["skin_tone_index"] = df["skin_tone_index"].fillna(-1).astype(int)

    # Encode metadata
    if config.use_metadata:
        df, _, _ = encode_metadata(df)
        # Ensure all metadata columns from training exist
        for col in config.metadata_columns:
            if col not in df.columns:
                df[col] = 0.0

    return df


def create_subset(
    df: pd.DataFrame,
    n: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample n rows, stratified by target, respecting patient grouping.
    All images from selected patients are included until we reach >= n.
    """
    rng = np.random.RandomState(random_state)

    # Shuffle patients
    patients = df["patient_id"].unique()
    rng.shuffle(patients)

    selected_rows = []
    for pid in patients:
        patient_rows = df[df["patient_id"] == pid]
        selected_rows.append(patient_rows)
        total = sum(len(r) for r in selected_rows)
        if total >= n:
            break

    subset = pd.concat(selected_rows, ignore_index=True)
    return subset


def get_kfold_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    random_state: int = 42,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Patient-level stratified k-fold split.
    Uses StratifiedGroupKFold to ensure:
    - No patient appears in both train and val
    - Each fold has approximately the same malignant/benign ratio

    Returns list of (train_df, val_df) tuples.
    """
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    splits = []
    X = df.index.values
    y = df["target"].values
    groups = df["patient_id"].values

    for train_idx, val_idx in skf.split(X, y, groups):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        splits.append((train_df, val_df))

    return splits


def get_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """
    Compute pos_weight for BCEWithLogitsLoss to handle class imbalance.
    pos_weight = num_negative / num_positive
    """
    n_neg = (df["target"] == 0).sum()
    n_pos = (df["target"] == 1).sum()
    if n_pos == 0:
        return torch.tensor([1.0])
    return torch.tensor([n_neg / n_pos], dtype=torch.float32)
