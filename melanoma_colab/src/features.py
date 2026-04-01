import cv2
import numpy as np
import pandas as pd
from typing import Tuple, List


def calculate_hu_moments(image_bgr: np.ndarray, n_moments: int = 4) -> np.ndarray:
    """
    Calculate first n log-transformed Hu moments from an image.
    Hu moments are rotation/scale/translation invariant shape descriptors.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    moments = cv2.moments(binary)
    hu_moments = cv2.HuMoments(moments)

    # Log-transform for numerical stability
    for i in range(7):
        hu_moments[i] = -1 * np.sign(hu_moments[i]) * np.log10(abs(hu_moments[i]) + 1e-10)

    return hu_moments.flatten()[:n_moments]


def calculate_color_histogram(image_bgr: np.ndarray, bins=(8, 8, 8)) -> np.ndarray:
    """Calculate normalized HSV color histogram."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def calculate_haralick_features(image_bgr: np.ndarray) -> np.ndarray:
    """Calculate Haralick texture features using mahotas."""
    import mahotas
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    features = mahotas.features.haralick(gray).mean(axis=0)
    return features


def extract_image_features(image_bgr: np.ndarray, n_hu: int = 4) -> np.ndarray:
    """
    Extract image-level features: first n Hu moments.
    Returns a flat numpy array of shape (n_hu,).
    """
    hu = calculate_hu_moments(image_bgr, n_moments=n_hu)
    return hu.astype(np.float32)


def encode_metadata(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], int]:
    """
    Encode metadata columns for model input:
    - One-hot encode sex (male, female)
    - One-hot encode anatom_site_general_challenge
    - Normalize age_approx to [0, 1]

    Returns:
        (df_with_encoded_columns, list_of_metadata_column_names, metadata_dim)
    """
    df = df.copy()
    metadata_cols = []

    # --- Sex: one-hot ---
    df["sex"] = df["sex"].fillna("unknown")
    sex_dummies = pd.get_dummies(df["sex"], prefix="sex", dtype=float)
    df = pd.concat([df, sex_dummies], axis=1)
    metadata_cols.extend(sex_dummies.columns.tolist())

    # --- Anatomical site: one-hot ---
    df["anatom_site_general_challenge"] = df["anatom_site_general_challenge"].fillna("unknown")
    site_dummies = pd.get_dummies(df["anatom_site_general_challenge"], prefix="site", dtype=float)
    df = pd.concat([df, site_dummies], axis=1)
    metadata_cols.extend(site_dummies.columns.tolist())

    # --- Age: normalize to [0, 1] ---
    df["age_approx"] = df["age_approx"].fillna(df["age_approx"].median())
    age_min = df["age_approx"].min()
    age_max = df["age_approx"].max()
    if age_max > age_min:
        df["age_norm"] = (df["age_approx"] - age_min) / (age_max - age_min)
    else:
        df["age_norm"] = 0.0
    metadata_cols.append("age_norm")

    metadata_dim = len(metadata_cols)
    return df, metadata_cols, metadata_dim
