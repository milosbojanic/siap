"""
Offline preprocessing: resize + hair removal + Hu moments for all images.
Saves results as .npy files for ultra-fast loading during training.
Supports resume — if interrupted, continues from where it left off.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

from src.preprocessing import remove_hairs
from src.features import extract_image_features


def preprocess_and_cache(df, img_dir, cache_dir, image_size, apply_hair_removal=True):
    """
    Preprocess all images and save to cache_dir as .npy files.

    For each image saves:
      - {image_name}_img.npy   — resized+cleaned BGR uint8 array (H, W, 3)
      - {image_name}_feat.npy  — Hu moments float32 array (4,)

    Args:
        df: DataFrame with 'image_name' column
        img_dir: directory containing original .jpg images
        cache_dir: where to save preprocessed .npy files
        image_size: target size (e.g. 128 for CNN, 224 for EfficientNet)
        apply_hair_removal: whether to apply hair removal
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Check if already done
    done_marker = os.path.join(cache_dir, f"_done_{image_size}_{apply_hair_removal}.marker")
    if os.path.exists(done_marker):
        print(f"Cache already exists at {cache_dir} — skipping.")
        return

    image_names = df["image_name"].unique()
    skipped = 0

    for img_name in tqdm(image_names, desc=f"Preprocessing {image_size}x{image_size}"):
        npy_img_path = os.path.join(cache_dir, f"{img_name}_img.npy")
        npy_feat_path = os.path.join(cache_dir, f"{img_name}_feat.npy")

        # Resume support — skip already processed
        if os.path.exists(npy_img_path) and os.path.exists(npy_feat_path):
            skipped += 1
            continue

        img_path = os.path.join(img_dir, f"{img_name}.jpg")
        img_bgr = cv2.imread(img_path)

        if img_bgr is None:
            print(f"WARNING: Could not read {img_path}, skipping.")
            continue

        # Resize first (much faster preprocessing on small images)
        img_bgr = cv2.resize(img_bgr, (image_size, image_size))

        # Hair removal on resized image
        if apply_hair_removal:
            img_bgr = remove_hairs(img_bgr)

        # Extract Hu moments
        hu_features = extract_image_features(img_bgr, n_hu=4)

        # Save as numpy
        np.save(npy_img_path, img_bgr)
        np.save(npy_feat_path, hu_features)

    if skipped > 0:
        print(f"Resumed: {skipped} images already cached, processed {len(image_names) - skipped} new.")

    # Write marker file
    with open(done_marker, "w") as f:
        f.write(f"size={image_size}, hair_removal={apply_hair_removal}, n_images={len(image_names)}\n")

    print(f"Done! {len(image_names)} images cached in {cache_dir}")
