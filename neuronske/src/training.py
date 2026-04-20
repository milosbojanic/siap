import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from typing import Dict, List, Tuple
import pandas as pd

from src.config import Config
from src.models import create_model
from src.dataset import MelanomaDataset, CachedMelanomaDataset
from src.augmentation import get_train_transforms, get_val_transforms
from src.data_utils import get_kfold_splits, get_class_weights, create_subset


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced binary classification.

    Focuses on hard-to-classify examples by down-weighting easy negatives.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: weighting factor for positive class (default 0.25)
        gamma: focusing parameter — higher = more focus on hard examples (default 2.0)
        pos_weight: optional class weight tensor (like BCEWithLogitsLoss)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # p_t = probability of correct class
        p_t = targets * probs + (1 - targets) * (1 - probs)
        # alpha_t = alpha for positives, (1-alpha) for negatives
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # BCE per element (numerically stable via log-sum-exp)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        loss = alpha_t * focal_weight * bce

        # Apply pos_weight if provided (extra boost for minority class)
        if self.pos_weight is not None:
            weight = targets * self.pos_weight + (1 - targets)
            loss = loss * weight

        return loss.mean()


def create_criterion(config: Config, pos_weight: torch.Tensor) -> nn.Module:
    """Create loss function based on config.loss_type."""
    if config.loss_type == "focal":
        return FocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight)
    else:
        # Default: weighted BCE
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, features, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images, features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Validation pass. Returns (avg_loss, accuracy, all_labels, all_probs)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, features, labels in tqdm(loader, desc="Val", leave=False):
            images = images.to(device)
            features = features.to(device)
            labels = labels.to(device)

            logits = model(images, features)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_labels), np.array(all_probs)


def train_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: Config,
    fold_idx: int,
) -> Dict:
    """
    Train a single fold. Returns dict with:
    - train_losses, val_losses, train_accs, val_accs (per epoch)
    - best_val_auc
    - val_labels, val_probs (for evaluation)
    - val_df (for fairness analysis)
    - model_path (saved best model)
    """
    device = config.get_device()
    config.ensure_dirs()

    # Create datasets
    train_transforms = get_train_transforms(config.image_size) if config.augment_train else get_val_transforms(config.image_size)
    val_transforms = get_val_transforms(config.image_size)

    # Use cached dataset if cache_dir is set (much faster — no per-image preprocessing)
    if config.cache_dir and os.path.isdir(config.cache_dir):
        train_dataset = CachedMelanomaDataset(
            train_df, config.cache_dir, config,
            transforms=train_transforms,
            metadata_columns=config.metadata_columns,
        )
        val_dataset = CachedMelanomaDataset(
            val_df, config.cache_dir, config,
            transforms=val_transforms,
            metadata_columns=config.metadata_columns,
        )
    else:
        train_dataset = MelanomaDataset(
            train_df, config.train_dir, config,
            transforms=train_transforms,
            metadata_columns=config.metadata_columns,
        )
        val_dataset = MelanomaDataset(
            val_df, config.train_dir, config,
            transforms=val_transforms,
            metadata_columns=config.metadata_columns,
        )

    pin = device.type == 'cuda'
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=pin,
    )

    # Create model and optimizer
    model = create_model(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    pos_weight = get_class_weights(train_df).to(device)
    criterion = create_criterion(config, pos_weight)

    # Training loop with early stopping
    best_val_auc = 0.0
    patience_counter = 0
    model_path = os.path.join(config.model_save_dir, f"{config.model_type}_fold{fold_idx}.pth")

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_labels, best_val_probs = None, None

    for epoch in range(config.epochs):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc, val_labels, val_probs = validate(model, val_loader, criterion, device)

        # Compute AUC
        try:
            val_auc = roc_auc_score(val_labels, val_probs)
        except ValueError:
            val_auc = 0.5  # Only one class present

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"  Fold {fold_idx} | Epoch {epoch+1}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_labels = val_labels
            best_val_probs = val_probs
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    return {
        "fold_idx": fold_idx,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "best_val_auc": best_val_auc,
        "val_labels": best_val_labels,
        "val_probs": best_val_probs,
        "val_df": val_df,
        "model_path": model_path,
    }


def run_cross_validation(
    df: pd.DataFrame,
    config: Config,
) -> Dict:
    """
    Full k-fold cross-validation.
    Returns:
    - per_fold_results: list of fold result dicts
    - oof_labels, oof_probs: out-of-fold predictions
    - mean_auc, std_auc
    """
    # Subset if quick_test
    if config.subset_size is not None:
        df = create_subset(df, config.subset_size)
        print(f"Using subset of {len(df)} samples")

    print(f"\nTarget distribution: {df['target'].value_counts().to_dict()}")
    print(f"Running {config.num_folds}-fold CV with {config.model_type} model\n")

    splits = get_kfold_splits(df, n_folds=config.num_folds)

    per_fold_results = []
    all_val_labels = []
    all_val_probs = []
    all_val_dfs = []

    for fold_idx, (train_df, val_df) in enumerate(splits):
        print(f"\n--- Fold {fold_idx + 1}/{config.num_folds} ---")
        print(f"  Train: {len(train_df)} samples ({train_df['target'].sum()} malignant)")
        print(f"  Val:   {len(val_df)} samples ({val_df['target'].sum()} malignant)")

        result = train_fold(train_df, val_df, config, fold_idx)
        per_fold_results.append(result)

        if result["val_labels"] is not None:
            all_val_labels.append(result["val_labels"])
            all_val_probs.append(result["val_probs"])
            all_val_dfs.append(result["val_df"])

    # Aggregate out-of-fold predictions
    oof_labels = np.concatenate(all_val_labels) if all_val_labels else np.array([])
    oof_probs = np.concatenate(all_val_probs) if all_val_probs else np.array([])
    oof_df = pd.concat(all_val_dfs, ignore_index=True) if all_val_dfs else pd.DataFrame()

    aucs = [r["best_val_auc"] for r in per_fold_results]
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    print(f"\n{'='*50}")
    print(f"Cross-Validation Results ({config.model_type})")
    print(f"Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"Per-fold AUCs: {[f'{a:.4f}' for a in aucs]}")
    print(f"{'='*50}")

    return {
        "per_fold_results": per_fold_results,
        "oof_labels": oof_labels,
        "oof_probs": oof_probs,
        "oof_df": oof_df,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
    }
