import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from typing import List, Dict, Optional
import os
import cv2

from src.preprocessing import remove_hairs


def plot_preprocessing_examples(img_dir: str, n: int = 4, seed: int = 42) -> plt.Figure:
    """Show n original vs hair-removed images side by side."""
    rng = np.random.RandomState(seed)
    all_imgs = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
    selected = rng.choice(all_imgs, size=min(n, len(all_imgs)), replace=False)

    fig, axes = plt.subplots(n, 2, figsize=(8, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, fname in enumerate(selected):
        img = cv2.imread(os.path.join(img_dir, fname))
        cleaned = remove_hairs(img)

        axes[i, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"Original: {fname[:15]}...")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title("Nakon uklanjanja dlaka")
        axes[i, 1].axis("off")

    fig.suptitle("Preprocesiranje: uklanjanje dlaka (Black-Hat + Telea)", fontsize=13)
    plt.tight_layout()
    return fig


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    fold_idx: Optional[int] = None,
) -> plt.Figure:
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-o", label="Train Loss", markersize=4)
    ax.plot(epochs, val_losses, "r-o", label="Val Loss", markersize=4)
    ax.set_xlabel("Epoha")
    ax.set_ylabel("Loss")
    title = "Krive gubitka"
    if fold_idx is not None:
        title += f" (Fold {fold_idx})"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_all_folds_losses(fold_results: List[Dict]) -> plt.Figure:
    """Plot loss curves for all folds in one figure."""
    n_folds = len(fold_results)
    fig, axes = plt.subplots(1, n_folds, figsize=(5 * n_folds, 4), squeeze=False)

    for i, result in enumerate(fold_results):
        ax = axes[0, i]
        epochs = range(1, len(result["train_losses"]) + 1)
        ax.plot(epochs, result["train_losses"], "b-", label="Train")
        ax.plot(epochs, result["val_losses"], "r-", label="Val")
        ax.set_title(f"Fold {i} (AUC={result['best_val_auc']:.3f})")
        ax.set_xlabel("Epoha")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Krive gubitka po foldovima", fontsize=13)
    plt.tight_layout()
    return fig


def plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    fold_results: Optional[List[Dict]] = None,
    label: str = "Model",
) -> plt.Figure:
    """
    Plot ROC curve. Optionally overlay per-fold ROC curves.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # Per-fold curves (lighter)
    if fold_results:
        for i, result in enumerate(fold_results):
            if result["val_labels"] is not None and len(np.unique(result["val_labels"])) > 1:
                fpr, tpr, _ = roc_curve(result["val_labels"], result["val_probs"])
                ax.plot(fpr, tpr, alpha=0.3, linewidth=1,
                        label=f"Fold {i} (AUC={result['best_val_auc']:.3f})")

    # Main OOF curve
    if len(np.unique(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, "b-", linewidth=2,
                label=f"{label} OOF (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC kriva")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_roc_comparison(
    results_dict: Dict[str, Dict],
) -> plt.Figure:
    """Plot ROC curves for multiple models on the same figure."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["blue", "red", "green", "orange"]

    for idx, (name, res) in enumerate(results_dict.items()):
        labels, probs = res["oof_labels"], res["oof_probs"]
        if len(np.unique(labels)) > 1:
            fpr, tpr, _ = roc_curve(labels, probs)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[idx % len(colors)], linewidth=2,
                    label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Poredjenje ROC krivih")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    title: str = "Matrica konfuzije",
) -> plt.Figure:
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
        ax=ax,
    )
    ax.set_xlabel("Predikcija")
    ax.set_ylabel("Stvarna vrednost")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_fairness_bars(
    fairness_df,
    group_col: str = "group",
    title: str = "Equalized Odds analiza",
) -> plt.Figure:
    """Plot TPR and FPR per group as grouped bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    groups = fairness_df[group_col].astype(str)
    x = np.arange(len(groups))
    width = 0.5

    # TPR
    axes[0].bar(x, fairness_df["tpr"], width, color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(groups, rotation=45, ha="right")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].set_title("TPR po grupama")
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(fairness_df["tpr"]):
        if not np.isnan(v):
            axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    # FPR
    axes[1].bar(x, fairness_df["fpr"], width, color="coral")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(groups, rotation=45, ha="right")
    axes[1].set_ylabel("False Positive Rate")
    axes[1].set_title("FPR po grupama")
    axes[1].set_ylim(0, max(fairness_df["fpr"].max() * 1.3, 0.1))
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(fairness_df["fpr"]):
        if not np.isnan(v):
            axes[1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    return fig


def plot_fold_comparison(fold_results: List[Dict]) -> plt.Figure:
    """Bar chart comparing AUC across folds."""
    fig, ax = plt.subplots(figsize=(7, 4))
    folds = [f"Fold {r['fold_idx']}" for r in fold_results]
    aucs = [r["best_val_auc"] for r in fold_results]

    bars = ax.bar(folds, aucs, color="steelblue")
    ax.axhline(y=np.mean(aucs), color="red", linestyle="--",
               label=f"Prosek: {np.mean(aucs):.4f}")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Poredjenje AUC-ROC po foldovima")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    return fig
