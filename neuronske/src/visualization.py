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


def plot_model_disagreement(
    results_dict: Dict[str, Dict],
    threshold: Optional[float] = None,
) -> plt.Figure:
    """
    Analyze and visualize examples where models disagree.

    Shows a table/chart of disagreement patterns between models:
    - How many examples each pair of models disagree on
    - Which model is correct more often in disagreement cases

    Args:
        results_dict: {model_name: {"oof_labels": array, "oof_probs": array}}
        threshold: classification threshold (default 0.5)
    """
    from sklearn.metrics import roc_curve
    import pandas as pd

    if threshold is None:
        threshold = 0.5

    model_names = list(results_dict.keys())
    labels = list(results_dict.values())[0]["oof_labels"]

    # Get predictions for each model (use optimal threshold per model)
    preds_dict = {}
    thresholds_dict = {}
    for name, res in results_dict.items():
        fpr, tpr, threshs = roc_curve(res["oof_labels"], res["oof_probs"])
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        opt_thresh = threshs[best_idx]
        thresholds_dict[name] = opt_thresh
        preds_dict[name] = (res["oof_probs"] >= opt_thresh).astype(int)

    n_models = len(model_names)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Disagreement matrix (pairwise)
    disagree_matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            disagree_matrix[i, j] = np.sum(
                preds_dict[model_names[i]] != preds_dict[model_names[j]]
            )

    sns.heatmap(
        disagree_matrix.astype(int), annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=model_names, yticklabels=model_names,
        ax=axes[0],
    )
    axes[0].set_title("Broj neslaganja izmedju modela")

    # Panel 2: When models disagree, who is correct?
    if n_models >= 2:
        correct_counts = {}
        for i, name_a in enumerate(model_names):
            for j, name_b in enumerate(model_names):
                if i >= j:
                    continue
                disagree_mask = preds_dict[name_a] != preds_dict[name_b]
                n_disagree = disagree_mask.sum()
                if n_disagree == 0:
                    continue
                a_correct = np.sum((preds_dict[name_a][disagree_mask] == labels[disagree_mask]))
                b_correct = np.sum((preds_dict[name_b][disagree_mask] == labels[disagree_mask]))
                pair = f"{name_a}\nvs\n{name_b}"
                correct_counts[pair] = {name_a: a_correct, name_b: b_correct}

        if correct_counts:
            pairs = list(correct_counts.keys())
            x = np.arange(len(pairs))
            width = 0.35

            for pair in pairs:
                names = list(correct_counts[pair].keys())
                vals = list(correct_counts[pair].values())
                axes[1].bar(x[pairs.index(pair)] - width/2, vals[0], width,
                           label=names[0] if pairs.index(pair) == 0 else "", color="steelblue")
                axes[1].bar(x[pairs.index(pair)] + width/2, vals[1], width,
                           label=names[1] if pairs.index(pair) == 0 else "", color="coral")
                axes[1].text(x[pairs.index(pair)] - width/2, vals[0] + 5, str(vals[0]),
                           ha="center", fontsize=9)
                axes[1].text(x[pairs.index(pair)] + width/2, vals[1] + 5, str(vals[1]),
                           ha="center", fontsize=9)

            axes[1].set_xticks(x)
            axes[1].set_xticklabels(pairs, fontsize=8)
            axes[1].set_ylabel("Broj tacnih predikcija")
            axes[1].set_title("Ko je u pravu kad se modeli ne slazu?")
            axes[1].legend(model_names[:2])
            axes[1].grid(axis="y", alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, "Nema neslaganja", ha="center", va="center",
                        fontsize=14, transform=axes[1].transAxes)
    else:
        axes[1].text(0.5, 0.5, "Potrebna barem 2 modela", ha="center", va="center",
                    fontsize=14, transform=axes[1].transAxes)

    fig.suptitle("Analiza neslaganja modela", fontsize=14)
    plt.tight_layout()
    return fig


def plot_error_analysis(
    df,
    labels: np.ndarray,
    probs: np.ndarray,
    img_dir: str,
    model_name: str = "Model",
    n: int = 4,
    threshold: Optional[float] = None,
) -> plt.Figure:
    """
    Show misclassified examples: False Negatives (missed melanomas) and
    False Positives (benign classified as malignant).

    Args:
        df: DataFrame with image_name column (aligned with labels/probs)
        labels: true labels
        probs: predicted probabilities
        img_dir: path to image directory
        model_name: model name for title
        n: number of examples per category
        threshold: classification threshold (uses optimal if None)
    """
    from sklearn.metrics import roc_curve

    if threshold is None:
        fpr, tpr, threshs = roc_curve(labels, probs)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        threshold = threshs[best_idx]

    preds = (probs >= threshold).astype(int)

    # False Negatives: actual=1, pred=0 (DANGEROUS — missed melanomas)
    fn_mask = (labels == 1) & (preds == 0)
    fn_indices = np.where(fn_mask)[0]

    # False Positives: actual=0, pred=1
    fp_mask = (labels == 0) & (preds == 1)
    fp_indices = np.where(fp_mask)[0]

    # Sort by confidence (FN: highest prob that was still below threshold, FP: highest prob)
    fn_sorted = fn_indices[np.argsort(-probs[fn_indices])][:n]
    fp_sorted = fp_indices[np.argsort(-probs[fp_indices])][:n]

    n_fn = len(fn_sorted)
    n_fp = len(fp_sorted)
    n_cols = max(n_fn, n_fp, 1)

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    # Row 1: False Negatives
    for i in range(n_cols):
        ax = axes[0, i]
        if i < n_fn:
            idx = fn_sorted[i]
            img_name = df.iloc[idx]["image_name"]
            img_path = os.path.join(img_dir, f"{img_name}.jpg")
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                ax.imshow(img)
            ax.set_title(f"FN: prob={probs[idx]:.3f}\n{img_name[:20]}", fontsize=9, color="red")
        ax.axis("off")

    # Row 2: False Positives
    for i in range(n_cols):
        ax = axes[1, i]
        if i < n_fp:
            idx = fp_sorted[i]
            img_name = df.iloc[idx]["image_name"]
            img_path = os.path.join(img_dir, f"{img_name}.jpg")
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                ax.imshow(img)
            ax.set_title(f"FP: prob={probs[idx]:.3f}\n{img_name[:20]}", fontsize=9, color="orange")
        ax.axis("off")

    fig.suptitle(
        f"Analiza gresaka — {model_name} (prag={threshold:.3f})\n"
        f"Gore: Propusteni melanomi (FN={fn_mask.sum()}) | "
        f"Dole: Lazni pozitivi (FP={fp_mask.sum()})",
        fontsize=12
    )
    plt.tight_layout()
    return fig
