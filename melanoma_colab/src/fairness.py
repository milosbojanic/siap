import numpy as np
import pandas as pd
from typing import Dict


def bin_age_groups(ages: pd.Series) -> pd.Series:
    """Bin ages into groups: <30, 30-44, 45-59, 60-74, 75+."""
    bins = [0, 30, 45, 60, 75, 200]
    labels = ["<30", "30-44", "45-59", "60-74", "75+"]
    return pd.cut(ages, bins=bins, labels=labels, right=False)


def bin_skin_tones(tones: pd.Series) -> pd.Series:
    """
    Group skin tone indices into categories:
    Light (0-3), Medium (4-6), Dark (7-9).
    """
    def _map(t):
        if t < 0:
            return "Unknown"
        elif t <= 3:
            return "Light (0-3)"
        elif t <= 6:
            return "Medium (4-6)"
        else:
            return "Dark (7-9)"

    return tones.map(_map)


def compute_equalized_odds(
    df: pd.DataFrame,
    labels: np.ndarray,
    preds: np.ndarray,
    group_column: str,
) -> pd.DataFrame:
    """
    Compute True Positive Rate (TPR) and False Positive Rate (FPR) per group.

    Equalized Odds requires that TPR and FPR are equal across all groups.

    Returns DataFrame with columns:
    [group, tpr, fpr, n_samples, n_positive, n_negative]
    """
    temp = df.copy()
    temp["label"] = labels
    temp["pred"] = preds

    results = []
    for group_name, group_df in temp.groupby(group_column):
        y_true = group_df["label"].values
        y_pred = group_df["pred"].values

        n_pos = (y_true == 1).sum()
        n_neg = (y_true == 0).sum()

        # TPR = TP / (TP + FN)
        if n_pos > 0:
            tpr = ((y_true == 1) & (y_pred == 1)).sum() / n_pos
        else:
            tpr = float("nan")

        # FPR = FP / (FP + TN)
        if n_neg > 0:
            fpr = ((y_true == 0) & (y_pred == 1)).sum() / n_neg
        else:
            fpr = float("nan")

        results.append({
            "group": group_name,
            "tpr": tpr,
            "fpr": fpr,
            "n_samples": len(group_df),
            "n_positive": int(n_pos),
            "n_negative": int(n_neg),
        })

    result_df = pd.DataFrame(results)

    # Compute disparity
    valid_tpr = result_df["tpr"].dropna()
    valid_fpr = result_df["fpr"].dropna()
    if len(valid_tpr) > 1:
        result_df.attrs["tpr_disparity"] = valid_tpr.max() - valid_tpr.min()
    if len(valid_fpr) > 1:
        result_df.attrs["fpr_disparity"] = valid_fpr.max() - valid_fpr.min()

    return result_df


def full_fairness_report(
    df: pd.DataFrame,
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, pd.DataFrame]:
    """
    Run equalized odds analysis for sex, age group, and skin tone.

    Returns dict mapping group_name -> fairness DataFrame.
    """
    preds = (probs >= threshold).astype(int)
    report = {}

    # --- By Sex ---
    if "sex" in df.columns:
        report["sex"] = compute_equalized_odds(df, labels, preds, "sex")

    # --- By Age Group ---
    if "age_approx" in df.columns:
        df_with_age = df.copy()
        df_with_age["age_group"] = bin_age_groups(df_with_age["age_approx"])
        report["age_group"] = compute_equalized_odds(df_with_age, labels, preds, "age_group")

    # --- By Skin Tone ---
    if "skin_tone_index" in df.columns:
        df_with_tone = df.copy()
        df_with_tone["skin_tone_group"] = bin_skin_tones(df_with_tone["skin_tone_index"])
        report["skin_tone"] = compute_equalized_odds(df_with_tone, labels, preds, "skin_tone_group")

    return report
