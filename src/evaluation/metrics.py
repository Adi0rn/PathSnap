"""Evaluation metrics module.

Computes candidate classification metrics, observation-level Top-1 accuracy,
log loss, confusion matrices, and route continuity.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix
)


def evaluate_candidate_classification(y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Compute standard binary classification metrics on candidate rows."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    loss = None
    if y_prob is not None:
        y_prob_clipped = np.clip(y_prob, 1e-15, 1.0 - 1e-15)
        loss = float(log_loss(y_true, y_prob_clipped))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "macro_f1": macro_f1,
        "log_loss": loss,
        "confusion_matrix": cm
    }


def compute_top1_accuracy(prediction_df: pd.DataFrame) -> float:
    """Compute observation-level Top-1 accuracy.

    For each GPS observation, checks if the highest-scoring candidate is the true arc (label == 1).

    Parameters
    ----------
    prediction_df : pd.DataFrame
        DataFrame containing 'trip_id', 'gps_index', 'top1_predicted', 'label'.

    Returns
    -------
    float: Top-1 Accuracy in range [0.0, 1.0]
    """
    if "label" not in prediction_df.columns or "top1_predicted" not in prediction_df.columns:
        raise ValueError("prediction_df must contain 'label' and 'top1_predicted' columns.")

    # Select the top1 candidate for each observation
    top1_rows = prediction_df[prediction_df["top1_predicted"] == 1]
    if top1_rows.empty:
        return 0.0

    hits = (top1_rows["label"] == 1).sum()
    total_observations = len(top1_rows)
    return float(hits / total_observations)


def compute_route_continuity(prediction_df: pd.DataFrame,
                             arcs_dict: Optional[Dict[str, pd.DataFrame]] = None) -> float:
    """Estimate sequential route continuity of Top-1 predictions along GPS tracks.

    Measures fraction of consecutive steps where the predicted arc is either unchanged
    or connects directly in the road network graph.
    """
    if "top1_predicted" not in prediction_df.columns:
        return 0.0

    top1_df = prediction_df[prediction_df["top1_predicted"] == 1].sort_values(["trip_id", "gps_index"])
    if len(top1_df) < 2:
        return 1.0

    continuous_count = 0
    total_transitions = 0

    for tid, group in top1_df.groupby("trip_id"):
        pred_arcs = group["candidate_arc_id"].values
        n = len(pred_arcs)
        if n < 2:
            continue

        for i in range(1, n):
            total_transitions += 1
            # Step continuity: same arc or transition
            if pred_arcs[i] == pred_arcs[i - 1]:
                continuous_count += 1
            elif arcs_dict and tid in arcs_dict:
                arcs_df = arcs_dict[tid]
                u_curr = arcs_df.iloc[pred_arcs[i - 1]]["v"]
                v_next = arcs_df.iloc[pred_arcs[i]]["u"]
                if u_curr == v_next:
                    continuous_count += 1
            else:
                # Approximate continuity: consecutive arc ids within distance
                if abs(pred_arcs[i] - pred_arcs[i - 1]) <= 2:
                    continuous_count += 1

    return float(continuous_count / max(total_transitions, 1))


def evaluate_model(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   y_prob: Optional[np.ndarray] = None,
                   prediction_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Consolidated evaluation function matching the integration contract."""
    metrics = evaluate_candidate_classification(y_true, y_pred, y_prob)
    if prediction_df is not None:
        metrics["top1_accuracy"] = compute_top1_accuracy(prediction_df)
        metrics["route_continuity"] = compute_route_continuity(prediction_df)
    return metrics
