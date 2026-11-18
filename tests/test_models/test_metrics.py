"""Unit tests for evaluation metrics module."""

import pytest
import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    evaluate_candidate_classification,
    compute_top1_accuracy,
    compute_route_continuity,
    evaluate_model
)


def test_evaluate_candidate_classification():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.1, 0.8])

    res = evaluate_candidate_classification(y_true, y_pred, y_prob)

    assert 0.0 <= res["accuracy"] <= 1.0
    assert 0.0 <= res["macro_f1"] <= 1.0
    assert res["log_loss"] is not None
    assert len(res["confusion_matrix"]) == 2


def test_evaluate_model_consolidated():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])

    df = pd.DataFrame({
        "trip_id": ["T1", "T1", "T1", "T1"],
        "gps_index": [0, 0, 1, 1],
        "candidate_arc_id": [10, 20, 20, 30],
        "top1_predicted": [0, 1, 1, 0],
        "label": [0, 1, 1, 0]
    })

    res = evaluate_model(y_true, y_pred, y_prob, prediction_df=df)

    assert "accuracy" in res
    assert "top1_accuracy" in res
    assert res["top1_accuracy"] == 1.0
    assert "route_continuity" in res
