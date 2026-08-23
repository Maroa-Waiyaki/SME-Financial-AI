"""Tests for the credit-risk scorecard.

The scorecard is intentionally dependency-free: it only needs numpy and pandas.
These tests can therefore run in the lean runtime image.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.scorecard import (
    CreditScorecard,
    average_precision,
    classification_metrics,
    risk_level_from_score,
    roc_auc,
)


def test_risk_level_from_score():
    assert risk_level_from_score(12) == "low"
    assert risk_level_from_score(55) == "medium"
    assert risk_level_from_score(95) == "high"


def test_credit_scorecard_fits_and_predicts():
    n = 400
    np.random.seed(0)
    X = np.random.randn(n, 5)
    y = (X[:, 0] + 0.5 * X[:, 1] - X[:, 2] > 0).astype(int)

    model = CreditScorecard(feature_names=["f0", "f1", "f2", "f3", "f4"])
    model.fit(X, y, epochs=1000, l2=0.01)

    acc = (model.predict(X) == y).mean()
    assert acc > 0.7
    proba = model.predict_proba(X[0])
    assert 0.0 <= proba[0] <= 1.0


def test_reason_codes_are_exact_and_signed():
    model = CreditScorecard(feature_names=["a", "b", "c"])
    model.mean = np.array([0.0, 0.0, 0.0])
    model.scale = np.array([1.0, 1.0, 1.0])
    model.coef = np.array([2.0, -1.0, 0.0])
    model.intercept = 0.0

    features = {"a": 1.0, "b": 1.0, "c": 1.0}
    codes = model.reason_codes(features, top_n=2)
    assert len(codes) == 2
    assert codes[0]["feature"] == "a"
    assert codes[0]["direction"] == "increases_risk"
    assert codes[1]["feature"] == "b"
    assert codes[1]["direction"] == "decreases_risk"


def test_assessment_contains_all_fields():
    model = CreditScorecard(feature_names=["a", "b"])
    model.mean = np.array([0.0, 0.0])
    model.scale = np.array([1.0, 1.0])
    model.coef = np.array([1.0, -0.5])
    model.intercept = 0.0

    features = {"a": 1.0, "b": 0.0}
    out = model.assess(features, top_n=1)
    assert "risk_score" in out
    assert "probability_of_default" in out
    assert "risk_level" in out
    assert "model_version" in out
    assert "reason_codes" in out
    assert len(out["reason_codes"]) == 1


def test_classification_metrics_basic():
    y_true = np.array([0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.7])
    metrics = classification_metrics(y_true, y_prob)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_roc_auc_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    assert roc_auc(y_true, y_prob) == 1.0


def test_pr_auc_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    assert average_precision(y_true, y_prob) == 1.0


def test_serialisation_roundtrip(tmp_path):
    model = CreditScorecard(feature_names=["x", "y"])
    model.mean = np.array([1.0, 2.0])
    model.scale = np.array([3.0, 4.0])
    model.coef = np.array([0.5, -0.5])
    model.intercept = 0.1

    path = tmp_path / "scorecard.json"
    model.save(path)
    loaded = CreditScorecard.load(path)
    assert loaded.feature_names == model.feature_names
    np.testing.assert_allclose(loaded.coef, model.coef)
    np.testing.assert_allclose(loaded.intercept, model.intercept)
