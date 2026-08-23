"""Credit-risk scorecard: regularised logistic regression implemented in NumPy.

Why logistic regression rather than a gradient-boosted ensemble?

1. **Deployability.** The default runtime image only ships numpy/pandas. The
   gradient-boosted path (``src/ml/credit_risk.py``) needs xgboost + shap, which
   are optional extras. This module has zero extra dependencies, so credit
   scoring works in every environment out of the box.
2. **Explainability.** Credit decisions need defensible, per-applicant reason
   codes. A linear model in log-odds space gives an *exact* additive attribution
   (``coefficient x standardised value``) rather than an approximation. This is
   how production credit scorecards are usually built and is far easier to
   defend to a credit committee or regulator.

The model is trained with full-batch gradient descent on the log-loss with L2
regularisation, and supports class weighting because the distress class is the
minority.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_NAME = "sme-credit-scorecard"
MODEL_VERSION = "1.0.0"

# Observable predictors. `defaulted` is retained as a legitimate feature
# (historic repayment behaviour) because it is no longer part of the target.
# `business_id`, `profile` and `target` are never features.
MODEL_FEATURES: list[str] = [
    "business_age_years",
    "number_of_employees",
    "monthly_revenue_estimate",
    "revenue_mean",
    "revenue_std",
    "revenue_trend",
    "expense_mean",
    "profit_mean",
    "profit_margin_mean",
    "cash_flow_mean",
    "cash_flow_min",
    "negative_months",
    "transaction_count",
    "withdrawal_count",
    "mpesa_fees",
    "total_invoices",
    "outstanding_invoices",
    "overdue_invoices",
    "unpaid_invoices",
    "overdue_rate",
    "loan_balance",
    "monthly_repayment",
    "defaulted",
    "customer_count",
    "anomaly_count",
    "expense_ratio",
    "loan_to_revenue",
    "repayment_burden",
    "revenue_volatility",
]

# Human-readable labels used when rendering reason codes.
FEATURE_LABELS: dict[str, str] = {
    "business_age_years": "Business age",
    "number_of_employees": "Number of employees",
    "monthly_revenue_estimate": "Estimated monthly revenue",
    "revenue_mean": "Average monthly revenue",
    "revenue_std": "Revenue variability",
    "revenue_trend": "Revenue trend",
    "expense_mean": "Average monthly expenses",
    "profit_mean": "Average monthly profit",
    "profit_margin_mean": "Average profit margin",
    "cash_flow_mean": "Average monthly cash flow",
    "cash_flow_min": "Worst monthly cash flow",
    "negative_months": "Months with negative cash flow",
    "transaction_count": "Transaction count",
    "withdrawal_count": "Cash withdrawal count",
    "mpesa_fees": "M-Pesa fees paid",
    "total_invoices": "Total invoices issued",
    "outstanding_invoices": "Outstanding receivables",
    "overdue_invoices": "Overdue invoices",
    "unpaid_invoices": "Unpaid invoices",
    "overdue_rate": "Share of invoices overdue",
    "loan_balance": "Outstanding loan balance",
    "monthly_repayment": "Monthly loan repayment",
    "defaulted": "Previous loan default",
    "customer_count": "Number of customers",
    "anomaly_count": "Unusual transactions detected",
    "expense_ratio": "Expense-to-revenue ratio",
    "loan_to_revenue": "Loan-to-revenue ratio",
    "repayment_burden": "Repayment burden on revenue",
    "revenue_volatility": "Revenue volatility",
}

DEFAULT_MODEL_PATH = Path("models") / "credit_scorecard.json"


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable logistic function.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def risk_level_from_score(score: float) -> str:
    """Map a 0-100 risk score onto the platform's three risk bands."""
    if score < 40:
        return "low"
    if score < 70:
        return "medium"
    return "high"


class CreditScorecard:
    """L2-regularised logistic regression with standardisation and reason codes."""

    def __init__(
        self,
        feature_names: list[str],
        mean: np.ndarray | None = None,
        scale: np.ndarray | None = None,
        coef: np.ndarray | None = None,
        intercept: float = 0.0,
        version: str = MODEL_VERSION,
        metrics: dict[str, Any] | None = None,
        trained_at: str | None = None,
    ) -> None:
        self.feature_names = list(feature_names)
        n = len(self.feature_names)
        self.mean = np.zeros(n) if mean is None else np.asarray(mean, dtype=float)
        self.scale = np.ones(n) if scale is None else np.asarray(scale, dtype=float)
        self.coef = np.zeros(n) if coef is None else np.asarray(coef, dtype=float)
        self.intercept = float(intercept)
        self.version = version
        self.metrics = metrics or {}
        self.trained_at = trained_at

    # ------------------------------------------------------------------ fitting

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.scale

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        learning_rate: float = 0.1,
        epochs: int = 4000,
        l2: float = 1e-3,
        class_weight: bool = True,
    ) -> CreditScorecard:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self.mean = X.mean(axis=0)
        std = X.std(axis=0)
        # Guard against zero-variance columns.
        self.scale = np.where(std < 1e-12, 1.0, std)
        Xs = self._standardise(X)

        n_samples, n_features = Xs.shape
        self.coef = np.zeros(n_features)
        self.intercept = 0.0

        if class_weight:
            n_pos = max(float(y.sum()), 1.0)
            n_neg = max(float(len(y) - y.sum()), 1.0)
            w_pos = n_samples / (2.0 * n_pos)
            w_neg = n_samples / (2.0 * n_neg)
        else:
            w_pos = w_neg = 1.0
        weights = np.where(y == 1, w_pos, w_neg)
        weight_sum = weights.sum()

        for _ in range(epochs):
            p = _sigmoid(Xs @ self.coef + self.intercept)
            residual = (p - y) * weights
            grad_coef = (Xs.T @ residual) / weight_sum + l2 * self.coef
            grad_intercept = residual.sum() / weight_sum
            self.coef -= learning_rate * grad_coef
            self.intercept -= learning_rate * grad_intercept

        return self

    # --------------------------------------------------------------- prediction

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return _sigmoid(self._standardise(X) @ self.coef + self.intercept)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def _vector_from_mapping(self, features: dict[str, Any]) -> np.ndarray:
        return np.array(
            [float(features.get(name, 0.0) or 0.0) for name in self.feature_names],
            dtype=float,
        )

    def reason_codes(self, features: dict[str, Any], top_n: int = 5) -> list[dict[str, Any]]:
        """Exact additive log-odds attribution for a single business.

        For a linear model the contribution of feature *i* to the log-odds is
        ``coef_i * standardised_value_i``. Positive contributions push the
        business towards higher risk, negative ones towards lower risk. These
        are exact, not approximations, so they can never contradict the model.
        """
        x = self._vector_from_mapping(features)
        xs = (x - self.mean) / self.scale
        contributions = self.coef * xs

        order = np.argsort(-np.abs(contributions))
        codes: list[dict[str, Any]] = []
        for idx in order[:top_n]:
            name = self.feature_names[idx]
            contribution = float(contributions[idx])
            if abs(contribution) < 1e-9:
                continue
            codes.append(
                {
                    "feature": name,
                    "label": FEATURE_LABELS.get(name, name),
                    "value": float(x[idx]),
                    "contribution": round(contribution, 4),
                    "direction": "increases_risk" if contribution > 0 else "decreases_risk",
                }
            )
        return codes

    def assess(self, features: dict[str, Any], top_n: int = 5) -> dict[str, Any]:
        """Score one business and return the full explainable assessment."""
        x = self._vector_from_mapping(features)
        probability = float(self.predict_proba(x)[0])
        score = round(probability * 100, 2)
        return {
            "probability_of_default": round(probability, 5),
            "risk_score": score,
            "risk_level": risk_level_from_score(score),
            "model_name": MODEL_NAME,
            "model_version": self.version,
            "reason_codes": self.reason_codes(features, top_n=top_n),
        }

    # ------------------------------------------------------------ serialisation

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": MODEL_NAME,
            "model_version": self.version,
            "trained_at": self.trained_at,
            "feature_names": self.feature_names,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
            "metrics": self.metrics,
        }

    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CreditScorecard:
        return cls(
            feature_names=payload["feature_names"],
            mean=np.asarray(payload["mean"], dtype=float),
            scale=np.asarray(payload["scale"], dtype=float),
            coef=np.asarray(payload["coef"], dtype=float),
            intercept=float(payload["intercept"]),
            version=payload.get("model_version", MODEL_VERSION),
            metrics=payload.get("metrics", {}),
            trained_at=payload.get("trained_at"),
        )

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> CreditScorecard:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Credit scorecard not found at {path}")
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------- metrics


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """ROC-AUC via the rank (Mann-Whitney U) formulation, ties handled."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(y_prob)
    sorted_prob = y_prob[order]
    ranks = np.empty(len(y_prob), dtype=float)
    i = 0
    while i < len(sorted_prob):
        j = i
        while j + 1 < len(sorted_prob) and sorted_prob[j + 1] == sorted_prob[i]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = average_rank
        i = j + 1

    sum_pos_ranks = ranks[y_true == 1].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def precision_recall_curve(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    total_pos = max(int(y_true.sum()), 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / total_pos
    return precision, recall


def average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """PR-AUC computed as average precision (no interpolation bias)."""
    precision, recall = precision_recall_curve(y_true, y_prob)
    if len(recall) == 0:
        return 0.0
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    """Full credit-risk metric set, including the false-negative count."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "threshold": threshold,
        "accuracy": round((tp + tn) / max(len(y_true), 1), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc(y_true, y_prob), 4),
        "pr_auc": round(average_precision(y_true, y_prob), 4),
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
        # Surfaced explicitly: a false negative is a risky borrower we approved.
        "false_negatives": fn,
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }


# -------------------------------------------------------------------- training


def _stratified_split(
    y: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_size)))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def prepare_dataset(features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Select the model matrix from an engineered feature frame."""
    feature_cols = [c for c in MODEL_FEATURES if c in features.columns]
    leaked = {"target", "profile", "business_id"} & set(feature_cols)
    if leaked:
        raise ValueError(f"Leaked columns present in feature matrix: {sorted(leaked)}")
    X = features[feature_cols].astype(float).to_numpy()
    y = features["target"].astype(int).to_numpy()
    return X, y, feature_cols


def train_scorecard(
    features: pd.DataFrame,
    *,
    test_size: float = 0.25,
    seed: int = 42,
    epochs: int = 4000,
    learning_rate: float = 0.1,
    l2: float = 1e-3,
) -> tuple[CreditScorecard, dict[str, Any]]:
    """Train the scorecard and return it alongside held-out + CV metrics."""
    X, y, feature_cols = prepare_dataset(features)
    train_idx, test_idx = _stratified_split(y, test_size, seed)

    model = CreditScorecard(feature_cols)
    model.fit(
        X[train_idx],
        y[train_idx],
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )

    test_prob = model.predict_proba(X[test_idx])
    train_prob = model.predict_proba(X[train_idx])
    test_metrics = classification_metrics(y[test_idx], test_prob)
    train_metrics = classification_metrics(y[train_idx], train_prob)

    # 5-fold stratified CV for stability, refitting from scratch each fold.
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(5)]
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        for position, sample in enumerate(idx):
            folds[position % 5].append(int(sample))

    cv_auc: list[float] = []
    for k in range(5):
        val = np.array(sorted(folds[k]))
        tr = np.array(sorted(i for j in range(5) if j != k for i in folds[j]))
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[val])) < 2:
            continue
        fold_model = CreditScorecard(feature_cols).fit(
            X[tr], y[tr], epochs=epochs, learning_rate=learning_rate, l2=l2
        )
        cv_auc.append(roc_auc(y[val], fold_model.predict_proba(X[val])))

    report = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "n_features": len(feature_cols),
        "features": feature_cols,
        "train": train_metrics,
        "test": test_metrics,
        "cv_roc_auc_mean": round(float(np.mean(cv_auc)), 4) if cv_auc else None,
        "cv_roc_auc_std": round(float(np.std(cv_auc)), 4) if cv_auc else None,
        # Train-vs-test AUC gap is the overfitting check.
        "overfit_gap_roc_auc": round(
            train_metrics["roc_auc"] - test_metrics["roc_auc"], 4
        ),
        "coefficients": {
            name: round(float(c), 4) for name, c in zip(feature_cols, model.coef)
        },
    }

    model.metrics = report
    model.trained_at = datetime.now(timezone.utc).isoformat()
    return model, report


_CACHED_MODEL: CreditScorecard | None = None


def get_scorecard(path: str | Path = DEFAULT_MODEL_PATH) -> CreditScorecard:
    """Load the trained scorecard once and reuse it across calls."""
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        _CACHED_MODEL = CreditScorecard.load(path)
    return _CACHED_MODEL


def reset_scorecard_cache() -> None:
    global _CACHED_MODEL
    _CACHED_MODEL = None
