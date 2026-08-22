from __future__ import annotations

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
import shap
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config.settings import get_settings
from src.features.credit_features import build_features, load_data


MODEL_FEATURES = [
    "business_age_years",
    "number_of_employees",
    "monthly_revenue_estimate",
    "revenue_mean",
    "revenue_std",
    "revenue_min",
    "revenue_max",
    "revenue_trend",
    "expense_mean",
    "expense_std",
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
    "total_loan_amount",
    "defaulted",
    "customer_count",
    "anomaly_count",
    "expense_ratio",
    "loan_to_revenue",
    "repayment_burden",
    "revenue_volatility",
]


def _prepare_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    dfs = load_data(data_dir)
    features = build_features(dfs)
    feature_cols = [c for c in MODEL_FEATURES if c in features.columns]
    X = features[feature_cols]
    y = features["target"]
    return X, y, feature_cols


def _evaluate(y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series) -> dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, y_prob) if len(y_true.unique()) > 1 else 0.0
    pr, re, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(re, pr) if len(pr) > 1 else 0.0
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist(),
    }


def train_credit_risk_model(
    data_dir: str | Path,
    output_dir: str | Path = "models",
    run_id: str | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    X, y, feature_cols = _prepare_data(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "use_label_encoder": False,
    }

    with mlflow.start_run(run_id=run_id):
        mlflow.log_params(params)
        mlflow.log_param("features", feature_cols)
        mlflow.log_param("n_samples", len(X))

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_prob = pd.Series(model.predict_proba(X_test)[:, 1], index=y_test.index)
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = _evaluate(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)

        # Cross-validation for stability
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores: list[float] = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            m = xgb.XGBClassifier(**params)
            m.fit(X_tr, y_tr)
            p = m.predict_proba(X_val)[:, 1]
            cv_scores.append(roc_auc_score(y_val, p))
        mlflow.log_metric("cv_roc_auc_mean", float(sum(cv_scores) / len(cv_scores)))

        # SHAP global feature importance
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, show=False)
        mlflow.log_artifact("shap_summary.png")

        # Save model and artefacts
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(model, Path(output_dir) / "credit_risk_model.pkl")
        with open(Path(output_dir) / "credit_risk_features.json", "w") as f:
            json.dump(feature_cols, f)
        mlflow.log_artifact(Path(output_dir) / "credit_risk_features.json")

        return {"model": model, "metrics": metrics, "features": feature_cols}


def load_credit_risk_model(model_path: str | Path = "models/credit_risk_model.pkl"):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Credit-risk model not found: {model_path}")
    return joblib.load(model_path)
