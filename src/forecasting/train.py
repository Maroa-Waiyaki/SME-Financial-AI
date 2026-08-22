from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from src.config.settings import get_settings
from src.features.credit_features import load_data


def _build_lag_features(df: pd.DataFrame, lags: list[int] = (30, 60, 90)) -> pd.DataFrame:
    for lag in lags:
        df[f"lag_{lag}"] = df["amount_revenue"].shift(lag)
    df["rolling_mean_90"] = df["amount_revenue"].rolling(window=90, min_periods=1).mean()
    df["rolling_std_90"] = df["amount_revenue"].rolling(window=90, min_periods=1).std().fillna(0)
    df["month"] = df["month"].dt.month
    df["year"] = df["month"].dt.year
    return df.dropna()


def train_revenue_forecaster(
    data_dir: str | Path,
    output_dir: str | Path = "models",
    business_id: str | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    dfs = load_data(data_dir)
    sales = dfs["sales"].copy()
    sales["date"] = pd.to_datetime(sales["date"])
    sales["month"] = sales["date"].dt.to_period("M").dt.to_timestamp()
    sales["amount_revenue"] = pd.to_numeric(sales["total_amount"], errors="coerce")

    if business_id:
        sales = sales[sales["business_id"] == business_id]

    monthly = sales.groupby(["business_id", "month"])["amount_revenue"].sum().reset_index()
    monthly = monthly.sort_values(["business_id", "month"])

    all_models = {}
    for bid, group in monthly.groupby("business_id"):
        if len(group) < 6:
            continue
        group = group.set_index("month").resample("MS").asfreq().fillna(0).reset_index()
        group = _build_lag_features(group)
        X = group.drop(columns=["amount_revenue", "business_id"])
        y = group["amount_revenue"]

        tscv = TimeSeriesSplit(n_splits=3)
        cv_mae = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            cv_mae.append(mean_absolute_error(y_test, preds))

        model.fit(X, y)
        all_models[bid] = model

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(all_models, Path(output_dir) / "forecast_revenue_models.pkl")
    with open(Path(output_dir) / "forecast_features.json", "w") as f:
        json.dump(list(X.columns), f)

    mlflow.log_param("business_count", len(all_models))
    mlflow.log_metric("mean_cv_mae", float(np.mean(cv_mae)))

    return {
        "models": all_models,
        "mean_cv_mae": float(np.mean(cv_mae)),
    }
