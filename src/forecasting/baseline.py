from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_monthly(data_dir: str | Path, business_id: str) -> pd.DataFrame:
    data_dir = Path(data_dir)
    sales = pd.read_csv(data_dir / "sales.csv")
    sales = sales[sales["business_id"] == business_id].copy()
    sales["date"] = pd.to_datetime(sales["date"])
    sales["amount"] = pd.to_numeric(sales["total_amount"], errors="coerce")
    sales["month"] = sales["date"].dt.to_period("M")
    sales_monthly = sales.groupby("month")["amount"].sum().reset_index()
    sales_monthly["month"] = sales_monthly["month"].dt.to_timestamp()

    expenses = pd.read_csv(data_dir / "expenses.csv")
    expenses = expenses[expenses["business_id"] == business_id].copy()
    expenses["date"] = pd.to_datetime(expenses["date"])
    expenses["amount"] = pd.to_numeric(expenses["amount"], errors="coerce")
    expenses["month"] = expenses["date"].dt.to_period("M")
    exp_monthly = expenses.groupby("month")["amount"].sum().reset_index()
    exp_monthly["month"] = exp_monthly["month"].dt.to_timestamp()

    monthly = sales_monthly.merge(exp_monthly, on="month", how="outer", suffixes=("_revenue", "_expenses")).fillna(0)
    monthly["profit"] = monthly["amount_revenue"] - monthly["amount_expenses"]
    monthly = monthly.sort_values("month")
    return monthly


def _seasonal_factor(monthly: pd.DataFrame, horizon_days: int, forecast_start: pd.Timestamp) -> float:
    if len(monthly) < 3:
        return 1.0
    monthly["month_of_year"] = monthly["month"].dt.month
    by_month = monthly.groupby("month_of_year")["amount_revenue"].mean()
    overall = monthly["amount_revenue"].mean()
    if overall == 0:
        return 1.0

    # Average the factors for the months covered by the horizon
    factors = []
    for day_offset in range(0, horizon_days, 30):
        month = (forecast_start + pd.Timedelta(days=day_offset)).month
        factors.append(by_month.get(month, overall) / overall)
    return float(np.mean(factors)) if factors else 1.0


def _forecast_series(monthly: pd.DataFrame, horizon_days: int, series: str, forecast_start: pd.Timestamp) -> float:
    if monthly.empty or monthly[series].isna().all():
        return 0.0
    daily_mean = monthly[series].mean() / 30.0
    if daily_mean == 0:
        return 0.0
    factor = _seasonal_factor(monthly, horizon_days, forecast_start)
    forecast = daily_mean * horizon_days * factor
    return float(forecast)


def forecast_business(
    data_dir: str | Path,
    business_id: str,
    horizons: list[int] = (30, 60, 90),
) -> dict[str, Any]:
    monthly = _load_monthly(data_dir, business_id)
    forecast_start = pd.Timestamp(datetime.utcnow().date() + timedelta(days=1))

    results: dict[str, Any] = {}
    for h in horizons:
        revenue = _forecast_series(monthly, h, "amount_revenue", forecast_start)
        expenses = _forecast_series(monthly, h, "amount_expenses", forecast_start)
        cash_flow = revenue - expenses

        # Simple confidence bands: ±20% of the forecast
        results[f"{h}d"] = {
            "horizon_days": h,
            "forecast_start": forecast_start.strftime("%Y-%m-%d"),
            "forecast_end": (forecast_start + pd.Timedelta(days=h)).strftime("%Y-%m-%d"),
            "revenue": round(revenue, 2),
            "expenses": round(expenses, 2),
            "cash_flow": round(cash_flow, 2),
            "revenue_lower": round(revenue * 0.8, 2),
            "revenue_upper": round(revenue * 1.2, 2),
            "assumption": (
                "Baseline forecast uses the business's average daily revenue and expense "
                "with a seasonal month-of-year adjustment. It does not include one-off events."
            ),
        }

    return {
        "business_id": business_id,
        "horizons": results,
    }
