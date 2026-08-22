from __future__ import annotations

import logging
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _start_of_day(value: str | date | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    return datetime.combine(datetime.strptime(value, "%Y-%m-%d").date(), time.min)


def _end_of_day(value: str | date | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.max)
    return datetime.combine(datetime.strptime(value, "%Y-%m-%d").date(), time.max)


def _compute_amount_zscore(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    mean = df["amount"].mean()
    std = df["amount"].std()
    if std == 0 or pd.isna(std):
        df["amount_zscore"] = 0.0
    else:
        df["amount_zscore"] = (df["amount"] - mean) / std
    return df


def _detect_amount_anomalies(df: pd.DataFrame, threshold: float) -> list[dict[str, Any]]:
    flagged = df[df["amount_zscore"].abs() >= threshold].copy()
    results: list[dict[str, Any]] = []
    for _, row in flagged.iterrows():
        severity = "high" if row["amount_zscore"] >= 4 else "medium" if row["amount_zscore"] >= 3 else "low"
        results.append(
            {
                "transaction_id": row["transaction_id"],
                "reason": (
                    f"Amount is {row['amount_zscore']:.2f} standard deviations "
                    f"above the business's normal transaction amount"
                ),
                "severity": severity,
                "amount": float(row["amount"]),
                "zscore": float(row["amount_zscore"]),
            }
        )
    return results


def _detect_timing_anomalies(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    df = df.copy()
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    business_hours = df[(df["hour"] >= 7) & (df["hour"] <= 19)]
    off_hours = df[(df["hour"] < 6) | (df["hour"] > 22)]
    results: list[dict[str, Any]] = []
    for _, row in off_hours.iterrows():
        results.append(
            {
                "transaction_id": row["transaction_id"],
                "reason": (
                    f"Transaction occurred at {row['hour']:02d}:00, "
                    "outside this business's normal transaction hours (07:00-19:00)"
                ),
                "severity": "medium",
                "amount": float(row["amount"]),
                "zscore": float(row.get("amount_zscore", 0.0)),
            }
        )
    return results


def _detect_frequency_anomalies(df: pd.DataFrame, threshold: float) -> list[dict[str, Any]]:
    if df.empty:
        return []
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    daily_counts = df.groupby("date").size()
    if daily_counts.empty:
        return []
    mean = daily_counts.mean()
    std = daily_counts.std()
    if std == 0 or pd.isna(std):
        return []
    flagged_days = daily_counts[(daily_counts - mean).abs() / std >= threshold].index
    results: list[dict[str, Any]] = []
    for day in flagged_days:
        day_df = df[df["date"] == day]
        for _, row in day_df.head(3).iterrows():
            z = (daily_counts.loc[day] - mean) / std
            results.append(
                {
                    "transaction_id": row["transaction_id"],
                    "reason": (
                        f"Unusually high transaction frequency on {day}: "
                        f"{daily_counts.loc[day]} transactions vs average {mean:.1f} "
                        f"({z:.2f} standard deviations)"
                    ),
                    "severity": "medium" if z < 5 else "high",
                    "amount": float(row["amount"]),
                    "zscore": float(row.get("amount_zscore", 0.0)),
                }
            )
    return results


def detect_anomalies_df(
    df: pd.DataFrame,
    business_id: str,
    z_threshold: float = 3.0,
) -> list[dict[str, Any]]:
    df = df.copy()
    df = _compute_amount_zscore(df)
    results: list[dict[str, Any]] = []
    results.extend(_detect_amount_anomalies(df, z_threshold))
    results.extend(_detect_timing_anomalies(df))
    results.extend(_detect_frequency_anomalies(df, z_threshold))
    # Deduplicate by transaction_id, keeping highest severity first
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    seen: dict[str, dict[str, Any]] = {}
    for r in results:
        if r["transaction_id"] not in seen or severity_rank.get(
            r["severity"], 99
        ) < severity_rank.get(seen[r["transaction_id"]]["severity"], 99):
            seen[r["transaction_id"]] = r
    return list(seen.values())


def detect_anomalies(
    session: Any,
    business_id: str,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    z_threshold: float = 3.0,
) -> list[dict[str, Any]]:
    from sqlalchemy import and_, select

    from src.database.models import Transaction

    stmt = select(
        Transaction.transaction_id,
        Transaction.timestamp,
        Transaction.amount,
        Transaction.transaction_type,
    ).where(
        and_(
            Transaction.business_id == business_id,
            Transaction.timestamp >= _start_of_day(start_date),
            Transaction.timestamp <= _end_of_day(end_date),
        )
    )
    rows = session.execute(stmt).all()
    df = pd.DataFrame(rows, columns=["transaction_id", "timestamp", "amount", "transaction_type"])
    return detect_anomalies_df(df, business_id, z_threshold)
