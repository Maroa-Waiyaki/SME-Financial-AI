from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config.settings import get_settings
from src.forecasting.baseline import forecast_business


def forecast_revenue(
    business_id: str,
    data_dir: str | None = None,
    horizons: list[int] = (30, 60, 90),
) -> dict[str, Any]:
    if data_dir is None:
        data_dir = get_settings().data_dir
    return forecast_business(data_dir, business_id, horizons)


def forecast_cashflow(
    business_id: str,
    data_dir: str | None = None,
    horizons: list[int] = (30, 60, 90),
) -> dict[str, Any]:
    result = forecast_revenue(business_id, data_dir, horizons)
    for h_key, values in result["horizons"].items():
        values["revenue"] = values["revenue"]
        values["expenses"] = values["expenses"]
        values["cash_flow"] = values["cash_flow"]
    return result
