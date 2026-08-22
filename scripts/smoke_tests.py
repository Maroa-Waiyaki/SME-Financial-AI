from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

# Make local packages importable without installation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.financial import calculations as calc
from src.features.credit_features import build_features, load_data
from src.forecasting.baseline import forecast_business
from tools.anomaly import detect_anomalies_df
from tools.rag import chunk_documents, load_documents


def main() -> None:
    # Financial calculations
    rev = calc.revenue([Decimal("1000.00"), Decimal("2500.50")])
    assert rev == Decimal("3500.50"), rev
    assert calc.profit(Decimal("10000.00"), Decimal("6500.00")) == Decimal("3500.00")
    assert calc.profit_margin(Decimal("10000.00"), Decimal("6500.00")) == Decimal("35.00")
    inflow = calc.cash_inflow([Decimal("1000"), Decimal("500")], [Decimal("100")])
    outflow = calc.cash_outflow(
        [Decimal("300")], [Decimal("100")], [Decimal("20")], [Decimal("0")], [Decimal("10")]
    )
    assert inflow == Decimal("1600.00"), inflow
    assert outflow == Decimal("430.00"), outflow
    assert calc.net_cash_flow(inflow, outflow) == Decimal("1170.00")
    print("calculations ok")

    # Credit features
    dfs = load_data("data/synthetic")
    df = build_features(dfs)
    assert len(df) == 100
    assert "target" in df.columns
    print("credit features ok:", df.shape)

    # Forecasting
    f = forecast_business("data/synthetic", "B000001")
    assert "30d" in f["horizons"]
    print("forecast ok")

    # Anomaly detection
    tx = pd.read_csv("data/synthetic/transactions.csv")
    biz = tx[tx["business_id"] == "B000001"]
    anomalies = detect_anomalies_df(biz, "B000001")
    print("anomaly ok:", len(anomalies))

    # RAG document loading
    docs = load_documents("docs")
    chunks = chunk_documents(docs)
    print("rag docs ok:", len(docs), "chunks:", len(chunks))

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
