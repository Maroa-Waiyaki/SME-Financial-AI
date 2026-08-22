"""Quick sanity checks for the generated synthetic data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/synthetic")


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv")


def main() -> None:
    transactions = load("transactions")
    businesses = load("businesses")

    print("Businesses:", len(businesses))
    print("Transactions:", len(transactions))
    print("Transaction types:", transactions["transaction_type"].value_counts().to_dict())
    print("Businesses with transactions:", transactions["business_id"].nunique())

    # Verify running-balance continuity per business (transaction_id order reflects generation order)
    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])
    transactions = transactions.sort_values(["business_id", "transaction_id"])

    inflow = {"RECEIPT", "DEPOSIT"}
    outflow = {"PAYMENT", "WITHDRAWAL", "FEE", "LOAN_REPAYMENT", "REFUND", "TRANSFER"}
    bad = 0

    for _, group in transactions.groupby("business_id"):
        group = group.sort_values("transaction_id").reset_index(drop=True)
        for i in range(1, len(group)):
            prev = group.iloc[i - 1]
            row = group.iloc[i]
            typ = row["transaction_type"]
            if typ in inflow:
                expected = round(prev["balance_after"] + row["amount"], 2)
            elif typ in outflow:
                expected = round(prev["balance_after"] - row["amount"], 2)
            else:
                expected = round(prev["balance_after"] - row["amount"], 2)

            if abs(row["balance_before"] - prev["balance_after"]) > 0.01 or abs(row["balance_after"] - expected) > 0.01:
                bad += 1
                break

    print("Balance-inconsistent business groups:", bad)
    assert bad == 0, "Transaction running balances are inconsistent"

    # Referential integrity
    sales = load("sales")
    customers = load("customers")
    assert set(sales["business_id"]).issubset(set(businesses["business_id"]))
    assert set(customers["business_id"]).issubset(set(businesses["business_id"]))
    assert set(transactions["business_id"]).issubset(set(businesses["business_id"]))
    print("Referential integrity: OK")


if __name__ == "__main__":
    main()
