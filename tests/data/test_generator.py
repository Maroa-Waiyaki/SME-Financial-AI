from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.generator import SMEDataGenerator


def test_generator_creates_all_tables() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = SMEDataGenerator(n_businesses=5, n_months=2, seed=1, output_dir=tmpdir)
        data = gen.generate()

        required = ["businesses", "customers", "sales", "expenses", "invoices", "loans", "transactions"]
        for table in required:
            assert table in data
            assert not data[table].empty


def test_business_customer_relationship() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = SMEDataGenerator(n_businesses=5, n_months=2, seed=2, output_dir=tmpdir)
        data = gen.generate()
        customer_biz = set(data["customers"]["business_id"])
        business_biz = set(data["businesses"]["business_id"])
        assert customer_biz.issubset(business_biz)


def test_sales_and_transactions_are_linked_by_business() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = SMEDataGenerator(n_businesses=5, n_months=2, seed=3, output_dir=tmpdir)
        data = gen.generate()
        assert set(data["sales"]["business_id"]).issubset(set(data["businesses"]["business_id"]))
        assert set(data["transactions"]["business_id"]).issubset(set(data["businesses"]["business_id"]))


def test_transaction_balance_consistency() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = SMEDataGenerator(n_businesses=5, n_months=2, seed=4, output_dir=tmpdir)
        data = gen.generate()
        transactions = data["transactions"].copy()
        transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])
        transactions = transactions.sort_values(["business_id", "timestamp"])

        inflow = {"RECEIPT", "DEPOSIT"}
        outflow = {"PAYMENT", "WITHDRAWAL", "FEE", "LOAN_REPAYMENT", "REFUND", "TRANSFER"}

        for business_id, group in transactions.groupby("business_id"):
            group = group.sort_values("timestamp").reset_index(drop=True)
            for i, row in group.iterrows():
                if i == 0:
                    continue
                prev = group.iloc[i - 1]
                if row["transaction_type"] in inflow:
                    expected = round(prev["balance_after"] + row["amount"], 2)
                elif row["transaction_type"] in outflow:
                    expected = round(prev["balance_after"] - row["amount"], 2)
                else:
                    expected = round(prev["balance_after"] - row["amount"], 2)
                assert abs(row["balance_before"] - prev["balance_after"]) < 0.01
                assert abs(row["balance_after"] - expected) < 0.01
