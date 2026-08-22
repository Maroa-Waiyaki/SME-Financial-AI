from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_data(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    return {name: pd.read_csv(data_dir / f"{name}.csv") for name in [
        "businesses", "customers", "sales", "expenses", "invoices", "loans", "transactions",
    ]}


def _coerce_dates(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    dfs = {k: v.copy() for k, v in dfs.items()}
    dfs["sales"]["date"] = pd.to_datetime(dfs["sales"]["date"])
    dfs["expenses"]["date"] = pd.to_datetime(dfs["expenses"]["date"])
    dfs["invoices"]["invoice_date"] = pd.to_datetime(dfs["invoices"]["invoice_date"])
    dfs["invoices"]["due_date"] = pd.to_datetime(dfs["invoices"]["due_date"])
    dfs["loans"]["start_date"] = pd.to_datetime(dfs["loans"]["start_date"])
    dfs["transactions"]["timestamp"] = pd.to_datetime(dfs["transactions"]["timestamp"])
    return dfs


def build_features(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    dfs = _coerce_dates(dfs)

    businesses = dfs["businesses"].copy()
    businesses["monthly_revenue_estimate"] = pd.to_numeric(businesses["monthly_revenue_estimate"], errors="coerce")
    businesses["business_age_years"] = pd.to_numeric(businesses["business_age_years"], errors="coerce").astype("Int64")
    businesses["number_of_employees"] = pd.to_numeric(businesses["number_of_employees"], errors="coerce").astype("Int64")

    # Sales / revenue
    sales = dfs["sales"].copy()
    sales["total_amount"] = pd.to_numeric(sales["total_amount"], errors="coerce")
    sales["month"] = sales["date"].dt.to_period("M")
    sales_monthly = sales.groupby(["business_id", "month"])["total_amount"].sum().reset_index(name="monthly_revenue")
    rev_stats = sales_monthly.groupby("business_id")["monthly_revenue"].agg(
        revenue_mean="mean",
        revenue_std="std",
        revenue_min="min",
        revenue_max="max",
    ).reset_index()

    # Revenue trend: slope over months using simple linear regression on mean monthly revenue
    def _trend(x: pd.Series) -> float:
        if len(x) < 2:
            return 0.0
        x_arr = np.arange(len(x))
        y_arr = x.to_numpy()
        return np.polyfit(x_arr, y_arr, 1)[0]

    rev_trend = sales_monthly.groupby("business_id")["monthly_revenue"].apply(_trend).reset_index(name="revenue_trend")

    # Expenses
    expenses = dfs["expenses"].copy()
    expenses["amount"] = pd.to_numeric(expenses["amount"], errors="coerce")
    expenses["month"] = expenses["date"].dt.to_period("M")
    exp_monthly = expenses.groupby(["business_id", "month"])["amount"].sum().reset_index(name="monthly_expenses")
    exp_stats = exp_monthly.groupby("business_id")["monthly_expenses"].agg(
        expense_mean="mean",
        expense_std="std",
    ).reset_index()

    # Profit margin per month
    rev_exp = sales_monthly.merge(exp_monthly, on=["business_id", "month"], how="outer").fillna(0)
    rev_exp["profit"] = rev_exp["monthly_revenue"] - rev_exp["monthly_expenses"]
    rev_exp["profit_margin"] = np.where(rev_exp["monthly_revenue"] > 0, rev_exp["profit"] / rev_exp["monthly_revenue"], 0)
    profit_stats = rev_exp.groupby("business_id").agg(
        profit_mean=("profit", "mean"),
        profit_margin_mean=("profit_margin", "mean"),
    ).reset_index()

    # Cash flow from transactions
    transactions = dfs["transactions"].copy()
    transactions["amount"] = pd.to_numeric(transactions["amount"], errors="coerce")
    transactions["balance_after"] = pd.to_numeric(transactions["balance_after"], errors="coerce")
    transactions["month"] = transactions["timestamp"].dt.to_period("M")

    inflow_types = {"RECEIPT", "DEPOSIT"}
    outflow_types = {"PAYMENT", "WITHDRAWAL", "FEE", "LOAN_REPAYMENT", "REFUND"}
    transactions["direction"] = transactions["transaction_type"].map(
        lambda t: "in" if t in inflow_types else ("out" if t in outflow_types else "other")
    )
    transactions["signed_amount"] = np.where(
        transactions["direction"] == "in",
        transactions["amount"],
        np.where(transactions["direction"] == "out", -transactions["amount"], 0.0),
    )
    cash_monthly = transactions.groupby(["business_id", "month"])["signed_amount"].sum().reset_index(name="net_cash_flow")
    cash_stats = cash_monthly.groupby("business_id")["net_cash_flow"].agg(
        cash_flow_mean="mean",
        cash_flow_min="min",
        negative_months=lambda x: (x < 0).sum(),
    ).reset_index()

    # Transaction type counts
    txn_counts = transactions.groupby("business_id")["transaction_id"].count().reset_index(name="transaction_count")
    withdrawal_count = transactions[transactions["transaction_type"] == "WITHDRAWAL"].groupby("business_id")["transaction_id"].count().reset_index(name="withdrawal_count")
    mpesa_fees = transactions[transactions["category"].str.contains("M-Pesa|M-PESA|MPESA", case=False, na=False)].groupby("business_id")["amount"].sum().reset_index(name="mpesa_fees")

    # Invoices
    invoices = dfs["invoices"].copy()
    invoices["amount"] = pd.to_numeric(invoices["amount"], errors="coerce")
    invoices["outstanding_amount"] = pd.to_numeric(invoices["outstanding_amount"], errors="coerce")
    invoices["amount_paid"] = pd.to_numeric(invoices["amount_paid"], errors="coerce")
    invoice_stats = invoices.groupby("business_id").agg(
        total_invoices=("invoice_id", "count"),
        outstanding_invoices=("outstanding_amount", "sum"),
        overdue_invoices=("status", lambda x: (x == "overdue").sum()),
        unpaid_invoices=("status", lambda x: (x == "unpaid").sum()),
    ).reset_index()
    invoice_stats["overdue_rate"] = invoice_stats["overdue_invoices"] / invoice_stats["total_invoices"].clip(lower=1)

    # Loans
    loans = dfs["loans"].copy()
    loans["loan_amount"] = pd.to_numeric(loans["loan_amount"], errors="coerce")
    loans["outstanding_balance"] = pd.to_numeric(loans["outstanding_balance"], errors="coerce")
    loans["monthly_repayment"] = pd.to_numeric(loans["monthly_repayment"], errors="coerce")
    loan_stats = loans.groupby("business_id").agg(
        loan_balance=("outstanding_balance", "sum"),
        monthly_repayment=("monthly_repayment", "sum"),
        total_loan_amount=("loan_amount", "sum"),
        defaulted=("status", lambda x: (x == "defaulted").any().astype(int)),
    ).reset_index()

    # Customers
    customer_count = dfs["customers"].groupby("business_id")["customer_id"].count().reset_index(name="customer_count")

    # Anomalies (simple count of extreme amount z-scores > 3)
    def _anomaly_count(group: pd.DataFrame) -> int:
        if len(group) < 5:
            return 0
        amounts = group["amount"].astype(float)
        mean = amounts.mean()
        std = max(amounts.std(), 1e-9)
        z = np.abs((amounts - mean) / std)
        return int((z > 3).sum())

    anomaly_stats = transactions.groupby("business_id").apply(_anomaly_count, include_groups=False).reset_index(name="anomaly_count")

    # Merge all
    features = businesses[["business_id", "business_age_years", "number_of_employees", "monthly_revenue_estimate", "profile"]].copy()
    for df in [rev_stats, rev_trend, exp_stats, profit_stats, cash_stats, txn_counts, withdrawal_count, mpesa_fees, invoice_stats, loan_stats, customer_count, anomaly_stats]:
        features = features.merge(df, on="business_id", how="left")

    # Fill numeric NAs
    numeric_cols = features.select_dtypes(include=[np.number]).columns.difference(["business_id"])
    features[numeric_cols] = features[numeric_cols].fillna(0)

    # Derived ratios
    features["expense_ratio"] = np.where(
        features["revenue_mean"] > 0,
        features["expense_mean"] / features["revenue_mean"],
        0,
    )
    features["loan_to_revenue"] = np.where(
        features["revenue_mean"] > 0,
        features["loan_balance"] / features["revenue_mean"],
        0,
    )
    features["repayment_burden"] = np.where(
        features["revenue_mean"] > 0,
        features["monthly_repayment"] / features["revenue_mean"],
        0,
    )
    features["revenue_volatility"] = np.where(
        features["revenue_mean"] > 0,
        features["revenue_std"] / features["revenue_mean"],
        0,
    )

    # Target: high-risk if struggling/anomalous or any defaulted loan
    high_risk_profiles = {"struggling", "anomalous"}
    profile_flag = features["profile"].isin(high_risk_profiles).astype(int)
    defaulted = features["defaulted"].fillna(0).astype(int)
    features["target"] = (profile_flag | defaulted).astype(int)

    features = features.drop(columns=["profile"], errors="ignore")
    return features


def get_business_features(dfs: dict[str, pd.DataFrame], business_id: str) -> dict[str, Any]:
    df = build_features(dfs)
    row = df[df["business_id"] == business_id]
    if row.empty:
        raise ValueError(f"Business {business_id} not found")
    return row.iloc[0].to_dict()
