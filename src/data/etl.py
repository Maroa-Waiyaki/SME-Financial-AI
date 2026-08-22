from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_FILES = {
    "businesses": [
        "business_id",
        "business_name",
        "sector",
        "county",
        "business_age_years",
        "number_of_employees",
        "monthly_revenue_estimate",
        "business_size",
        "registration_status",
        "created_at",
    ],
    "customers": [
        "customer_id",
        "business_id",
        "customer_name",
        "customer_type",
        "location",
        "customer_since",
        "credit_limit",
        "payment_terms",
    ],
    "sales": [
        "sale_id",
        "business_id",
        "customer_id",
        "date",
        "product_category",
        "quantity",
        "unit_price",
        "total_amount",
        "payment_method",
    ],
    "expenses": [
        "expense_id",
        "business_id",
        "date",
        "category",
        "supplier",
        "amount",
        "payment_method",
        "description",
    ],
    "invoices": [
        "invoice_id",
        "business_id",
        "customer_id",
        "invoice_date",
        "due_date",
        "amount",
        "amount_paid",
        "outstanding_amount",
        "status",
    ],
    "loans": [
        "loan_id",
        "business_id",
        "loan_amount",
        "interest_rate",
        "term_months",
        "start_date",
        "outstanding_balance",
        "monthly_repayment",
        "status",
    ],
    "transactions": [
        "transaction_id",
        "business_id",
        "timestamp",
        "transaction_type",
        "amount",
        "balance_before",
        "balance_after",
        "category",
        "channel",
        "counterparty",
        "reference",
        "location",
        "status",
    ],
}


class DataValidationError(Exception):
    """Raised when synthetic data fails validation."""


def load_csvs(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    result: dict[str, pd.DataFrame] = {}
    for name, columns in REQUIRED_FILES.items():
        path = data_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Expected file not found: {path}")
        df = pd.read_csv(path, dtype=str)
        result[name] = df
    return result


def validate(dfs: dict[str, pd.DataFrame]) -> list[str]:
    errors: list[str] = []
    for name, df in dfs.items():
        required = REQUIRED_FILES[name]
        missing = [c for c in required if c not in df.columns]
        if missing:
            errors.append(f"{name}: missing columns {missing}")
            continue

        if df.empty:
            errors.append(f"{name}: table is empty")
            continue

        # Duplicated primary keys
        pk_col = f"{name[:-1]}_id" if name != "businesses" else "business_id"
        if pk_col in df.columns and df[pk_col].duplicated().any():
            n = df[pk_col].duplicated().sum()
            errors.append(f"{name}: {n} duplicate {pk_col} values")

        # Business referential integrity for non-business tables
        if name != "businesses" and "business_id" in df.columns:
            orphan = set(df["business_id"].dropna()) - set(dfs["businesses"]["business_id"])
            if orphan:
                errors.append(f"{name}: {len(orphan)} orphan business_id values")

        # Customer referential integrity for sales/invoices
        if name in ("sales", "invoices") and "customer_id" in df.columns:
            valid_customers = set(dfs["customers"]["customer_id"].dropna())
            present = set(df["customer_id"].dropna())
            orphan = present - valid_customers
            if orphan:
                errors.append(f"{name}: {len(orphan)} orphan customer_id values")

    return errors


def _coerce_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date


def _coerce_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _coerce_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def clean(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    cleaned: dict[str, pd.DataFrame] = {}

    businesses = dfs["businesses"].copy()
    businesses["business_age_years"] = _coerce_int(businesses["business_age_years"])
    businesses["number_of_employees"] = _coerce_int(businesses["number_of_employees"])
    businesses["monthly_revenue_estimate"] = _coerce_numeric(businesses["monthly_revenue_estimate"])
    businesses["created_at"] = _coerce_date(businesses["created_at"])
    cleaned["businesses"] = businesses

    customers = dfs["customers"].copy()
    customers["customer_since"] = _coerce_date(customers["customer_since"])
    customers["credit_limit"] = _coerce_numeric(customers["credit_limit"])
    cleaned["customers"] = customers

    sales = dfs["sales"].copy()
    sales["date"] = _coerce_date(sales["date"])
    sales["quantity"] = _coerce_int(sales["quantity"])
    sales["unit_price"] = _coerce_numeric(sales["unit_price"])
    sales["total_amount"] = _coerce_numeric(sales["total_amount"])
    sales["customer_id"] = sales["customer_id"].where(sales["customer_id"].notna(), None)
    cleaned["sales"] = sales

    expenses = dfs["expenses"].copy()
    expenses["date"] = _coerce_date(expenses["date"])
    expenses["amount"] = _coerce_numeric(expenses["amount"])
    cleaned["expenses"] = expenses

    invoices = dfs["invoices"].copy()
    invoices["invoice_date"] = _coerce_date(invoices["invoice_date"])
    invoices["due_date"] = _coerce_date(invoices["due_date"])
    invoices["amount"] = _coerce_numeric(invoices["amount"])
    invoices["amount_paid"] = _coerce_numeric(invoices["amount_paid"])
    invoices["outstanding_amount"] = _coerce_numeric(invoices["outstanding_amount"])
    cleaned["invoices"] = invoices

    loans = dfs["loans"].copy()
    loans["start_date"] = _coerce_date(loans["start_date"])
    loans["loan_amount"] = _coerce_numeric(loans["loan_amount"])
    loans["interest_rate"] = _coerce_numeric(loans["interest_rate"])
    loans["term_months"] = _coerce_int(loans["term_months"])
    loans["outstanding_balance"] = _coerce_numeric(loans["outstanding_balance"])
    loans["monthly_repayment"] = _coerce_numeric(loans["monthly_repayment"])
    cleaned["loans"] = loans

    transactions = dfs["transactions"].copy()
    transactions["timestamp"] = _coerce_datetime(transactions["timestamp"])
    for col in ("amount", "balance_before", "balance_after"):
        transactions[col] = _coerce_numeric(transactions[col])
    transactions["location"] = transactions["location"].where(transactions["location"].notna(), None)
    cleaned["transactions"] = transactions

    return cleaned


def transform(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    # Placeholder for feature-engineering transformations.
    return dfs


def load_to_postgres(dfs: dict[str, pd.DataFrame], engine: Any) -> None:
    from src.database.engine import init_db

    init_db(engine)
    load_order = ["businesses", "customers", "sales", "expenses", "invoices", "loans", "transactions"]
    for name in load_order:
        df = dfs[name]
        df.to_sql(name, con=engine, if_exists="append", index=False, method="multi", chunksize=1000)
        logger.info(f"Loaded {len(df)} rows into {name}")


def run_etl(data_dir: str | Path, engine: Any | None = None, dry_run: bool = False) -> dict[str, pd.DataFrame]:
    logger.info(f"Starting ETL from {data_dir}")
    dfs = load_csvs(data_dir)
    errors = validate(dfs)
    if errors:
        raise DataValidationError("Validation failed:\n" + "\n".join(errors))

    dfs = clean(dfs)
    dfs = transform(dfs)
    logger.info("Validation and cleaning successful")

    if dry_run:
        logger.info("Dry run: no database writes")
        return dfs

    if engine is None:
        from src.database.engine import get_engine
        engine = get_engine()

    load_to_postgres(dfs, engine)
    logger.info("ETL complete")
    return dfs
