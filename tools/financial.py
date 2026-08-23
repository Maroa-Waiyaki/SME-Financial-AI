from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Sequence

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from src.database.models import Business, Expense, Invoice, Loan, Sale, Transaction
from src.financial import calculations as calc


CASH_INFLOW_TYPES = {"RECEIPT", "DEPOSIT"}
CASH_OUTFLOW_TYPES = {"PAYMENT", "WITHDRAWAL", "FEE", "LOAN_REPAYMENT", "REFUND"}


def _coerce_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def _start_of_day(value: str | date | datetime) -> datetime:
    return datetime.combine(_coerce_date(value), time.min)


def _end_of_day(value: str | date | datetime) -> datetime:
    return datetime.combine(_coerce_date(value), time.max)


def get_business_profile(session: Session, business_id: str) -> dict[str, Any] | None:
    business = session.get(Business, business_id)
    if business is None:
        return None
    return {
        "business_id": business.business_id,
        "business_name": business.business_name,
        "sector": business.sector,
        "county": business.county,
        "business_age_years": business.business_age_years,
        "number_of_employees": business.number_of_employees,
        "monthly_revenue_estimate": business.monthly_revenue_estimate,
        "business_size": business.business_size,
        "registration_status": business.registration_status,
        "created_at": business.created_at,
    }


def get_transactions(
    session: Session,
    business_id: str,
    start_date: str | date,
    end_date: str | date,
    limit: int = 100,
) -> Sequence[Transaction]:
    stmt = (
        select(Transaction)
        .where(
            and_(
                Transaction.business_id == business_id,
                Transaction.timestamp >= _start_of_day(start_date),
                Transaction.timestamp <= _end_of_day(end_date),
            )
        )
        .order_by(Transaction.timestamp)
        .limit(limit)
    )
    return session.execute(stmt).scalars().all()


def _sum_by_type(
    session: Session,
    business_id: str,
    start_date: str | date,
    end_date: str | date,
    types: set[str],
) -> Decimal:
    stmt = (
        select(func.coalesce(func.sum(Transaction.amount), Decimal("0")))
        .where(
            and_(
                Transaction.business_id == business_id,
                Transaction.transaction_type.in_(types),
                Transaction.timestamp >= _start_of_day(start_date),
                Transaction.timestamp <= _end_of_day(end_date),
            )
        )
    )
    result = session.execute(stmt).scalar_one()
    return Decimal(str(result)) if result is not None else Decimal("0")


def get_revenue(session: Session, business_id: str, start_date: str | date, end_date: str | date) -> Decimal:
    """Total receipt/return cash inflows (RECEIPT + DEPOSIT) in the period."""
    return _sum_by_type(session, business_id, start_date, end_date, CASH_INFLOW_TYPES)


def get_expenses(session: Session, business_id: str, start_date: str | date, end_date: str | date) -> Decimal:
    """Total cash outflows (PAYMENT + WITHDRAWAL + FEE + LOAN_REPAYMENT + REFUND)."""
    return _sum_by_type(session, business_id, start_date, end_date, CASH_OUTFLOW_TYPES)


def calculate_profit(session: Session, business_id: str, start_date: str | date, end_date: str | date) -> Decimal:
    rev = get_revenue(session, business_id, start_date, end_date)
    exp = get_expenses(session, business_id, start_date, end_date)
    return calc.profit(rev, exp)


def calculate_profit_margin(session: Session, business_id: str, start_date: str | date, end_date: str | date) -> Decimal:
    rev = get_revenue(session, business_id, start_date, end_date)
    exp = get_expenses(session, business_id, start_date, end_date)
    return calc.profit_margin(rev, exp)


def get_cash_flow(session: Session, business_id: str, start_date: str | date, end_date: str | date) -> dict[str, Decimal]:
    inflow = _sum_by_type(session, business_id, start_date, end_date, CASH_INFLOW_TYPES)
    outflow = _sum_by_type(session, business_id, start_date, end_date, CASH_OUTFLOW_TYPES)
    return {
        "cash_inflows": inflow,
        "cash_outflows": outflow,
        "net_cash_flow": calc.net_cash_flow(inflow, outflow),
    }


def get_outstanding_receivables(session: Session, business_id: str) -> Decimal:
    stmt = select(func.coalesce(func.sum(Invoice.outstanding_amount), Decimal("0"))).where(
        Invoice.business_id == business_id
    )
    result = session.execute(stmt).scalar_one()
    return Decimal(str(result)) if result is not None else Decimal("0")


def get_loan_balance(session: Session, business_id: str) -> Decimal:
    stmt = select(func.coalesce(func.sum(Loan.outstanding_balance), Decimal("0"))).where(
        Loan.business_id == business_id
    )
    result = session.execute(stmt).scalar_one()
    return Decimal(str(result)) if result is not None else Decimal("0")


def get_top_expenses(session: Session, business_id: str, start_date: str | date, end_date: str | date, n: int = 5):
    stmt = (
        select(Expense.category, func.sum(Expense.amount).label("total"))
        .where(
            and_(
                Expense.business_id == business_id,
                Expense.date >= _coerce_date(start_date),
                Expense.date <= _coerce_date(end_date),
            )
        )
        .group_by(Expense.category)
        .order_by(func.sum(Expense.amount).desc())
        .limit(n)
    )
    return [dict(row._mapping) for row in session.execute(stmt)]


def get_annual_sales_summary(
    session: Session,
    business_id: str,
    start_date: str | date = "2021-01-01",
    end_date: str | date = "2023-12-31",
) -> list[dict[str, Any]]:
    """Return yearly aggregated revenue and expense metrics for multi-year comparisons."""
    year_col = func.date_trunc("year", Sale.date).label("year")
    stmt = (
        select(
            year_col,
            func.coalesce(func.sum(Sale.total_amount), Decimal("0")).label("revenue"),
            func.count(Sale.sale_id).label("sales_count"),
        )
        .where(
            and_(
                Sale.business_id == business_id,
                Sale.date >= _coerce_date(start_date),
                Sale.date <= _coerce_date(end_date),
            )
        )
        .group_by(year_col)
        .order_by(year_col)
    )
    rows = session.execute(stmt).all()
    results = []
    for r in rows:
        y_str = r[0].strftime("%Y") if isinstance(r[0], (date, datetime)) else str(r[0])
        results.append({
            "year": y_str,
            "revenue": Decimal(str(r[1])),
            "sales_count": int(r[2]),
        })
    return results


def get_monthly_sales_trend(
    session: Session,
    business_id: str,
    start_date: str | date = "2021-01-01",
    end_date: str | date = "2023-12-31",
) -> list[dict[str, Any]]:
    """Return monthly aggregated sales revenue and count, ordered chronologically."""
    month_col = func.date_trunc("month", Sale.date).label("month")
    stmt = (
        select(
            month_col,
            func.coalesce(func.sum(Sale.total_amount), Decimal("0")).label("revenue"),
            func.count(Sale.sale_id).label("sales_count"),
        )
        .where(
            and_(
                Sale.business_id == business_id,
                Sale.date >= _coerce_date(start_date),
                Sale.date <= _coerce_date(end_date),
            )
        )
        .group_by(month_col)
        .order_by(month_col)
    )
    rows = session.execute(stmt).all()
    results = []
    for r in rows:
        m_str = r[0].strftime("%Y-%m") if isinstance(r[0], (date, datetime)) else str(r[0])
        results.append({
            "month": m_str,
            "revenue": Decimal(str(r[1])),
            "sales_count": int(r[2]),
        })
    return results


def get_best_and_worst_months(
    session: Session,
    business_id: str,
    start_date: str | date = "2021-01-01",
    end_date: str | date = "2023-12-31",
) -> dict[str, Any]:
    """Find the highest and lowest revenue months for a business."""
    trend = get_monthly_sales_trend(session, business_id, start_date, end_date)
    if not trend:
        return {"best_month": None, "worst_month": None, "monthly_breakdown": []}

    sorted_by_rev = sorted(trend, key=lambda x: x["revenue"], reverse=True)
    return {
        "best_month": sorted_by_rev[0],
        "worst_month": sorted_by_rev[-1],
        "all_months": trend,
    }
    stmt = (
        select(Sale.customer_id, func.sum(Sale.total_amount).label("total"))
        .where(
            and_(
                Sale.business_id == business_id,
                Sale.date >= _coerce_date(start_date),
                Sale.date <= _coerce_date(end_date),
            )
        )
        .group_by(Sale.customer_id)
        .order_by(func.sum(Sale.total_amount).desc())
        .limit(n)
    )
    return [dict(row._mapping) for row in session.execute(stmt)]


def get_financial_summary(
    session: Session,
    business_id: str,
    start_date: str | date,
    end_date: str | date,
) -> dict[str, Any]:
    rev = get_revenue(session, business_id, start_date, end_date)
    exp = get_expenses(session, business_id, start_date, end_date)
    profit = calc.profit(rev, exp)
    margin = calc.profit_margin(rev, exp)
    cf = get_cash_flow(session, business_id, start_date, end_date)
    receivables = get_outstanding_receivables(session, business_id)
    loans = get_loan_balance(session, business_id)
    top_expenses = get_top_expenses(session, business_id, start_date, end_date)

    return {
        "business_id": business_id,
        "period_start": _coerce_date(start_date).isoformat(),
        "period_end": _coerce_date(end_date).isoformat(),
        "revenue": rev,
        "expenses": exp,
        "profit": profit,
        "profit_margin_percent": margin,
        "cash_inflows": cf["cash_inflows"],
        "cash_outflows": cf["cash_outflows"],
        "net_cash_flow": cf["net_cash_flow"],
        "outstanding_receivables": receivables,
        "outstanding_loans": loans,
        "expense_ratio_percent": calc.expense_ratio(exp, rev),
        "top_expense_categories": top_expenses,
    }


def compare_periods(
    session: Session,
    business_id: str,
    current_start: str | date,
    current_end: str | date,
    previous_start: str | date,
    previous_end: str | date,
) -> dict[str, Any]:
    cur_rev = get_revenue(session, business_id, current_start, current_end)
    cur_exp = get_expenses(session, business_id, current_start, current_end)
    cur_profit = calc.profit(cur_rev, cur_exp)
    cur_cf = get_cash_flow(session, business_id, current_start, current_end)

    prev_rev = get_revenue(session, business_id, previous_start, previous_end)
    prev_exp = get_expenses(session, business_id, previous_start, previous_end)
    prev_profit = calc.profit(prev_rev, prev_exp)
    prev_cf = get_cash_flow(session, business_id, previous_start, previous_end)

    return {
        "current": {
            "revenue": cur_rev,
            "expenses": cur_exp,
            "profit": cur_profit,
            "net_cash_flow": cur_cf["net_cash_flow"],
        },
        "previous": {
            "revenue": prev_rev,
            "expenses": prev_exp,
            "profit": prev_profit,
            "net_cash_flow": prev_cf["net_cash_flow"],
        },
        "revenue_growth_percent": calc.revenue_growth(cur_rev, prev_rev),
        "expense_change": calc.expense_change(cur_exp, prev_exp),
        "profit_change": calc.expense_change(cur_profit, prev_profit),
        "cash_flow_change": calc.expense_change(cur_cf["net_cash_flow"], prev_cf["net_cash_flow"]),
    }
