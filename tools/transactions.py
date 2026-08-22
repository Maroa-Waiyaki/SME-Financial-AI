from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

from sqlalchemy import and_, desc, func, select
from sqlalchemy.orm import Session

from src.database.models import Transaction


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


def get_transactions(
    session: Session,
    business_id: str,
    start_date: str | date,
    end_date: str | date,
    limit: int = 100,
    transaction_type: str | None = None,
) -> list[Transaction]:
    conditions = [
        Transaction.business_id == business_id,
        Transaction.timestamp >= _start_of_day(start_date),
        Transaction.timestamp <= _end_of_day(end_date),
    ]
    if transaction_type:
        conditions.append(Transaction.transaction_type == transaction_type)
    stmt = select(Transaction).where(and_(*conditions)).order_by(desc(Transaction.timestamp)).limit(limit)
    return list(session.execute(stmt).scalars().all())


def _sum_by_type(session: Session, business_id: str, start: datetime, end: datetime) -> dict[str, Decimal]:
    stmt = (
        select(Transaction.transaction_type, func.coalesce(func.sum(Transaction.amount), Decimal("0")))
        .where(
            and_(
                Transaction.business_id == business_id,
                Transaction.timestamp >= start,
                Transaction.timestamp <= end,
            )
        )
        .group_by(Transaction.transaction_type)
    )
    return {row[0]: Decimal(str(row[1])) for row in session.execute(stmt)}


def _count_by(session: Session, business_id: str, start: datetime, end: datetime, column: Any) -> dict[Any, int]:
    stmt = (
        select(column, func.count())
        .where(
            and_(
                Transaction.business_id == business_id,
                Transaction.timestamp >= start,
                Transaction.timestamp <= end,
            )
        )
        .group_by(column)
    )
    return {row[0]: row[1] for row in session.execute(stmt)}


def summarize_transactions(
    session: Session,
    business_id: str,
    start_date: str | date,
    end_date: str | date,
) -> dict[str, Any]:
    start = _start_of_day(start_date)
    end = _end_of_day(end_date)
    stmt_count = select(func.count()).where(
        and_(
            Transaction.business_id == business_id,
            Transaction.timestamp >= start,
            Transaction.timestamp <= end,
        )
    )
    count = session.execute(stmt_count).scalar_one() or 0
    by_type = _sum_by_type(session, business_id, start, end)
    by_channel = _count_by(session, business_id, start, end, Transaction.channel)
    by_category = _count_by(session, business_id, start, end, Transaction.category)
    by_type_count = _count_by(session, business_id, start, end, Transaction.transaction_type)
    return {
        "business_id": business_id,
        "period_start": start_date,
        "period_end": end_date,
        "transaction_count": count,
        "total_by_type": {k: float(v) for k, v in by_type.items()},
        "count_by_channel": by_channel,
        "count_by_category": by_category,
        "count_by_type": by_type_count,
    }


def get_transaction_volume(
    session: Session,
    business_id: str,
    start_date: str | date,
    end_date: str | date,
) -> int:
    start = _start_of_day(start_date)
    end = _end_of_day(end_date)
    stmt = select(func.count()).where(
        and_(
            Transaction.business_id == business_id,
            Transaction.timestamp >= start,
            Transaction.timestamp <= end,
        )
    )
    return session.execute(stmt).scalar_one() or 0


def get_top_transactions(
    session: Session,
    business_id: str,
    start_date: str | date,
    end_date: str | date,
    n: int = 5,
) -> list[dict[str, Any]]:
    start = _start_of_day(start_date)
    end = _end_of_day(end_date)
    stmt = (
        select(Transaction)
        .where(
            and_(
                Transaction.business_id == business_id,
                Transaction.timestamp >= start,
                Transaction.timestamp <= end,
            )
        )
        .order_by(desc(Transaction.amount))
        .limit(n)
    )
    rows = session.execute(stmt).scalars().all()
    return [
        {
            "transaction_id": r.transaction_id,
            "timestamp": r.timestamp,
            "transaction_type": r.transaction_type,
            "amount": r.amount,
            "category": r.category,
            "channel": r.channel,
            "counterparty": r.counterparty,
        }
        for r in rows
    ]


def compare_transaction_periods(
    session: Session,
    business_id: str,
    current_start: str | date,
    current_end: str | date,
    previous_start: str | date,
    previous_end: str | date,
) -> dict[str, Any]:
    current = summarize_transactions(session, business_id, current_start, current_end)
    previous = summarize_transactions(session, business_id, previous_start, previous_end)
    current_total = sum(current["total_by_type"].values())
    previous_total = sum(previous["total_by_type"].values())
    volume_change = current["transaction_count"] - previous["transaction_count"]
    return {
        "current": current,
        "previous": previous,
        "total_volume_change": volume_change,
        "total_value_change": round(current_total - previous_total, 2),
    }
