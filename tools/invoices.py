from __future__ import annotations

from sqlalchemy.orm import Session

from src.database.models import Invoice


def get_last_invoices(db: Session, business_id: str, n: int = 5) -> list[dict]:
    invoices = (
        db.query(Invoice)
        .filter(Invoice.business_id == business_id)
        .order_by(Invoice.invoice_date.desc())
        .limit(n)
        .all()
    )
    return [
        {
            "invoice_id": i.invoice_id,
            "invoice_date": i.invoice_date.isoformat(),
            "due_date": i.due_date.isoformat(),
            "amount": float(i.amount),
            "amount_paid": float(i.amount_paid),
            "outstanding_amount": float(i.outstanding_amount),
            "status": i.status,
            "customer_id": i.customer_id,
        }
        for i in invoices
    ]


def get_outstanding_invoices(db: Session, business_id: str) -> list[dict]:
    invoices = (
        db.query(Invoice)
        .filter(Invoice.business_id == business_id, Invoice.outstanding_amount > 0)
        .order_by(Invoice.due_date.asc())
        .all()
    )
    return [
        {
            "invoice_id": i.invoice_id,
            "invoice_date": i.invoice_date.isoformat(),
            "due_date": i.due_date.isoformat(),
            "amount": float(i.amount),
            "amount_paid": float(i.amount_paid),
            "outstanding_amount": float(i.outstanding_amount),
            "status": i.status,
            "customer_id": i.customer_id,
        }
        for i in invoices
    ]
