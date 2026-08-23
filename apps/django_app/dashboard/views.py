from __future__ import annotations

from decimal import Decimal
from typing import Any

from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView, LogoutView
from django.shortcuts import render
from sqlalchemy import and_, desc, func, select

from src.database.engine import get_db
from src.database.models import (
    Business,
    Expense,
    ModelPrediction,
    Sale,
    Transaction,
)
from tools.anomaly import detect_anomalies
from tools.document_parser import extract_text_from_file
from tools.financial import get_business_profile, get_financial_summary
from tools.financial_reports import (
    export_financial_excel,
    export_financial_pptx,
    generate_balance_sheet,
    generate_pnl_statement,
)
from tools.rag import get_index, reset_index
from tools.transactions import get_transaction_volume, summarize_transactions

DEFAULT_START = "2021-01-01"
DEFAULT_END = "2023-12-31"
DEFAULT_BUSINESS_ID = "B000001"


def _f(value: Any) -> float:
    """Coerce Decimal/None/str into a plain float."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _money(value: Any) -> str:
    """Format a number as a KES amount with thousands separators."""
    return f"{_f(value):,.2f}"


def _business_options(db) -> list[dict[str, str]]:
    rows = db.execute(
        select(Business.business_id, Business.business_name).order_by(Business.business_id)
    ).all()
    return [{"business_id": r[0], "business_name": r[1]} for r in rows]


def _summary_to_plain(summary: dict[str, Any]) -> dict[str, Any]:
    """Convert a get_financial_summary() result into template-safe primitives."""
    plain = {
        "business_id": summary.get("business_id"),
        "period_start": summary.get("period_start"),
        "period_end": summary.get("period_end"),
    }
    for key in (
        "revenue",
        "expenses",
        "profit",
        "profit_margin_percent",
        "cash_inflows",
        "cash_outflows",
        "net_cash_flow",
        "outstanding_receivables",
        "outstanding_loans",
        "expense_ratio_percent",
    ):
        plain[key] = _f(summary.get(key))
        plain[f"{key}_fmt"] = _money(summary.get(key))
    plain["top_expense_categories"] = [
        {"category": row.get("category"), "total": _f(row.get("total")), "total_fmt": _money(row.get("total"))}
        for row in (summary.get("top_expense_categories") or [])
    ]
    return plain


class DashboardLoginView(LoginView):
    template_name = "dashboard/login.html"


class DashboardLogoutView(LogoutView):
    next_page = "/login/"


def landing_page(request):
    """Public portfolio landing page highlighting samstatsai and the platform."""
    return render(request, "dashboard/landing.html")


@login_required
def dashboard_index(request):
    business_id = request.GET.get("business_id") or "B000001"
    context: dict = {"business_id": business_id}

    try:
        with get_db() as db:
            businesses = db.query(Business).order_by(Business.business_id).limit(100).all()
            context["businesses"] = [
                {
                    "business_id": b.business_id,
                    "business_name": b.business_name,
                    "sector": b.sector,
                    "county": b.county,
                    "business_size": b.business_size,
                    "monthly_revenue_estimate": b.monthly_revenue_estimate,
                }
                for b in businesses
            ]

            profile = get_business_profile(db, business_id)
            if profile:
                context["profile"] = profile
                context["summary"] = get_financial_summary(
                    db, business_id, "2021-01-01", "2023-12-31"
                )
    except Exception as exc:
        context["error"] = f"Could not load data: {exc}"

    return render(request, "dashboard/index.html", context)


def _monthly_breakdown(db, business_id: str, start: str, end: str) -> list[dict[str, Any]]:
    """Monthly revenue (sales), expenses and profit for the period."""
    rev_month = func.date_trunc("month", Sale.date).label("month")
    rev_stmt = (
        select(rev_month, func.coalesce(func.sum(Sale.total_amount), 0))
        .where(
            and_(
                Sale.business_id == business_id,
                Sale.date >= start,
                Sale.date <= end,
            )
        )
        .group_by(rev_month)
        .order_by(rev_month)
    )
    exp_month = func.date_trunc("month", Expense.date).label("month")
    exp_stmt = (
        select(exp_month, func.coalesce(func.sum(Expense.amount), 0))
        .where(
            and_(
                Expense.business_id == business_id,
                Expense.date >= start,
                Expense.date <= end,
            )
        )
        .group_by(exp_month)
        .order_by(exp_month)
    )

    revenue_by_month: dict[str, float] = {}
    for row in db.execute(rev_stmt):
        revenue_by_month[row[0].strftime("%Y-%m")] = _f(row[1])

    expenses_by_month: dict[str, float] = {}
    for row in db.execute(exp_stmt):
        expenses_by_month[row[0].strftime("%Y-%m")] = _f(row[1])

    months = sorted(set(revenue_by_month) | set(expenses_by_month))
    breakdown: list[dict[str, Any]] = []
    for month in months:
        revenue = round(revenue_by_month.get(month, 0.0), 2)
        expenses = round(expenses_by_month.get(month, 0.0), 2)
        breakdown.append(
            {
                "month": month,
                "revenue": revenue,
                "expenses": expenses,
                "profit": round(revenue - expenses, 2),
                "revenue_fmt": _money(revenue),
                "expenses_fmt": _money(expenses),
                "profit_fmt": _money(revenue - expenses),
            }
        )
    return breakdown


def _expense_categories(db, business_id: str, start: str, end: str) -> list[dict[str, Any]]:
    stmt = (
        select(Expense.category, func.coalesce(func.sum(Expense.amount), 0).label("total"))
        .where(
            and_(
                Expense.business_id == business_id,
                Expense.date >= start,
                Expense.date <= end,
            )
        )
        .group_by(Expense.category)
        .order_by(desc("total"))
    )
    rows = [{"category": r[0], "total": round(_f(r[1]), 2)} for r in db.execute(stmt)]
    grand_total = sum(r["total"] for r in rows) or 0.0
    for row in rows:
        row["total_fmt"] = _money(row["total"])
        row["share_percent"] = round((row["total"] / grand_total) * 100, 1) if grand_total else 0.0
    return rows


def _portfolio_distribution(db) -> dict[str, Any]:
    size_stmt = (
        select(Business.business_size, func.count())
        .group_by(Business.business_size)
        .order_by(func.count().desc())
    )
    by_size = [{"label": r[0] or "unknown", "count": int(r[1])} for r in db.execute(size_stmt)]

    by_risk: list[dict[str, Any]] = []
    try:
        risk_stmt = (
            select(ModelPrediction.risk_level, func.count())
            .group_by(ModelPrediction.risk_level)
            .order_by(func.count().desc())
        )
        by_risk = [{"label": r[0] or "unknown", "count": int(r[1])} for r in db.execute(risk_stmt)]
    except Exception:
        by_risk = []

    return {
        "by_size": by_size,
        "by_risk": by_risk,
        "has_risk_data": bool(by_risk),
        "total_businesses": sum(item["count"] for item in by_size),
    }


@login_required
def analytics_page(request):
    business_id = request.GET.get("business_id") or DEFAULT_BUSINESS_ID
    context: dict = {
        "business_id": business_id,
        "period_start": DEFAULT_START,
        "period_end": DEFAULT_END,
        "businesses": [],
        "monthly_data": [],
        "expense_categories": [],
        "portfolio": {"by_size": [], "by_risk": [], "has_risk_data": False, "total_businesses": 0},
    }

    try:
        with get_db() as db:
            context["businesses"] = _business_options(db)

            profile = get_business_profile(db, business_id)
            if profile is None:
                context["error"] = f"Business {business_id} was not found."
            else:
                context["profile"] = {
                    "business_id": profile["business_id"],
                    "business_name": profile["business_name"],
                    "sector": profile["sector"],
                    "county": profile["county"],
                    "business_size": profile["business_size"],
                    "registration_status": profile["registration_status"],
                    "business_age_years": profile["business_age_years"],
                    "number_of_employees": profile["number_of_employees"],
                    "monthly_revenue_estimate": _f(profile["monthly_revenue_estimate"]),
                    "monthly_revenue_estimate_fmt": _money(profile["monthly_revenue_estimate"]),
                }
                summary = get_financial_summary(db, business_id, DEFAULT_START, DEFAULT_END)
                context["summary"] = _summary_to_plain(summary)
                context["monthly_data"] = _monthly_breakdown(
                    db, business_id, DEFAULT_START, DEFAULT_END
                )
                context["expense_categories"] = _expense_categories(
                    db, business_id, DEFAULT_START, DEFAULT_END
                )

            context["portfolio"] = _portfolio_distribution(db)
    except Exception as exc:  # pragma: no cover - defensive, mirrors dashboard_index
        context["error"] = f"Could not load data: {exc}"

    return render(request, "dashboard/analytics.html", context)


def _recent_transactions(db, business_id: str, limit: int = 50) -> list[dict[str, Any]]:
    stmt = (
        select(Transaction)
        .where(Transaction.business_id == business_id)
        .order_by(desc(Transaction.timestamp))
        .limit(limit)
    )
    rows = db.execute(stmt).scalars().all()
    return [
        {
            "transaction_id": t.transaction_id,
            "timestamp": t.timestamp.strftime("%Y-%m-%d %H:%M") if t.timestamp else "",
            "transaction_type": t.transaction_type,
            "amount": _f(t.amount),
            "amount_fmt": _money(t.amount),
            "category": t.category,
            "channel": t.channel,
            "counterparty": t.counterparty,
            "status": t.status,
        }
        for t in rows
    ]


def _transaction_categories(db, business_id: str, start: str, end: str) -> list[dict[str, Any]]:
    stmt = (
        select(
            Transaction.category,
            func.count().label("count"),
            func.coalesce(func.sum(Transaction.amount), 0).label("total"),
        )
        .where(
            and_(
                Transaction.business_id == business_id,
                Transaction.timestamp >= f"{start} 00:00:00",
                Transaction.timestamp <= f"{end} 23:59:59",
            )
        )
        .group_by(Transaction.category)
        .order_by(desc("total"))
    )
    return [
        {
            "category": r[0],
            "count": int(r[1]),
            "total": round(_f(r[2]), 2),
            "total_fmt": _money(r[2]),
        }
        for r in db.execute(stmt)
    ]


@login_required
def transactions_page(request):
    business_id = request.GET.get("business_id") or DEFAULT_BUSINESS_ID
    context: dict = {
        "business_id": business_id,
        "period_start": DEFAULT_START,
        "period_end": DEFAULT_END,
        "businesses": [],
        "transactions": [],
        "anomalies": [],
        "anomaly_ids": [],
        "category_data": [],
    }

    try:
        with get_db() as db:
            context["businesses"] = _business_options(db)

            profile = get_business_profile(db, business_id)
            if profile is None:
                context["error"] = f"Business {business_id} was not found."
            else:
                context["profile"] = {
                    "business_id": profile["business_id"],
                    "business_name": profile["business_name"],
                    "sector": profile["sector"],
                    "county": profile["county"],
                    "business_size": profile["business_size"],
                }

                summary = summarize_transactions(db, business_id, DEFAULT_START, DEFAULT_END)
                total_by_type = {k: _f(v) for k, v in (summary.get("total_by_type") or {}).items()}
                inflow_types = {"RECEIPT", "DEPOSIT"}
                inflow = sum(v for k, v in total_by_type.items() if k in inflow_types)
                outflow = sum(v for k, v in total_by_type.items() if k not in inflow_types)

                context["summary"] = {
                    "transaction_count": int(summary.get("transaction_count") or 0),
                    "total_by_type": total_by_type,
                    "count_by_channel": dict(summary.get("count_by_channel") or {}),
                    "count_by_type": dict(summary.get("count_by_type") or {}),
                    "inflow": round(inflow, 2),
                    "inflow_fmt": _money(inflow),
                    "outflow": round(outflow, 2),
                    "outflow_fmt": _money(outflow),
                    "net_flow": round(inflow - outflow, 2),
                    "net_flow_fmt": _money(inflow - outflow),
                }
                context["volume"] = get_transaction_volume(
                    db, business_id, DEFAULT_START, DEFAULT_END
                )
                context["transactions"] = _recent_transactions(db, business_id, 50)
                context["category_data"] = _transaction_categories(
                    db, business_id, DEFAULT_START, DEFAULT_END
                )

                raw_anomalies = detect_anomalies(db, business_id, DEFAULT_START, DEFAULT_END)
                anomalies = [
                    {
                        "transaction_id": a.get("transaction_id"),
                        "reason": a.get("reason"),
                        "severity": (a.get("severity") or "low").lower(),
                        "amount": _f(a.get("amount")),
                        "amount_fmt": _money(a.get("amount")),
                        "zscore": round(_f(a.get("zscore")), 2),
                    }
                    for a in raw_anomalies
                ]
                severity_rank = {"high": 0, "medium": 1, "low": 2}
                anomalies.sort(
                    key=lambda a: (severity_rank.get(a["severity"], 99), -float(_f(a["amount"])))
                )
                context["anomalies"] = anomalies
                context["anomaly_ids"] = [a["transaction_id"] for a in anomalies]
                context["anomaly_count"] = len(anomalies)
                anomaly_lookup = {a["transaction_id"]: a for a in anomalies}
                for txn in context["transactions"]:
                    match = anomaly_lookup.get(txn["transaction_id"])
                    txn["is_anomaly"] = match is not None
                    txn["anomaly_severity"] = match["severity"] if match else ""
                    txn["anomaly_reason"] = match["reason"] if match else ""
    except Exception as exc:  # pragma: no cover - defensive, mirrors dashboard_index
        context["error"] = f"Could not load data: {exc}"

    return render(request, "dashboard/transactions.html", context)


@login_required
def chat_page(request):
    return render(request, "dashboard/chat.html")


@login_required
def reports_page(request):
    business_id = request.GET.get("business_id") or DEFAULT_BUSINESS_ID
    context = {"business_id": business_id, "businesses": [], "pnl": None, "balance_sheet": None, "error": None}

    try:
        with get_db() as db:
            context["businesses"] = _business_options(db)
            context["pnl"] = generate_pnl_statement(db, business_id, "2023-01-01", "2023-12-31")
            context["balance_sheet"] = generate_balance_sheet(db, business_id, "2023-12-31")
    except Exception as exc:
        context["error"] = f"Failed to generate reports: {exc}"

    return render(request, "dashboard/reports.html", context)


@login_required
def export_pptx_view(request):
    from django.http import HttpResponse
    business_id = request.GET.get("business_id") or DEFAULT_BUSINESS_ID
    with get_db() as db:
        pnl = generate_pnl_statement(db, business_id, "2023-01-01", "2023-12-31")
        bs = generate_balance_sheet(db, business_id, "2023-12-31")
        content = export_financial_pptx(pnl, bs)

    response = HttpResponse(content, content_type="application/vnd.openxmlformats-officedocument.presentationml.presentation")
    response["Content-Disposition"] = f'attachment; filename="{business_id}_Financial_Review.pptx"'
    return response


@login_required
def export_xlsx_view(request):
    from django.http import HttpResponse
    business_id = request.GET.get("business_id") or DEFAULT_BUSINESS_ID
    with get_db() as db:
        pnl = generate_pnl_statement(db, business_id, "2023-01-01", "2023-12-31")
        bs = generate_balance_sheet(db, business_id, "2023-12-31")
        content = export_financial_excel(pnl, bs)

    response = HttpResponse(content, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    response["Content-Disposition"] = f'attachment; filename="{business_id}_Financial_Model.xlsx"'
    return response


@login_required
def documents_page(request):
    from pathlib import Path
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir = docs_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    context = {"businesses": [], "documents": [], "message": None, "error": None}

    with get_db() as db:
        context["businesses"] = _business_options(db)

    if request.method == "POST":
        uploaded_file = request.FILES.get("file")
        business_id = request.POST.get("business_id", "").strip()

        if not uploaded_file:
            context["error"] = "Please choose a file to upload."
        else:
            try:
                content = uploaded_file.read()
                text = extract_text_from_file(content, uploaded_file.name)
                if not text.strip():
                    context["error"] = f"Could not extract readable text from {uploaded_file.name}."
                else:
                    prefix = f"{business_id}_" if business_id else ""
                    safe_name = f"{prefix}{Path(uploaded_file.name).stem}.md"
                    target = uploads_dir / safe_name

                    header = f"# Document: {uploaded_file.name}\n"
                    if business_id:
                        header += f"**Associated Business ID:** {business_id}\n\n"
                    target.write_text(header + text, encoding="utf-8")

                    reset_index()
                    context["message"] = f"Successfully uploaded and indexed '{uploaded_file.name}' ({len(text)} characters). It is now searchable in AI Chat."
            except Exception as exc:
                context["error"] = f"Upload failed: {exc}"

    # List all documents
    doc_list = []
    for f in sorted(docs_dir.glob("*.md")):
        doc_list.append({"name": f.name, "type": "System Policy", "size": f"{f.stat().st_size} bytes"})
    for f in sorted(uploads_dir.glob("*.md")):
        doc_list.append({"name": f.name, "type": "Business Upload", "size": f"{f.stat().st_size} bytes"})
    context["documents"] = doc_list

    return render(request, "dashboard/documents.html", context)
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir = docs_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    context = {"businesses": [], "documents": [], "message": None, "error": None}

    with get_db() as db:
        context["businesses"] = _business_options(db)

    if request.method == "POST":
        uploaded_file = request.FILES.get("file")
        business_id = request.POST.get("business_id", "").strip()

        if not uploaded_file:
            context["error"] = "Please choose a file to upload."
        else:
            try:
                content = uploaded_file.read()
                text = extract_text_from_file(content, uploaded_file.name)
                if not text.strip():
                    context["error"] = f"Could not extract readable text from {uploaded_file.name}."
                else:
                    prefix = f"{business_id}_" if business_id else ""
                    safe_name = f"{prefix}{Path(uploaded_file.name).stem}.md"
                    target = uploads_dir / safe_name

                    header = f"# Document: {uploaded_file.name}\n"
                    if business_id:
                        header += f"**Associated Business ID:** {business_id}\n\n"
                    target.write_text(header + text, encoding="utf-8")

                    reset_index()
                    context["message"] = f"Successfully uploaded and indexed '{uploaded_file.name}' ({len(text)} characters). It is now searchable in AI Chat."
            except Exception as exc:
                context["error"] = f"Upload failed: {exc}"

    # List all documents
    doc_list = []
    for f in sorted(docs_dir.glob("*.md")):
        doc_list.append({"name": f.name, "type": "System Policy", "size": f"{f.stat().st_size} bytes"})
    for f in sorted(uploads_dir.glob("*.md")):
        doc_list.append({"name": f.name, "type": "Business Upload", "size": f"{f.stat().st_size} bytes"})
    context["documents"] = doc_list

    return render(request, "dashboard/documents.html", context)
