from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field


class BusinessOut(BaseModel):
    business_id: str
    business_name: str
    sector: str
    county: str
    business_age_years: int
    number_of_employees: int
    monthly_revenue_estimate: Decimal
    business_size: str
    registration_status: str
    created_at: date

    model_config = {"from_attributes": True}


class PeriodParams(BaseModel):
    start_date: date
    end_date: date


class FinancialSummaryOut(BaseModel):
    business_id: str
    period_start: date
    period_end: date
    revenue: Decimal
    expenses: Decimal
    profit: Decimal
    profit_margin_percent: Decimal
    cash_inflows: Decimal
    cash_outflows: Decimal
    net_cash_flow: Decimal
    outstanding_receivables: Decimal
    outstanding_loans: Decimal
    expense_ratio_percent: Decimal
    top_expense_categories: list[dict[str, Any]]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    business_id: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    response: str
    intent: str
    result: dict


class HealthOut(BaseModel):
    status: str
    version: str = "0.1.0"
