from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base


class Business(Base):
    __tablename__ = "businesses"

    business_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    business_name: Mapped[str] = mapped_column(String(255), nullable=False)
    sector: Mapped[str] = mapped_column(String(64), nullable=False)
    county: Mapped[str] = mapped_column(String(64), nullable=False)
    business_age_years: Mapped[int] = mapped_column(Integer, nullable=False)
    number_of_employees: Mapped[int] = mapped_column(Integer, nullable=False)
    monthly_revenue_estimate: Mapped[float] = mapped_column(Numeric(15, 2), nullable=False)
    business_size: Mapped[str] = mapped_column(String(16), nullable=False)
    registration_status: Mapped[str] = mapped_column(String(32), nullable=False)
    profile: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[date] = mapped_column(Date, nullable=False)

    customers: Mapped[list["Customer"]] = relationship("Customer", back_populates="business")
    transactions: Mapped[list["Transaction"]] = relationship("Transaction", back_populates="business")
    sales: Mapped[list["Sale"]] = relationship("Sale", back_populates="business")
    expenses: Mapped[list["Expense"]] = relationship("Expense", back_populates="business")
    invoices: Mapped[list["Invoice"]] = relationship("Invoice", back_populates="business")
    loans: Mapped[list["Loan"]] = relationship("Loan", back_populates="business")
    credit_assessments: Mapped[list["CreditAssessment"]] = relationship(
        "CreditAssessment", back_populates="business"
    )

    __table_args__ = (
        Index("ix_businesses_county", "county"),
        Index("ix_businesses_sector", "sector"),
        Index("ix_businesses_business_size", "business_size"),
    )


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    business_id: Mapped[str] = mapped_column(ForeignKey("businesses.business_id"), nullable=False)
    customer_name: Mapped[str] = mapped_column(String(255), nullable=False)
    customer_type: Mapped[str] = mapped_column(String(32), nullable=False)
    location: Mapped[str] = mapped_column(String(64), nullable=False)
    customer_since: Mapped[date] = mapped_column(Date, nullable=False)
    credit_limit: Mapped[float] = mapped_column(Numeric(15, 2), nullable=False)
    payment_terms: Mapped[str] = mapped_column(String(32), nullable=False)

    business: Mapped[Business] = relationship("Business", back_populates="customers")
    sales: Mapped[list["Sale"]] = relationship("Sale", back_populates="customer")
    invoices: Mapped[list["Invoice"]] = relationship("Invoice", back_populates="customer")

    __table_args__ = (Index("ix_customers_business_id", "business_id"),)


class Transaction(Base):
    __tablename__ = "transactions"

    transaction_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    business_id: Mapped[str] = mapped_column(ForeignKey("businesses.business_id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    transaction_type: Mapped[str] = mapped_column(String(32), nullable=False)
    amount: Mapped[float] = mapped_column(Numeric(15, 2), nullable=False)
    balance_before: Mapped[float] = mapped_column(Numeric(15, 2), nullable=False)
    balance_after: Mapped[float] = mapped_column(Numeric(15, 2), nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    channel: Mapped[str] = mapped_column(String(32), nullable=False)
    counterparty: Mapped[str] = mapped_column(String(255), nullable=False)
    reference: Mapped[str] = mapped_column(String(64), nullable=False)
    location: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)

    business: Mapped[Business] = relationship("Business", back_populates="transactions")

    __table_args__ = (
        Index("ix_transactions_business_id", "business_id"),
        Index("ix_transactions_timestamp", "timestamp"),
        Index("ix_transactions_transaction_type", "transaction_type"),
        Index("ix_transactions_category", "category"),
        Index("ix_transactions_channel", "channel"),
        Index("ix_transactions_status", "status"),
    )


class Sale(Base):
    __tablename__ = "sales"

    sale_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    business_id: Mapped[str] = mapped_column(ForeignKey("businesses.business_id"), nullable=False)
    customer_id: Mapped[str | None] = mapped_column(
        ForeignKey("customers.customer_id"), nullable=True
    )
    date: Mapped[date] = mapped_column(Date, nullable=False)
    product_category: Mapped[str] = mapped_column(String(64), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    unit_price: Mapped[float] = mapped_column(Numeric(15, 2), nullable=False)
    total_amount: Mapped[float] = mapped_column(Numeric(15, 2), nullable=False)
    payment_method: Mapped[str] = mapped_column(String(32), nullable=False)

    business: Mapped[Business] = relationship("Business", back_populates="sales")
    customer: Mapped[Customer | None] = relationship("Customer", back_populates="sales")

    __table_args__ = (
        Index("ix_sales_business_id", "business_id"),
        Index("ix_sales_customer_id", "customer_id"),
        Index("ix_sales_date", "date"),
    )


class Expense(Base):
    __tablename__ = "expenses"

    expense_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    business_id: Mapped[str] = mapped_column(ForeignKey("businesses.business_id"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    supplier: Mapped[str] = mapped_column(String(255), nullable=False)
    amount: Mapped[float] = mapped_column(Numeric(15, 2), nullable=False)
    payment_method: Mapped[str] = mapped_column(String(32), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    business: Mapped[Business] = relationship("Business", back_populates="expenses")

    __table_args__ = (
        Index("ix_expenses_business_id", "business_id"),
        Index("ix_expenses_date", "date"),
        Index("ix_expenses_category", "category"),
    )


class Invoice(Base):
    __tablename__ = "invoices"

    invoice_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    business_id: Mapped[str] = mapped_column(ForeignKey("businesses.business_id"), nullable=False)
    customer_id: Mapped[str | None] = mapped_column(
        ForeignKey("customers.customer_id"), nullable=True
    )
    invoice_date: Mapped[date] = mapped_column(Date, nullable=False)
    due_date: Mapped[date] = mapped_column(Date, nullable=False)
    amount: Mapped[float] = mapped_column(Numeric(15, 2), nullable=False)
    amount_paid: Mapped[float] = mapped_column(Numeric(15, 2), nullable=False)
    outstanding_amount: Mapped[float] = mapped_column(Numeric(15, 2), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)

    business: Mapped[Business] = relationship("Business", back_populates="invoices")
    customer: Mapped[Customer | None] = relationship("Customer", back_populates="invoices")

    __table_args__ = (
        Index("ix_invoices_business_id", "business_id"),
        Index("ix_invoices_customer_id", "customer_id"),
        Index("ix_invoices_status", "status"),
        Index("ix_invoices_due_date", "due_date"),
    )


class Loan(Base):
    __tablename__ = "loans"

    loan_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    business_id: Mapped[str] = mapped_column(ForeignKey("businesses.business_id"), nullable=False)
    loan_amount: Mapped[float] = mapped_column(Numeric(15, 2), nullable=False)
    interest_rate: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    term_months: Mapped[int] = mapped_column(Integer, nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    outstanding_balance: Mapped[float] = mapped_column(Numeric(15, 2), nullable=False)
    monthly_repayment: Mapped[float] = mapped_column(Numeric(15, 2), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)

    business: Mapped[Business] = relationship("Business", back_populates="loans")

    __table_args__ = (
        Index("ix_loans_business_id", "business_id"),
        Index("ix_loans_status", "status"),
    )


class CreditAssessment(Base):
    __tablename__ = "credit_assessments"

    assessment_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    business_id: Mapped[str] = mapped_column(ForeignKey("businesses.business_id"), nullable=False)
    assessment_date: Mapped[date] = mapped_column(Date, nullable=False)
    risk_score: Mapped[float] = mapped_column(Numeric(5, 2), nullable=False)
    risk_level: Mapped[str] = mapped_column(String(32), nullable=False)
    probability_of_default: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)

    business: Mapped[Business] = relationship("Business", back_populates="credit_assessments")

    __table_args__ = (
        Index("ix_credit_assessments_business_id", "business_id"),
        Index("ix_credit_assessments_assessment_date", "assessment_date"),
    )


class Alert(Base):
    """Risk / anomaly alerts raised by the monitoring pipeline or agents."""

    __tablename__ = "alerts"

    alert_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    business_id: Mapped[str] = mapped_column(ForeignKey("businesses.business_id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    alert_type: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    recommended_action: Mapped[str] = mapped_column(String(32), nullable=False)
    evidence: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False, default="monitoring")
    acknowledged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    business: Mapped[Business] = relationship("Business")

    __table_args__ = (
        Index("ix_alerts_business_id", "business_id"),
        Index("ix_alerts_severity", "severity"),
        Index("ix_alerts_alert_type", "alert_type"),
        Index("ix_alerts_created_at", "created_at"),
        Index("ix_alerts_acknowledged", "acknowledged"),
    )


class AgentDecision(Base):
    """Audit record of every specialist agent invocation and its outcome."""

    __tablename__ = "agent_decisions"

    decision_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    trace_id: Mapped[str] = mapped_column(String(36), nullable=False)
    business_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    agent_name: Mapped[str] = mapped_column(String(64), nullable=False)
    intent: Mapped[str | None] = mapped_column(String(64), nullable=True)
    question: Mapped[str | None] = mapped_column(Text, nullable=True)
    tools_used: Mapped[list | None] = mapped_column(JSON, nullable=True)
    output_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ok")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_agent_decisions_trace_id", "trace_id"),
        Index("ix_agent_decisions_business_id", "business_id"),
        Index("ix_agent_decisions_agent_name", "agent_name"),
        Index("ix_agent_decisions_created_at", "created_at"),
    )


class ModelPrediction(Base):
    """Every credit-risk model inference, with the reason codes that produced it."""

    __tablename__ = "model_predictions"

    prediction_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    business_id: Mapped[str] = mapped_column(ForeignKey("businesses.business_id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    probability_of_default: Mapped[float] = mapped_column(Numeric(6, 5), nullable=False)
    risk_score: Mapped[float] = mapped_column(Numeric(5, 2), nullable=False)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)
    features: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    reason_codes: Mapped[list | None] = mapped_column(JSON, nullable=True)

    business: Mapped[Business] = relationship("Business")

    __table_args__ = (
        Index("ix_model_predictions_business_id", "business_id"),
        Index("ix_model_predictions_created_at", "created_at"),
        Index("ix_model_predictions_risk_level", "risk_level"),
    )


class MonitoringRun(Base):
    """Record of each proactive monitoring sweep."""

    __tablename__ = "monitoring_runs"

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    businesses_scanned: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    alerts_created: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="running")
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    __table_args__ = (Index("ix_monitoring_runs_started_at", "started_at"),)
