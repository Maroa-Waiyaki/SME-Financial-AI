from __future__ import annotations

from decimal import Decimal

import pytest

from src.financial import calculations as calc


def test_revenue_sums_receipts() -> None:
    receipts = [Decimal("1000.00"), Decimal("2500.50"), Decimal("0.00")]
    assert calc.revenue(receipts) == Decimal("3500.50")


def test_profit() -> None:
    rev = Decimal("10000.00")
    exp = Decimal("6500.00")
    assert calc.profit(rev, exp) == Decimal("3500.00")


def test_profit_margin() -> None:
    rev = Decimal("10000.00")
    exp = Decimal("6500.00")
    assert calc.profit_margin(rev, exp) == Decimal("35.00")


def test_profit_margin_zero_revenue() -> None:
    assert calc.profit_margin(Decimal("0"), Decimal("100")) == Decimal("0")


def test_cash_flow() -> None:
    receipts = [Decimal("1000"), Decimal("500")]
    deposits = [Decimal("100")]
    payments = [Decimal("300")]
    withdrawals = [Decimal("100")]
    fees = [Decimal("20")]
    loan_repayments = [Decimal("0")]
    refunds = [Decimal("10")]

    inflow = calc.cash_inflow(receipts, deposits)
    outflow = calc.cash_outflow(payments, withdrawals, fees, loan_repayments, refunds)
    assert inflow == Decimal("1600.00")
    assert outflow == Decimal("430.00")
    assert calc.net_cash_flow(inflow, outflow) == Decimal("1170.00")


def test_expense_ratio() -> None:
    assert calc.expense_ratio(Decimal("6000"), Decimal("10000")) == Decimal("60.00")


def test_revenue_growth() -> None:
    assert calc.revenue_growth(Decimal("12000"), Decimal("10000")) == Decimal("20.00")


def test_outstanding_receivables() -> None:
    assert calc.outstanding_receivables([Decimal("1000"), Decimal("500.50")]) == Decimal("1500.50")


def test_debt_obligations() -> None:
    assert calc.debt_obligations([Decimal("3000"), Decimal("1200")]) == Decimal("4200.00")
