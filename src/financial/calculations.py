from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Iterable

MONEY_QUANT = Decimal("0.01")
PCT_QUANT = Decimal("0.01")


def _money(value: Decimal) -> Decimal:
    return value.quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)


def _pct(value: Decimal) -> Decimal:
    return value.quantize(PCT_QUANT, rounding=ROUND_HALF_UP)


def revenue(receipts: Iterable[Decimal]) -> Decimal:
    return _money(sum(receipts, Decimal("0")))


def expenses(items: Iterable[Decimal]) -> Decimal:
    return _money(sum(items, Decimal("0")))


def profit(rev: Decimal, exp: Decimal) -> Decimal:
    return _money(rev - exp)


def profit_margin(rev: Decimal, exp: Decimal) -> Decimal:
    if rev <= 0:
        return Decimal("0")
    return _pct((rev - exp) / rev * Decimal("100"))


def cash_inflow(receipts: Iterable[Decimal], deposits: Iterable[Decimal]) -> Decimal:
    return _money(sum(receipts, Decimal("0")) + sum(deposits, Decimal("0")))


def cash_outflow(
    payments: Iterable[Decimal],
    withdrawals: Iterable[Decimal],
    fees: Iterable[Decimal],
    loan_repayments: Iterable[Decimal],
    refunds: Iterable[Decimal],
) -> Decimal:
    return _money(
        sum(payments, Decimal("0"))
        + sum(withdrawals, Decimal("0"))
        + sum(fees, Decimal("0"))
        + sum(loan_repayments, Decimal("0"))
        + sum(refunds, Decimal("0"))
    )


def net_cash_flow(inflow: Decimal, outflow: Decimal) -> Decimal:
    return _money(inflow - outflow)


def expense_ratio(exp: Decimal, rev: Decimal) -> Decimal:
    if rev <= 0:
        return Decimal("0")
    return _pct(exp / rev * Decimal("100"))


def revenue_growth(current: Decimal, previous: Decimal) -> Decimal:
    if previous <= 0:
        return Decimal("0")
    return _pct((current - previous) / previous * Decimal("100"))


def outstanding_receivables(invoices: Iterable[Decimal]) -> Decimal:
    return _money(sum(invoices, Decimal("0")))


def debt_obligations(loans: Iterable[Decimal]) -> Decimal:
    return _money(sum(loans, Decimal("0")))


def current_ratio(assets: Decimal, liabilities: Decimal) -> Decimal:
    if liabilities <= 0:
        return Decimal("0")
    return _money(assets / liabilities)


def expense_change(current: Decimal, previous: Decimal) -> Decimal:
    return _money(current - previous)
