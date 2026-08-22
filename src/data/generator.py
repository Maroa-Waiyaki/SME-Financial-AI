from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import pandas as pd

from src.data.constants import (
    BUSINESS_SUFFIXES,
    COUNTIES,
    CUSTOMER_TYPES,
    MONTH_SEASONALITY,
    PAYMENT_METHODS,
    PRODUCT_CATEGORIES,
    PROFILES,
    REGISTRATION_STATUS,
    SECTORS,
    TOWNS,
)

FIRST_NAMES = [
    "John", "Mary", "Peter", "Grace", "James", "Jane", "Daniel", "Lucy",
    "Michael", "Agnes", "Joseph", "Esther", "David", "Ann", "Paul", "Caroline",
    "George", "Sarah", "Samuel", "Rose", "Stephen", "Mercy", "Francis", "Lilian",
    "Alex", "Irene", "Kevin", "Joyce", "Charles", "Vivian", "Eric", "Betty",
]

LAST_NAMES = [
    "Mwangi", "Wanjiru", "Kamau", "Omondi", "Ochieng", "Mutua", "Njoroge", "Achieng",
    "Kariuki", "Muthoni", "Maina", "Wambui", "Odhiambo", "Ndungu", "Githinji", "Atieno",
    "Kiptoo", "Cherono", "Kipchirchir", "Langat", "Rotich", "Anyango", "Mugo", "Mbai",
]

SUPPLIER_PREFIXES = [
    "Japwa", "Beta", "Apex", "Prime", "Royal", "Swift", "Top", "Green",
    "Metro", "Global", "Unity", "East Africa", "Highland", "Coastal", "Savannah",
]

SUPPLIER_SUFFIXES = [
    "Suppliers", "Distributors", "Wholesalers", "Traders", "Limited", "Enterprises",
    "Solutions", "Services", "Logistics", "Hub",
]


def _id(prefix: str, index: int, width: int = 6) -> str:
    return f"{prefix}{index:0{width}}"


def _choice(rng: np.random.Generator, items: list[str], p: list[float] | None = None) -> str:
    return rng.choice(items, p=p).item()


def _month_days(month_start: pd.Timestamp) -> int:
    return month_start.days_in_month


class SMEDataGenerator:
    def __init__(
        self,
        n_businesses: int = 1000,
        n_months: int = 18,
        start_date: str = "2023-01-01",
        seed: int = 42,
        output_dir: str = "data/synthetic",
    ) -> None:
        self.n_businesses = n_businesses
        self.n_months = n_months
        self.start_date = pd.Timestamp(start_date)
        self.seed = seed
        self.output_dir = output_dir
        self.rng = np.random.default_rng(seed)

        self._business_counter = 0
        self._customer_counter = 0
        self._sale_counter = 0
        self._expense_counter = 0
        self._invoice_counter = 0
        self._loan_counter = 0
        self._transaction_counter = 0

        os.makedirs(self.output_dir, exist_ok=True)

    def generate(self) -> dict[str, pd.DataFrame]:
        print(f"Generating {self.n_businesses} businesses for {self.n_months} months...")
        self.businesses = self._generate_businesses()
        print(f"  -> {len(self.businesses)} businesses")

        self.customers = self._generate_customers(self.businesses)
        print(f"  -> {len(self.customers)} customers")

        self.sales = self._generate_sales(self.businesses, self.customers)
        print(f"  -> {len(self.sales)} sales")

        self.expenses = self._generate_expenses(self.businesses)
        print(f"  -> {len(self.expenses)} expenses")

        self.invoices = self._generate_invoices(self.businesses, self.customers)
        print(f"  -> {len(self.invoices)} invoices")

        self.loans = self._generate_loans(self.businesses)
        print(f"  -> {len(self.loans)} loans")

        self.transactions = self._generate_transactions(
            self.businesses, self.sales, self.expenses, self.invoices, self.loans
        )
        print(f"  -> {len(self.transactions)} transactions")

        self._save_csvs()
        return {
            "businesses": self.businesses,
            "customers": self.customers,
            "sales": self.sales,
            "expenses": self.expenses,
            "invoices": self.invoices,
            "loans": self.loans,
            "transactions": self.transactions,
        }

    def _next_business_id(self) -> str:
        self._business_counter += 1
        return _id("B", self._business_counter, 6)

    def _next_customer_id(self) -> str:
        self._customer_counter += 1
        return _id("C", self._customer_counter, 7)

    def _next_sale_id(self) -> str:
        self._sale_counter += 1
        return _id("S", self._sale_counter, 8)

    def _next_expense_id(self) -> str:
        self._expense_counter += 1
        return _id("E", self._expense_counter, 8)

    def _next_invoice_id(self) -> str:
        self._invoice_counter += 1
        return _id("I", self._invoice_counter, 8)

    def _next_loan_id(self) -> str:
        self._loan_counter += 1
        return _id("L", self._loan_counter, 6)

    def _next_transaction_id(self) -> str:
        self._transaction_counter += 1
        return _id("TX", self._transaction_counter, 9)

    def _random_time(self, date: pd.Timestamp, off_hours: bool = False) -> pd.Timestamp:
        if off_hours:
            hour = int(self.rng.integers(0, 6))
        else:
            hour = int(self.rng.integers(8, 19))
        minute = int(self.rng.integers(0, 60))
        return date + pd.Timedelta(hours=hour, minutes=minute)

    def _customer_name(self, customer_type: str) -> str:
        if customer_type == "individual":
            return f"{_choice(self.rng, FIRST_NAMES)} {_choice(self.rng, LAST_NAMES)}"
        town = _choice(self.rng, list(TOWNS.keys()))
        suffix = _choice(self.rng, ["Traders", "Supplies", "Enterprises", "Ltd", "Services"])
        return f"{town} {suffix}"

    def _supplier_name(self) -> str:
        return f"{_choice(self.rng, SUPPLIER_PREFIXES)} {_choice(self.rng, SUPPLIER_SUFFIXES)}"

    def _business_name(self, sector: str, county: str, rng: np.random.Generator) -> str:
        town = _choice(rng, TOWNS.get(county, [county]))
        suffix = _choice(rng, BUSINESS_SUFFIXES)
        return f"{town} {sector} {suffix}"

    def _assign_size(self, revenue: float, employees: int) -> str:
        if revenue >= 2_000_000 or employees > 50:
            return "medium"
        if revenue >= 200_000 or employees > 10:
            return "small"
        return "micro"

    def _generate_businesses(self) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        profile_names = list(PROFILES.keys())
        profile_weights = [PROFILES[p]["weight"] for p in profile_names]

        for _ in range(self.n_businesses):
            profile_name = _choice(self.rng, profile_names, profile_weights)
            profile = PROFILES[profile_name]

            sector = _choice(self.rng, SECTORS)

            # County loosely related to sector
            if sector in ["Agriculture"]:
                county = _choice(self.rng, COUNTIES, [0.05, 0.10, 0.05, 0.05, 0.20, 0.25, 0.10, 0.10, 0.05, 0.05])
            elif sector in ["Transport", "Construction"]:
                county = _choice(self.rng, COUNTIES, [0.30, 0.15, 0.10, 0.05, 0.15, 0.05, 0.10, 0.05, 0.05, 0.00])
            elif sector in ["Retail", "Wholesale", "Electronics", "Clothing", "Beauty", "Hospitality", "Restaurant"]:
                county = _choice(self.rng, COUNTIES, [0.45, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.03, 0.015, 0.005])
            else:
                county = _choice(self.rng, COUNTIES)

            base_revenue = int(np.exp(self.rng.normal(12.5, 0.9)))
            monthly_revenue = int(base_revenue * self.rng.lognormal(0.0, profile["volatility"]))

            if profile_name == "high_growth":
                monthly_revenue = int(monthly_revenue * self.rng.uniform(0.9, 1.5))
            elif profile_name == "struggling":
                monthly_revenue = int(monthly_revenue * self.rng.uniform(0.3, 0.8))
            elif profile_name == "seasonal":
                monthly_revenue = int(monthly_revenue * self.rng.uniform(0.7, 1.4))

            # Sector-level revenue scaling
            sector_multipliers = {
                "Retail": 0.8, "Wholesale": 1.2, "Restaurant": 0.7, "Transport": 0.9,
                "Agriculture": 0.6, "Professional services": 1.5, "Hardware": 1.0, "Clothing": 0.9,
                "Electronics": 1.3, "Beauty": 0.7, "Construction": 1.4, "Hospitality": 0.9, "General trade": 0.8,
            }
            monthly_revenue = int(monthly_revenue * sector_multipliers.get(sector, 1.0))
            monthly_revenue = max(monthly_revenue, 20_000)

            employees = max(1, int(monthly_revenue / self.rng.uniform(80_000, 180_000)))
            business_size = self._assign_size(monthly_revenue, employees)
            if business_size == "micro":
                employees = max(1, min(employees, 10))
            elif business_size == "small":
                employees = max(2, min(employees, 50))
            else:
                employees = max(3, min(employees, 200))

            age = int(self.rng.integers(1, 26))
            created_at = self.start_date - pd.DateOffset(years=age, months=int(self.rng.integers(0, 12)))

            records.append({
                "business_id": self._next_business_id(),
                "business_name": self._business_name(sector, county, self.rng),
                "sector": sector,
                "county": county,
                "business_age_years": age,
                "number_of_employees": employees,
                "monthly_revenue_estimate": monthly_revenue,
                "business_size": business_size,
                "registration_status": _choice(self.rng, REGISTRATION_STATUS, [0.6, 0.3, 0.1]),
                "profile": profile_name,
                "created_at": created_at.strftime("%Y-%m-%d"),
            })

        return pd.DataFrame(records)

    def _generate_customers(self, businesses: pd.DataFrame) -> pd.DataFrame:
        records: list[dict[str, Any]] = []

        for _, row in businesses.iterrows():
            n = int(self.rng.integers(3, 16))
            if row["business_size"] == "medium":
                n += 5
            elif row["business_size"] == "small":
                n += 2

            for _ in range(n):
                customer_type = _choice(self.rng, CUSTOMER_TYPES, [0.7, 0.3])
                since = pd.to_datetime(row["created_at"]) + pd.Timedelta(days=int(self.rng.integers(0, 365)))
                if since >= self.start_date:
                    since = self.start_date - pd.Timedelta(days=30)

                records.append({
                    "customer_id": self._next_customer_id(),
                    "business_id": row["business_id"],
                    "customer_name": self._customer_name(customer_type),
                    "customer_type": customer_type,
                    "location": _choice(self.rng, COUNTIES),
                    "customer_since": since.strftime("%Y-%m-%d"),
                    "credit_limit": int(row["monthly_revenue_estimate"] * self.rng.uniform(0.05, 0.25)),
                    "payment_terms": _choice(self.rng, ["Net 15", "Net 30", "Net 45"], [0.2, 0.6, 0.2]),
                })

        return pd.DataFrame(records)

    def _revenue_for_month(self, row: pd.Series, month_start: pd.Timestamp, month_index: int) -> float:
        profile = PROFILES[row["profile"]]
        base = row["monthly_revenue_estimate"]
        trend = (1 + profile["revenue_trend"]) ** month_index
        season = MONTH_SEASONALITY[month_start.month]
        if row["profile"] == "seasonal":
            season = 1.0 + (season - 1.0) * 1.5
        noise = self.rng.lognormal(0.0, profile["volatility"])
        return max(10_000.0, base * trend * season * noise)

    def _generate_sales(self, businesses: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
        records: list[dict[str, Any]] = []

        customer_map = customers.groupby("business_id")["customer_id"].apply(list).to_dict()
        customer_name_map = customers.set_index("customer_id")["customer_name"].to_dict()

        for _, row in businesses.iterrows():
            customers_for_business = customer_map.get(row["business_id"], [])
            if not customers_for_business:
                customers_for_business = [None]

            for m_idx in range(self.n_months):
                month_start = self.start_date + pd.DateOffset(months=m_idx)
                month_revenue = self._revenue_for_month(row, month_start, m_idx)

                # Stronger weekend dip for retail/restaurant; less for professional services
                days = _month_days(month_start)
                daily_base = month_revenue / days

                for day in range(days):
                    date = month_start + pd.Timedelta(days=day)
                    is_weekend = date.dayofweek >= 5
                    weekend_factor = 0.6 if is_weekend and row["sector"] in ["Retail", "Restaurant", "Hospitality"] else 1.0

                    daily_revenue = daily_base * weekend_factor * self.rng.lognormal(0.0, 0.25)
                    if daily_revenue < 100:
                        continue

                    # Number of sales for the day
                    avg_sale = max(1_000, month_revenue / self.rng.uniform(80, 150))
                    n_sales = max(1, min(3, int(daily_revenue / avg_sale)))

                    per_sale = daily_revenue / n_sales
                    for _ in range(n_sales):
                        total = per_sale * self.rng.lognormal(0.0, 0.15)
                        qty = int(self.rng.integers(1, 6))
                        unit_price = round(total / qty, 2)
                        customer_id = _choice(self.rng, customers_for_business)

                        records.append({
                            "sale_id": self._next_sale_id(),
                            "business_id": row["business_id"],
                            "customer_id": customer_id,
                            "date": date.strftime("%Y-%m-%d"),
                            "product_category": _choice(self.rng, PRODUCT_CATEGORIES),
                            "quantity": qty,
                            "unit_price": unit_price,
                            "total_amount": round(total, 2),
                            "payment_method": _choice(self.rng, PAYMENT_METHODS, [0.60, 0.15, 0.20, 0.05]),
                        })

        return pd.DataFrame(records)

    def _generate_expenses(self, businesses: pd.DataFrame) -> pd.DataFrame:
        records: list[dict[str, Any]] = []

        for _, row in businesses.iterrows():
            for m_idx in range(self.n_months):
                month_start = self.start_date + pd.DateOffset(months=m_idx)
                month_revenue = self._revenue_for_month(row, month_start, m_idx)
                profile = PROFILES[row["profile"]]
                expense_ratio = self.rng.normal(profile["expense_ratio_mean"], profile["expense_ratio_std"])
                expense_ratio = max(0.2, min(0.95, expense_ratio))
                total_expense = month_revenue * expense_ratio

                # Category weights and amounts
                rent = self.rng.uniform(8_000, 40_000) * (1.3 if row["business_size"] == "medium" else 1.0)
                salary_per_employee = self.rng.uniform(15_000, 40_000)
                salaries = row["number_of_employees"] * salary_per_employee
                transport = month_revenue * self.rng.uniform(0.03, 0.08)
                utilities = self.rng.uniform(3_000, 10_000)
                inventory = month_revenue * self.rng.uniform(0.20, 0.40) if row["sector"] in ["Retail", "Wholesale", "Electronics", "Clothing"] else month_revenue * self.rng.uniform(0.05, 0.15)
                marketing = month_revenue * self.rng.uniform(0.01, 0.05)
                taxes = month_revenue * self.rng.uniform(0.02, 0.05)
                mpesa_fees = month_revenue * self.rng.uniform(0.005, 0.015)
                bank_fees = self.rng.uniform(500, 2_500)
                maintenance = self.rng.uniform(2_000, 12_000)
                other = max(0, total_expense - (rent + salaries + transport + utilities + inventory + marketing + taxes + mpesa_fees + bank_fees + maintenance))

                amounts = {
                    "Rent": rent,
                    "Salaries": salaries,
                    "Transport": transport,
                    "Utilities": utilities,
                    "Inventory": inventory,
                    "Marketing": marketing,
                    "Taxes": taxes,
                    "M-Pesa fees": mpesa_fees,
                    "Bank fees": bank_fees,
                    "Maintenance": maintenance,
                    "Other": other,
                    "Loan repayment": 0.0,  # Filled from actual loan repayments later
                }

                for category, amount in amounts.items():
                    if amount <= 0:
                        continue
                    payment_method = _choice(self.rng, PAYMENT_METHODS, [0.45, 0.30, 0.20, 0.05])
                    day_offset = int(self.rng.integers(0, _month_days(month_start)))
                    date = (month_start + pd.Timedelta(days=day_offset)).strftime("%Y-%m-%d")
                    records.append({
                        "expense_id": self._next_expense_id(),
                        "business_id": row["business_id"],
                        "date": date,
                        "category": category,
                        "supplier": self._supplier_name(),
                        "amount": round(amount, 2),
                        "payment_method": payment_method,
                        "description": f"{category} for {month_start.strftime('%B %Y')}",
                    })

        return pd.DataFrame(records)

    def _generate_invoices(self, businesses: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
        records: list[dict[str, Any]] = []

        customer_map = customers.groupby("business_id")["customer_id"].apply(list).to_dict()

        for _, row in businesses.iterrows():
            customers_for_business = customer_map.get(row["business_id"], [])
            if not customers_for_business:
                continue

            profile = PROFILES[row["profile"]]

            for m_idx in range(self.n_months):
                month_start = self.start_date + pd.DateOffset(months=m_idx)
                month_revenue = self._revenue_for_month(row, month_start, m_idx)

                n_invoices = int(self.rng.integers(3, max(4, min(len(customers_for_business), 10))))
                total_invoice_amount = month_revenue * self.rng.uniform(0.10, 0.40)
                per_invoice = total_invoice_amount / n_invoices

                for _ in range(n_invoices):
                    customer_id = _choice(self.rng, customers_for_business)
                    invoice_date = month_start + pd.Timedelta(days=int(self.rng.integers(0, _month_days(month_start))))
                    due_date = invoice_date + pd.Timedelta(days=int(self.rng.integers(15, 91)))
                    amount = per_invoice * self.rng.lognormal(0.0, 0.2)

                    r = self.rng.random()
                    if r < (1 - profile["overdue_rate"]):
                        status = "paid"
                        amount_paid = amount
                    elif r < (1 - profile["overdue_rate"] + 0.4 * profile["overdue_rate"]):
                        status = "partial"
                        amount_paid = amount * self.rng.uniform(0.2, 0.8)
                    elif r < (1 - profile["overdue_rate"] + 0.7 * profile["overdue_rate"]):
                        status = "overdue"
                        amount_paid = amount * self.rng.uniform(0.0, 0.4)
                    else:
                        status = "unpaid"
                        amount_paid = 0.0

                    records.append({
                        "invoice_id": self._next_invoice_id(),
                        "business_id": row["business_id"],
                        "customer_id": customer_id,
                        "invoice_date": invoice_date.strftime("%Y-%m-%d"),
                        "due_date": due_date.strftime("%Y-%m-%d"),
                        "amount": round(amount, 2),
                        "amount_paid": round(amount_paid, 2),
                        "outstanding_amount": round(amount - amount_paid, 2),
                        "status": status,
                    })

        return pd.DataFrame(records)

    def _generate_loans(self, businesses: pd.DataFrame) -> pd.DataFrame:
        records: list[dict[str, Any]] = []

        for _, row in businesses.iterrows():
            profile = PROFILES[row["profile"]]
            if self.rng.random() > profile["loan_prob"]:
                continue

            n_loans = int(self.rng.integers(1, 3))
            for _ in range(n_loans):
                month_start = self.start_date + pd.DateOffset(months=int(self.rng.integers(0, max(1, self.n_months - 6))))
                term = int(self.rng.integers(6, 37))
                loan_amount = int(row["monthly_revenue_estimate"] * self.rng.uniform(3, 12))
                annual_rate = self.rng.uniform(0.12, 0.24)
                monthly_rate = annual_rate / 12
                monthly_repayment = (loan_amount * monthly_rate) / (1 - (1 + monthly_rate) ** (-term))

                # Determine status
                r = self.rng.random()
                if r < 0.50:
                    status = "active"
                    months_paid = int(self.rng.integers(0, term))
                elif r < 0.85:
                    status = "closed"
                    months_paid = term
                else:
                    status = "defaulted"
                    months_paid = int(self.rng.integers(0, term))

                outstanding = max(0.0, monthly_repayment * (term - months_paid))

                records.append({
                    "loan_id": self._next_loan_id(),
                    "business_id": row["business_id"],
                    "loan_amount": round(loan_amount, 2),
                    "interest_rate": round(annual_rate, 4),
                    "term_months": term,
                    "start_date": month_start.strftime("%Y-%m-%d"),
                    "outstanding_balance": round(outstanding, 2),
                    "monthly_repayment": round(monthly_repayment, 2),
                    "status": status,
                })

        return pd.DataFrame(records)

    def _generate_transactions(
        self,
        businesses: pd.DataFrame,
        sales: pd.DataFrame,
        expenses: pd.DataFrame,
        invoices: pd.DataFrame,
        loans: pd.DataFrame,
    ) -> pd.DataFrame:
        events: list[dict[str, Any]] = []

        customer_name_map = {}
        # Rebuild from existing data
        # (customer names not needed after final merge; omitted for speed)

        # Sales -> RECEIPT
        for _, row in sales.iterrows():
            date = pd.to_datetime(row["date"])
            events.append({
                "business_id": row["business_id"],
                "timestamp": self._random_time(date),
                "transaction_type": "RECEIPT",
                "amount": float(row["total_amount"]),
                "category": row["product_category"],
                "channel": row["payment_method"],
                "counterparty": row.get("customer_id", "Walk-in Customer"),
                "reference": f"SALE{row['sale_id'][1:]}",
                "location": None,
                "status": "completed",
            })

            # Fee for channel
            fee = self._channel_fee("RECEIPT", row["payment_method"], float(row["total_amount"]))
            if fee > 0:
                events.append({
                    "business_id": row["business_id"],
                    "timestamp": self._random_time(date) + pd.Timedelta(minutes=1),
                    "transaction_type": "FEE",
                    "amount": fee,
                    "category": f"{row['payment_method']} fee",
                    "channel": row["payment_method"],
                    "counterparty": "Service Provider",
                    "reference": f"FEE{row['sale_id'][1:]}",
                    "location": None,
                    "status": "completed",
                })

        # Expenses -> PAYMENT
        for _, row in expenses.iterrows():
            date = pd.to_datetime(row["date"])
            events.append({
                "business_id": row["business_id"],
                "timestamp": self._random_time(date),
                "transaction_type": "PAYMENT",
                "amount": float(row["amount"]),
                "category": row["category"],
                "channel": row["payment_method"],
                "counterparty": row["supplier"],
                "reference": f"EXP{row['expense_id'][1:]}",
                "location": None,
                "status": "completed",
            })
            fee = self._channel_fee("PAYMENT", row["payment_method"], float(row["amount"]))
            if fee > 0:
                events.append({
                    "business_id": row["business_id"],
                    "timestamp": self._random_time(date) + pd.Timedelta(minutes=1),
                    "transaction_type": "FEE",
                    "amount": fee,
                    "category": f"{row['payment_method']} fee",
                    "channel": row["payment_method"],
                    "counterparty": "Service Provider",
                    "reference": f"FEE{row['expense_id'][1:]}",
                    "location": None,
                    "status": "completed",
                })

        # Invoices paid -> RECEIPT
        for _, row in invoices.iterrows():
            if row["amount_paid"] <= 0:
                continue
            due = pd.to_datetime(row["due_date"])
            days_after = int(self.rng.integers(-10, 45))
            if row["status"] in ["overdue", "unpaid"]:
                days_after = int(self.rng.integers(15, 90))
            pay_date = due + pd.Timedelta(days=days_after)
            if pay_date < self.start_date or pay_date > self.start_date + pd.DateOffset(months=self.n_months):
                continue
            events.append({
                "business_id": row["business_id"],
                "timestamp": self._random_time(pay_date),
                "transaction_type": "RECEIPT",
                "amount": float(row["amount_paid"]),
                "category": "Invoice",
                "channel": _choice(self.rng, ["MPESA", "BANK"], [0.65, 0.35]),
                "counterparty": row["customer_id"],
                "reference": f"INV{row['invoice_id'][1:]}",
                "location": None,
                "status": "completed",
            })

        # Loans -> LOAN_REPAYMENT
        for _, row in loans.iterrows():
            if row["status"] == "defaulted":
                continue
            start = pd.to_datetime(row["start_date"])
            n_payments = int(row["term_months"])
            if row["status"] == "active":
                end_date = min(
                    start + pd.DateOffset(months=n_payments),
                    self.start_date + pd.DateOffset(months=self.n_months),
                )
                n_payments = max(0, (end_date.year - start.year) * 12 + end_date.month - start.month)
            for p in range(n_payments):
                pay_date = start + pd.DateOffset(months=p)
                if pay_date > self.start_date + pd.DateOffset(months=self.n_months):
                    break
                events.append({
                    "business_id": row["business_id"],
                    "timestamp": self._random_time(pay_date),
                    "transaction_type": "LOAN_REPAYMENT",
                    "amount": float(row["monthly_repayment"]),
                    "category": "Loan repayment",
                    "channel": "BANK",
                    "counterparty": "Lender",
                    "reference": f"LOAN{row['loan_id'][1:]}P{p:03d}",
                    "location": None,
                    "status": "completed",
                })

        # Inject business-level cash management / anomalies
        for _, row in businesses.iterrows():
            for m_idx in range(self.n_months):
                month_start = self.start_date + pd.DateOffset(months=m_idx)
                month_revenue = self._revenue_for_month(row, month_start, m_idx)

                # 0-2 deposits per month
                for _ in range(int(self.rng.integers(0, 3))):
                    day = int(self.rng.integers(0, _month_days(month_start)))
                    date = month_start + pd.Timedelta(days=day)
                    amount = month_revenue * self.rng.uniform(0.05, 0.25)
                    events.append({
                        "business_id": row["business_id"],
                        "timestamp": self._random_time(date),
                        "transaction_type": "DEPOSIT",
                        "amount": round(amount, 2),
                        "category": "Cash deposit",
                        "channel": _choice(self.rng, ["MPESA", "BANK"], [0.7, 0.3]),
                        "counterparty": "Cash Agent",
                        "reference": f"DEP{self._transaction_counter + len(events)}",
                        "location": None,
                        "status": "completed",
                    })

                # 0-2 withdrawals per month
                for _ in range(int(self.rng.integers(0, 3))):
                    day = int(self.rng.integers(0, _month_days(month_start)))
                    date = month_start + pd.Timedelta(days=day)
                    amount = month_revenue * self.rng.uniform(0.05, 0.20)
                    events.append({
                        "business_id": row["business_id"],
                        "timestamp": self._random_time(date),
                        "transaction_type": "WITHDRAWAL",
                        "amount": round(amount, 2),
                        "category": "Cash withdrawal",
                        "channel": "CASH",
                        "counterparty": "Agent",
                        "reference": f"WTH{self._transaction_counter + len(events)}",
                        "location": None,
                        "status": "completed",
                    })

                # Refunds
                if self.rng.random() < 0.05:
                    day = int(self.rng.integers(0, _month_days(month_start)))
                    date = month_start + pd.Timedelta(days=day)
                    amount = month_revenue * self.rng.uniform(0.01, 0.05)
                    events.append({
                        "business_id": row["business_id"],
                        "timestamp": self._random_time(date),
                        "transaction_type": "REFUND",
                        "amount": round(amount, 2),
                        "category": "Refund",
                        "channel": _choice(self.rng, ["MPESA", "BANK"], [0.6, 0.4]),
                        "counterparty": "Customer",
                        "reference": f"RFD{self._transaction_counter + len(events)}",
                        "location": None,
                        "status": "completed",
                    })

                # Anomalies
                if row["profile"] == "anomalous" and self.rng.random() < 0.15:
                    day = int(self.rng.integers(0, _month_days(month_start)))
                    date = month_start + pd.Timedelta(days=day)
                    amount = month_revenue * self.rng.uniform(1.5, 5.0)
                    events.append({
                        "business_id": row["business_id"],
                        "timestamp": self._random_time(date, off_hours=True),
                        "transaction_type": _choice(self.rng, ["PAYMENT", "WITHDRAWAL", "DEPOSIT"]),
                        "amount": round(amount, 2),
                        "category": "Unusual",
                        "channel": _choice(self.rng, ["MPESA", "BANK"]),
                        "counterparty": "Unknown Counterparty",
                        "reference": f"ANM{self._transaction_counter + len(events)}",
                        "location": _choice(self.rng, COUNTIES),
                        "status": "completed",
                    })

        # Assign balance
        events_df = pd.DataFrame(events)
        events_df = events_df.sort_values(["business_id", "timestamp"]).reset_index(drop=True)

        balances: dict[str, float] = {}
        records: list[dict[str, Any]] = []

        location_map = businesses.set_index("business_id")["county"].to_dict()

        for _, row in events_df.iterrows():
            business_id = row["business_id"]
            if business_id not in balances:
                biz = businesses[businesses["business_id"] == business_id].iloc[0]
                balances[business_id] = float(biz["monthly_revenue_estimate"]) * self.rng.uniform(0.3, 2.0)

            amount = round(float(row["amount"]), 2)
            ttype = row["transaction_type"]

            if ttype in ("RECEIPT", "DEPOSIT"):
                delta = amount
            elif ttype in ("PAYMENT", "WITHDRAWAL", "FEE", "LOAN_REPAYMENT", "REFUND"):
                delta = -amount
            elif ttype == "TRANSFER":
                delta = -amount
            else:
                delta = -amount

            balance_before = round(balances[business_id], 2)
            balance_after = round(balance_before + delta, 2)

            # Mark extreme overdrafts as failed; still record the computed balance
            if balance_after < -1_000_000:
                row["status"] = "failed"

            balances[business_id] = balance_after

            records.append({
                "transaction_id": self._next_transaction_id(),
                "business_id": business_id,
                "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "transaction_type": ttype,
                "amount": round(amount, 2),
                "balance_before": round(balance_before, 2),
                "balance_after": round(balance_after, 2),
                "category": row["category"],
                "channel": row["channel"],
                "counterparty": row["counterparty"],
                "reference": row["reference"],
                "location": row["location"] if row["location"] else location_map.get(business_id),
                "status": row["status"],
            })

        return pd.DataFrame(records)

    def _channel_fee(self, transaction_type: str, channel: str, amount: float) -> float:
        if channel == "MPESA":
            return max(5.0, amount * 0.005)
        if channel == "BANK":
            return 100.0 if transaction_type == "PAYMENT" else 0.0
        if channel == "CARD":
            return max(20.0, amount * 0.015)
        return 0.0

    def _save_csvs(self) -> None:
        for name, df in {
            "businesses": self.businesses,
            "customers": self.customers,
            "sales": self.sales,
            "expenses": self.expenses,
            "invoices": self.invoices,
            "loans": self.loans,
            "transactions": self.transactions,
        }.items():
            path = os.path.join(self.output_dir, f"{name}.csv")
            df.to_csv(path, index=False)
            print(f"Saved {path} ({len(df)} rows)")

        # Metadata about the synthetic run
        with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
            json.dump(
                {
                    "n_businesses": self.n_businesses,
                    "n_months": self.n_months,
                    "start_date": self.start_date.strftime("%Y-%m-%d"),
                    "seed": self.seed,
                    "note": "Synthetic data for Kenyan SMEs; not real businesses.",
                },
                f,
                indent=2,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Kenyan SME data.")
    parser.add_argument("--n-businesses", type=int, default=1000, help="Number of businesses")
    parser.add_argument("--n-months", type=int, default=18, help="Number of months of history")
    parser.add_argument("--start-date", type=str, default="2023-01-01")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data/synthetic")
    args = parser.parse_args()

    gen = SMEDataGenerator(
        n_businesses=args.n_businesses,
        n_months=args.n_months,
        start_date=args.start_date,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    gen.generate()


if __name__ == "__main__":
    main()
