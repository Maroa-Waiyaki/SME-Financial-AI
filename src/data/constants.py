from __future__ import annotations

COUNTIES = [
    "Nairobi",
    "Kiambu",
    "Mombasa",
    "Kisumu",
    "Nakuru",
    "Uasin Gishu",
    "Machakos",
    "Kajiado",
    "Murang'a",
    "Nyeri",
]

SECTORS = [
    "Retail",
    "Wholesale",
    "Restaurant",
    "Transport",
    "Agriculture",
    "Professional services",
    "Hardware",
    "Clothing",
    "Electronics",
    "Beauty",
    "Construction",
    "Hospitality",
    "General trade",
]

BUSINESS_SUFFIXES = [
    "Stores",
    "Traders",
    "Enterprises",
    "Solutions",
    "Hub",
    "Mart",
    "Centre",
    "Services",
    "Supplies",
    "Point",
]

REGISTRATION_STATUS = ["registered", "informal", "limited"]

BUSINESS_SIZES = ["micro", "small", "medium"]

CUSTOMER_TYPES = ["individual", "business"]

PRODUCT_CATEGORIES = [
    "Electronics",
    "Clothing",
    "Food & Beverages",
    "Household",
    "Construction Materials",
    "Agricultural Inputs",
    "Beauty Products",
    "Transport Services",
    "Professional Fees",
    "Other",
]

EXPENSE_CATEGORIES = [
    "Rent",
    "Salaries",
    "Transport",
    "Utilities",
    "Inventory",
    "Marketing",
    "Taxes",
    "M-Pesa fees",
    "Bank fees",
    "Loan repayment",
    "Maintenance",
    "Other",
]

TRANSACTION_TYPES = [
    "PAYMENT",
    "RECEIPT",
    "TRANSFER",
    "WITHDRAWAL",
    "DEPOSIT",
    "FEE",
    "REFUND",
    "LOAN_REPAYMENT",
]

CHANNELS = ["MPESA", "BANK", "CASH", "CARD"]

PAYMENT_METHODS = ["MPESA", "BANK", "CASH", "CARD"]

INVOICE_STATUS = ["paid", "partial", "overdue", "unpaid"]

LOAN_STATUS = ["active", "closed", "defaulted"]

# Business behavioral profiles with probabilities and financial characteristics
PROFILES = {
    "healthy": {
        "weight": 0.45,
        "revenue_trend": 0.02,  # monthly growth
        "expense_ratio_mean": 0.55,
        "expense_ratio_std": 0.08,
        "volatility": 0.15,
        "overdue_rate": 0.05,
        "loan_prob": 0.10,
        "anomaly_prob": 0.02,
    },
    "high_growth": {
        "weight": 0.20,
        "revenue_trend": 0.08,
        "expense_ratio_mean": 0.65,
        "expense_ratio_std": 0.10,
        "volatility": 0.30,
        "overdue_rate": 0.15,
        "loan_prob": 0.30,
        "anomaly_prob": 0.03,
    },
    "seasonal": {
        "weight": 0.15,
        "revenue_trend": 0.00,
        "expense_ratio_mean": 0.60,
        "expense_ratio_std": 0.12,
        "volatility": 0.35,
        "overdue_rate": 0.10,
        "loan_prob": 0.15,
        "anomaly_prob": 0.02,
    },
    "struggling": {
        "weight": 0.15,
        "revenue_trend": -0.03,
        "expense_ratio_mean": 0.80,
        "expense_ratio_std": 0.15,
        "volatility": 0.25,
        "overdue_rate": 0.40,
        "loan_prob": 0.45,
        "anomaly_prob": 0.05,
    },
    "anomalous": {
        "weight": 0.05,
        "revenue_trend": 0.01,
        "expense_ratio_mean": 0.60,
        "expense_ratio_std": 0.10,
        "volatility": 0.40,
        "overdue_rate": 0.20,
        "loan_prob": 0.25,
        "anomaly_prob": 0.40,
    },
}

# Seasonal multipliers per month (1 = baseline)
MONTH_SEASONALITY = {
    1: 0.85,
    2: 0.80,
    3: 0.95,
    4: 1.00,
    5: 1.05,
    6: 1.10,
    7: 1.05,
    8: 1.00,
    9: 1.10,
    10: 1.20,
    11: 1.30,
    12: 1.40,
}

# Synthetic town/neighborhood names per county
TOWNS = {
    "Nairobi": ["CBD", "Westlands", "Eastleigh", "Kilimani", "Karen", "Ngara"],
    "Kiambu": ["Thika", "Kiambu Town", "Ruiru", "Juja", "Kikuyu"],
    "Mombasa": ["Mombasa CBD", "Nyali", "Likoni", "Changamwe", "Kisauni"],
    "Kisumu": ["Kisumu CBD", "Milimani", "Nyalenda", "Manyatta"],
    "Nakuru": ["Nakuru CBD", "Naivasha", "Molo", "Gilgil"],
    "Uasin Gishu": ["Eldoret", "Turbo", "Moiben", "Soy"],
    "Machakos": ["Machakos Town", "Athi River", "Mlolongo"],
    "Kajiado": ["Kajiado Town", "Kitengela", "Ngong", "Ongata Rongai"],
    "Murang'a": ["Murang'a Town", "Kangema", "Maragua"],
    "Nyeri": ["Nyeri Town", "Karatina", "Othaya"],
}
