from __future__ import annotations

from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView, LogoutView
from django.shortcuts import render

from src.database.engine import get_db
from src.database.models import Business
from tools.financial import get_business_profile, get_financial_summary


class DashboardLoginView(LoginView):
    template_name = "dashboard/login.html"


class DashboardLogoutView(LogoutView):
    next_page = "/login/"


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
                    db, business_id, "2023-01-01", "2023-12-31"
                )
    except Exception as exc:
        context["error"] = f"Could not load data: {exc}"

    return render(request, "dashboard/index.html", context)


@login_required
def analytics_page(request):
    return render(request, "dashboard/analytics.html")


@login_required
def transactions_page(request):
    return render(request, "dashboard/transactions.html")


@login_required
def chat_page(request):
    return render(request, "dashboard/chat.html")
