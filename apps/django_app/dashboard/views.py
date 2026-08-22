from __future__ import annotations

from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView, LogoutView
from django.shortcuts import render

from src.database.engine import get_db
from tools.financial import get_business_profile, get_financial_summary


class DashboardLoginView(LoginView):
    template_name = "dashboard/login.html"


class DashboardLogoutView(LogoutView):
    next_page = "/login/"


@login_required
def dashboard_index(request):
    business_id = request.GET.get("business_id") or request.user.username
    context: dict = {"business_id": business_id}

    try:
        db = next(get_db())
        try:
            profile = get_business_profile(db, business_id)
            if profile:
                context["profile"] = profile
                context["summary"] = get_financial_summary(
                    db, business_id, "2023-01-01", "2023-12-31"
                )
        finally:
            db.close()
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
