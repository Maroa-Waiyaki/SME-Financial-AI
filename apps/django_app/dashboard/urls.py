from __future__ import annotations

from django.urls import path

from . import views

urlpatterns = [
    path("", views.landing_page, name="landing"),
    path("login/", views.DashboardLoginView.as_view(), name="login"),
    path("logout/", views.DashboardLogoutView.as_view(), name="logout"),
    path("dashboard/", views.dashboard_index, name="index"),
    path("analytics/", views.analytics_page, name="analytics"),
    path("transactions/", views.transactions_page, name="transactions"),
    path("reports/", views.reports_page, name="reports"),
    path("reports/export/pptx/", views.export_pptx_view, name="export_pptx"),
    path("reports/export/xlsx/", views.export_xlsx_view, name="export_xlsx"),
    path("documents/", views.documents_page, name="documents"),
    path("chat/", views.chat_page, name="chat"),
]
