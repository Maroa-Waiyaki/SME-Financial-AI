from __future__ import annotations

from django.urls import path

from . import views

urlpatterns = [
    path("login/", views.DashboardLoginView.as_view(), name="login"),
    path("logout/", views.DashboardLogoutView.as_view(), name="logout"),
    path("", views.dashboard_index, name="index"),
    path("analytics/", views.analytics_page, name="analytics"),
    path("transactions/", views.transactions_page, name="transactions"),
    path("chat/", views.chat_page, name="chat"),
]
