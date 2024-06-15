# predictor/urls.py
from django.urls import path
from .views import backtest_view

urlpatterns = [
    path('backtest/<str:ticker>/', backtest_view, name='backtest_view'),
]
