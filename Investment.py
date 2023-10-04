import backtrader as bt
import pandas as pd

class WeeklyCapitalInjectionAnalyzer(bt.Analyzer):
    params = (
        ("injection_amount", 100.0),  # Amount to inject into the account every week
    )

    def __init__(self):
        self.week_start_cash = 0.0
        self.last_year = None

    def start(self):
        self.week_start_cash = self.strategy.broker.get_cash()
        self.last_year = None

    def prenext(self):
        # Store the initial cash at the beginning of the week
        self.week_start_cash = self.strategy.broker.get_cash()
        self.last_year = None

    def next(self):
        current_datetime = pd.Timestamp(self.data.datetime[0])
        current_year = current_datetime.year

        if self.last_year is None:
            self.last_year = current_year

        if current_year != self.last_year:
            # A new year has started, reset the cash
            self.week_start_cash = self.strategy.broker.get_cash()
            self.last_year = current_year

        current_week = current_datetime.week  # Get the week number

        if current_week != pd.Timestamp(self.data.datetime[-1]).week:
            # A week has passed, inject capital into the account
            cash_to_inject = self.params.injection_amount
            self.strategy.broker.add_cash(cash_to_inject)
            self.week_start_cash = self.strategy.broker.get_cash()