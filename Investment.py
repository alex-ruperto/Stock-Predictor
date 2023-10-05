import backtrader as bt
import pandas as pd

class WeeklyCapitalInjectionAnalyzer(bt.Analyzer):
    params = (
        ("injection_amount", 100.0),  # Amount to inject into the account every week
    )

    def __init__(self):
        self.last_injection_date = None

    def start(self):
        self.last_injection_date = pd.Timestamp(self.data.datetime[0])

    def next(self):
        current_datetime = pd.Timestamp(self.data.datetime[0])
        
        # Check if a week has passed since the last injection
        if (current_datetime - self.last_injection_date).days >= 7:
            # A week has passed, inject capital into the account
            cash_to_inject = self.params.injection_amount
            self.strategy.broker.add_cash(cash_to_inject)
            self.last_injection_date = current_datetime
