import backtrader as bt
from BaseStrategy import BaseStrategy
import pandas as pd

class MLStrategy (BaseStrategy):
    params = [
        ("model", None)
    ]
    
    def __init__(self):
        super().__init__()
        self.initialize_lists(200)
        # Simple Moving Average
        self.sma1 = bt.indicators.SimpleMovingAverage(self.data.close, period=50)
        self.sma2 = bt.indicators.SimpleMovingAverage(self.data.close, period=200)
        # RSI
        self.rsi = bt.indicators.RelativeStrengthIndex(period=14)
        # MACD
        self.macd = bt.indicators.MACD(self.data.close, period_me1=12, period_me2=26, period_signal=9)
        
    def next(self):
        current_data = [self.sma1[0], self.sma2[0], self.rsi[0], self.macd.macd[0], self.macd.signal[0]]
        prediction = None
        if self.params.model is not None:
            df = pd.DataFrame([current_data], columns=['SMA1', 'SMA2', 'RSI', 'MACD_Line', 'Signal_Line'])
            prediction = self.params.model.predict(df)[0]
        
        if prediction == 1:  # If the model predicts the stock will go up.
            self.buy_dates.append(bt.num2date(self.data.datetime[0])) # add the date of when it bought
            self.buy()
        elif self.position.size > 0: 
            self.sell_dates.append(bt.num2date(self.data.datetime[0])) # add the date of when it bought
            self.sell()
        
        self.update_lists()




