import backtrader as bt
from BaseStrategy import BaseStrategy
import pandas as pd
from ml_models import bollinger_bands, stochastic_oscillator

class MLStrategy (BaseStrategy):
    params = [
        ("model", None)
    ]
    
    def __init__(self):
        super().__init__()
        self.initialize_lists(200)
        # Simple Moving Average
        self.sma_short = bt.indicators.SimpleMovingAverage(self.data.close, period=50)
        self.sma_long = bt.indicators.SimpleMovingAverage(self.data.close, period=200)

        # RSI
        self.rsi = bt.indicators.RelativeStrengthIndex(period=14)
        
        # MACD
        self.macd = bt.indicators.MACD(self.data.close, period_me1=12, period_me2=26, period_signal=9)
        
        # Bollinger Bands
        self.bollinger = bt.indicators.BollingerBands(self.data.close, period=20, devfactor=2)
        self.upper_bollinger = self.bollinger.lines.top
        self.lower_bollinger = self.bollinger.lines.bot

        # Stochastic Oscillator
        self.stochastic = bt.indicators.Stochastic(self.data)
        self.k_line = self.stochastic.lines.percK
        self.d_line = self.stochastic.lines.percD
        
    def next(self):
        current_data = [
            self.sma_short[0], self.sma_long[0], self.rsi[0], self.macd.macd[0], self.macd.signal[0],
            self.upper_bollinger[0], self.lower_bollinger[0], self.k_line[0], self.d_line[0]
        ]
        prediction = None
        if self.params.model is not None:
            df = pd.DataFrame([current_data], columns=['SMA1', 'SMA2', 'RSI', 'MACD_Line', 'Signal_Line', 'Upper_Bollinger', 'Lower_Bollinger', 'K_Line', 'D_Line'])
            prediction = self.params.model.predict(df)[0]
        
        if prediction == 1:  # If the model predicts the stock will go up.
            self.buy_dates.append(bt.num2date(self.data.datetime[0])) # add the date of when it bought
            self.buy()
        elif self.position.size > 0: 
            self.sell_dates.append(bt.num2date(self.data.datetime[0])) # add the date of when it bought
            self.sell()
        
        self.update_lists()
