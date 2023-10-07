import backtrader as bt
from BaseStrategy import BaseStrategy
import pandas as pd
from ml_models import bollinger_bands, stochastic_oscillator
from collections import deque
import torch

class MLStrategy (BaseStrategy):
    params = [
        ("model", None)
    ]
    
    def __init__(self):
        super().__init__()
        self.initialize_lists()
        self.actual_movements = []
        self.predictions = []
        # init rolling window
        self.rolling_window = deque (maxlen=60)

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

    
    def prenext(self): # this will be called before next method. for all data points before long SMA minimum period.
        self.add_cash()
        self.update_lists()
    
    # BUYING DECISIONS MADE HERE
    def next(self): # this won't be called until the longest period. that is the long SMA, which is 200 data points.
        self.add_cash()
        self.days_since_rebalance += 1 # add one to the count
        
        if self.days_since_rebalance >= self.p.rebalance_days: # execute rebalance and reset counter variable
            self.rebalance()
            self.days_since_rebalance = 0
            self.just_rebalanced = True
        
        current_data = [
            self.sma_short[0], self.sma_long[0], self.rsi[0], self.macd.macd[0], self.macd.signal[0],
            self.upper_bollinger[0], self.lower_bollinger[0], self.k_line[0], self.d_line[0]
        ]
        
        self.rolling_window.append(current_data)

        prediction = None

        if len(self.rolling_window) == 60:
            df = pd.DataFrame(self.rolling_window, columns=['SMA1', 'SMA2', 'RSI', 'MACD_Line', 'Signal_Line', 'Upper_Bollinger', 'Lower_Bollinger', 'K_Line', 'D_Line'])
            input_data = torch.tensor(df.values.reshape(1, 60, len(df.columns)), dtype=torch.float32)
            with torch.no_grad(): # don't compute gradients during inference to save memory and speed up predictions
                prediction = self.p.model(input_data)
                self.predictions.append(1 if prediction > 0.5 else 0)  # Assuming 1 is for "price will go up" and 0 otherwise
                # Capture the actual movement in the next timestep (up or down)
                if self.data._idx + 1 < len(self.data):  # Check if a next bar is available
                    next_close = self.data.close[1]
                else:
                    next_close = self.data.close[0]  # Current close as fallback

                movement = 1 if next_close > self.data.close[0] else 0
                self.actual_movements.append(movement)
        
        if prediction is not None and prediction.item() > 0.5:
            self.buy_dates.append(bt.num2date(self.data.datetime[0])) # add the date of when it bought
            self.buy()
        elif prediction is not None and self.position.size > 0 and prediction.item() < 0.5:
            self.sell_dates.append(bt.num2date(self.data.datetime[0])) # add the date of when it bought
            self.sell() 
        
        self.update_lists()
