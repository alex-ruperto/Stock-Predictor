import backtrader as bt
from Strategies.BaseStrategy import BaseStrategy
import pandas as pd
from collections import deque
import torch

class MLStrategy (BaseStrategy):
    params = [
        ("model", None)
    ]
    
    def __init__(self):
        super().__init__()
        self.initialize_lists()
        self.previous_close = None
        self.actual_movements = []
        self.predictions = []
        # init rolling window
        self.rolling_window = deque (maxlen=60)

        # Simple Moving Average
        self.sma1 = bt.indicators.SimpleMovingAverage(self.data.close, period=50)
        self.sma2 = bt.indicators.SimpleMovingAverage(self.data.close, period=100)

        # RSI
        self.rsi = bt.indicators.RelativeStrengthIndex(period=14)
        
        # EMA
        self.ema1 = bt.indicators.EMA(self.data.close, period=12)
        self.ema2 = bt.indicators.EMA(self.data.close, period=26)       

        # Volatility
        self.volatility = bt.indicators.StandardDeviation(self.data.close, period=14)

        # Price Rate of Change
        self.roc = bt.indicators.RateOfChange(self.data.close, period=10)

        # Average True Range
        self.atr = bt.indicators.AverageTrueRange(self.data, period=14) 
        
    
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
            self.data.close[0], 
            self.data.high[0], 
            self.data.low[0], 
            self.data.trade_count[0],
            self.data.open[0],
            self.data.volume[0],
            self.data.vwap[0],
            self.sma1[0], 
            self.sma2[0],
            self.rsi[0],
            self.ema1[0],
            self.ema2[0],
            self.volatility[0],
            self.roc[0],
            self.atr[0]
        ]
        
        self.rolling_window.append(current_data)

        prediction = None

        if len(self.rolling_window) == 60:
            df = pd.DataFrame(self.rolling_window, columns=[
                'Close', 'High', 'Low', 'Trade_count', 'Open', 'Volume', 'Vwap',
                'SMA1', 'SMA2', 'RSI', 'EMA1', 'EMA2', 
                'Volatility', 'ROC', 'ATR'
            ])
            # Ensure DataFrame is used directly for prediction, maintaining feature names
            with torch.no_grad():  # don't compute gradients during inference
                prediction = self.p.model.predict(df.iloc[-1:])  # Use DataFrame directly
                self.predictions.append(1 if prediction > 0.5 else 0)
                current_close = self.data.close[0]
                previous_close = self.previous_close if self.previous_close is not None else current_close
                movement = 1 if previous_close < current_close else 0
                self.actual_movements.append(movement)
        
        if prediction is not None and prediction.item() > 0.5:
            self.buy_dates.append(bt.num2date(self.data.datetime[0])) # add the date of when it bought
            print("Buying on " + str(bt.num2date(self.data.datetime[0])))
            self.buy()
        elif prediction is not None and self.position.size > 0 and prediction.item() < 0.5:
            self.sell_dates.append(bt.num2date(self.data.datetime[0])) # add the date of when it bought
            print("Selling on " + str(bt.num2date(self.data.datetime[0])))	
            self.sell() 
        
        self.update_lists()
        self.previous_close = self.data.close[0] # update previous_close as the value of the close right now.
