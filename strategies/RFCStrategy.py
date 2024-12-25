import backtrader as bt
from strategies.BaseStrategy import BaseStrategy
import pandas as pd
from collections import deque
import torch
import numpy as np

class RFCStrategy (BaseStrategy):
    params = (
        ("model", None),
    )
    
    def __init__(self):
        super().__init__()
        self.initialize_lists()
        self.previous_close = None
        self.actual_movements = []
        self.predictions = []
        # Access the indicators from the custom data feed
        self.sma1 = self.data.sma1
        self.sma2 = self.data.sma2
        self.rsi = self.data.rsi
        self.ema1 = self.data.ema1
        self.ema2 = self.data.ema2
        self.volatility = self.data.volatility
        self.roc = self.data.roc
        self.atr = self.data.atr
    
    def create_prediction_dataframe(self, ):
        # Create a DataFrame from prediction_data with the correct column names
        prediction_data_df = pd.DataFrame([[
            self.data.close[0],  # Close
            self.data.high[0],   # High
            self.data.low[0],    # Low
            self.data.trade_count[0],
            self.data.open[0],   # Open
            self.data.volume[0], # Volume
            self.data.vwap[0],   # Vwap
            self.sma1[0],        # SMA1
            self.sma2[0],        # SMA2
            self.rsi[0],         # RSI
            self.ema1[0],        # EMA1
            self.ema2[0],        # EMA2
            self.volatility[0],  # Volatility
            self.roc[0],         # ROC
            self.atr[0]          # ATR
        ]], columns=[
            'close', 'high', 'low', 'trade_count', 'open', 'volume', 'vwap', 
            'sma1', 'sma2', 'rsi', 'ema1', 'ema2', 'volatility', 'roc', 'atr'
        ])
        return prediction_data_df

    def prenext(self): # this will be called before next method. for all data points before long SMA minimum period.
        self.add_cash()
        self.update_lists()
    
    # iterate through each candlestick and execute the strategy
    def next(self): 
        self.add_cash()
        self.days_since_rebalance += 1 # add one to the count
        
        if self.days_since_rebalance >= self.p.rebalance_days: # execute rebalance and reset counter variable
            self.rebalance()
            self.days_since_rebalance = 0
            self.just_rebalanced = True
        
        prediction_data_df = self.create_prediction_dataframe()

        # Check for NaN values in prediction data
        if prediction_data_df.isnull().any().any():
            current_date = self.data.datetime.date(0)  # Get the current date
            print(f"Skipping prediction for {current_date} due to NaN values in data")
            return  # Skip this prediction

        # Prediction logic
        with torch.no_grad():  # don't compute gradients during inference
            prediction = self.p.model.predict(prediction_data_df)

            # ensure the prediction value is extracted correctly, whether the model returns a list, numpy array, or single value.
            prediction_value = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction

            self.predictions.append(1 if prediction_value > 0.5 else 0)
            current_close = self.data.close[0]
            previous_close = self.previous_close if self.previous_close is not None else current_close
            movement = 1 if previous_close < current_close else 0
            self.actual_movements.append(movement)
        
        if prediction is not None and prediction_value > 0.5:
            self.buy_dates.append(bt.num2date(self.data.datetime[0])) # add the date of when it bought
            self.buy()
        elif prediction is not None and self.position.size > 0 and prediction_value < 0.5:
            self.sell_dates.append(bt.num2date(self.data.datetime[0])) # add the date of when it bought
            self.sell() 
        
        self.update_lists()
        self.previous_close = self.data.close[0] # update previous_close as the value of the close right now.
