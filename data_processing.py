import backtrader as bt
import alpaca_trade_api as tradeapi
from alpaca_trade_api import TimeFrame
import alpacaconfig as config
import pandas as pd
from random_forest_model import train_random_forest_model
from Strategies.RFCStrategy import RFCStrategy
from random_forest_model import preprocess_data

alpaca_api = tradeapi.REST(config.ALPACA_KEY, config.ALPACA_SECRET_KEY, base_url=config.APCA_API_BASE_URL)

def backtest(ticker): # backtest function for an individual stock
    # get data from alpaca
    stock_data = alpaca_api.get_bars(ticker, TimeFrame.Hour, start="2022-01-01", end="2023-01-01").df
    if stock_data.empty:
        raise ValueError("No data fetched from Alpaca.")
    
    # Prepare data for backtrader
    stock_data.columns = [col.lower() for col in stock_data.columns]  # Ensure column names are in lowercase
    
    # Preprocess data with indicators
    preprocessed_data = preprocess_data(stock_data)

    data_feed = CustomDataWithIndicators(dataname=preprocessed_data)
    
    # Initialize Cerebro engine
    cerebro = bt.Cerebro()
    cerebro.adddata(data_feed)
    cerebro.broker.set_cash(100)

    # Train model and add strategy to Cerebro
    print("Training Random Forest Classifier Model for " + ticker + "...")
    model = train_random_forest_model(preprocessed_data)
    cerebro.addstrategy(RFCStrategy, model=model)
    # Run backtest
    print("Running backtest for " + ticker + "..."	)
    strategies = cerebro.run()
    strategy = strategies[0]

    # Extract strategy data for analysis
    dates = stock_data.index.tolist()
    closes = strategy.data.close.array
    sma_short = strategy.sma1.array
    sma_long = strategy.sma2.array
    rsi = strategy.rsi.array
    ema_short = strategy.ema1.array
    ema_long = strategy.ema2.array      
    volatility = strategy.volatility.array 
    roc = strategy.roc.array               
    atr = strategy.atr.array

    # Extract account and trading values
    cash_values = strategy.cash_values
    account_values = strategy.account_values
    position_sizes = strategy.position_sizes

    # Extract and convert naive buy and sell datetime objects to timezone-aware Timestamps in UTC
    buys_x = [pd.Timestamp(date).tz_localize('UTC') for date in strategy.buy_dates]
    sells_x = [pd.Timestamp(date).tz_localize('UTC') for date in strategy.sell_dates]

    buys_y = [closes[dates.index(date)] if date in dates else None for date in buys_x]
    sells_y = [closes[dates.index(date)] if date in dates else None for date in sells_x]

    print("Backtest for " + ticker + " completed.")

    return dates, closes, sma_short, sma_long, rsi, ema_short, ema_long, volatility, roc, atr, cash_values, account_values, position_sizes, buys_x, buys_y, sells_x, sells_y

class CustomDataWithIndicators(bt.feeds.PandasData):
    lines = (
        'sma1', 'sma2', 'rsi', 'ema1', 'ema2', 'volatility', 'roc', 'atr',
        'close', 'high', 'low', 'open', 'volume', 'trade_count', 'vwap'
    )
    params = (
        ('sma1', -1),
        ('sma2', -1),
        ('rsi', -1),
        ('ema1', -1),
        ('ema2', -1),
        ('volatility', -1),
        ('roc', -1),
        ('atr', -1),
        ('close', -1),
        ('high', -1),
        ('low', -1),
        ('open', -1),
        ('volume', -1),
        ('trade_count', -1),
        ('vwap', -1),
    )
