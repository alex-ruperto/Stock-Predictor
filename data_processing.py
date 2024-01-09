import backtrader as bt
import alpaca_trade_api as tradeapi
from alpaca_trade_api import TimeFrame
import alpacaconfig as config
import pandas as pd
from random_forest_model import train_random_forest_model
from Strategies.MLStrategy import MLStrategy

alpaca_api = tradeapi.REST(config.ALPACA_KEY, config.ALPACA_SECRET_KEY, base_url=config.APCA_API_BASE_URL)

def backtest(ticker): # backtest function for an individual stock
    # get data from alpaca
    stock_data = alpaca_api.get_bars(ticker, TimeFrame.Hour, start="2022-06-01", end="2023-01-01").df
    if stock_data.empty:
        raise ValueError("No data fetched from Alpaca.")
    
    # Prepare data for backtrader
    stock_data.columns = [col.lower() for col in stock_data.columns]  # Ensure column names are in lowercase
    data = CustomData(dataname=stock_data)

    # Initialize Cerebro engine
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    print("Training Random Forest Classifier Model for " + ticker + "...")
    # Train model and add strategy to Cerebro
    model = train_random_forest_model(stock_data)
    cerebro.addstrategy(MLStrategy, model=model)

    # Run backtest
    print("Running backtest for " + ticker + "..."	)
    strategies = cerebro.run()
    strategy = strategies[0]

    # Extract strategy data for analysis
    dates = stock_data.index.tolist()
    closes = stock_data['Close'].tolist()
    sma_short = [strategy.sma1[0] for _ in dates]
    sma_long = [strategy.sma2[0] for _ in dates]
    rsi = [strategy.rsi[0] for _ in dates]   
    ema_short = [strategy.ema1[0] for _ in dates]
    ema_long = [strategy.ema2[0] for _ in dates]           
    volatility = [strategy.volatility[0] for _ in dates]  
    roc = [strategy.roc[0] for _ in dates]               
    atr = [strategy.atr[0] for _ in dates]

    # Extract account and trading values
    cash_values = strategy.cash_values
    account_values = strategy.account_values
    position_sizes = strategy.position_sizes

    # Extract and convert naive buy and sell datetime objects to timezone-aware Timestamps in UTC
    buys_x = [pd.Timestamp(date).tz_localize('UTC') for date in strategy.buy_dates]
    sells_x = [pd.Timestamp(date).tz_localize('UTC') for date in strategy.sell_dates]

    buys_y = [closes[dates.index(date)] if date in dates else None for date in buys_x]
    sells_y = [closes[dates.index(date)] if date in dates else None for date in sells_x]


    # Check which buy/sell dates were not found in the dates list
    missing_buys = [date for date, price in zip(buys_x, buys_y) if price is None]
    missing_sells = [date for date, price in zip(sells_x, sells_y) if price is None]

    print("Missing buy dates after conversion:", missing_buys)
    print("Missing sell dates after conversion:", missing_sells)

    return dates, closes, sma_short, sma_long, rsi, ema_short, ema_long, volatility, roc, atr, cash_values, account_values, position_sizes, buys_x, buys_y, sells_x, sells_y

class CustomData(bt.feeds.PandasData):
    lines = ('trade_count', 'vwap',)
    params = (
        ('trade_count', -1),
        ('vwap', -1),
    )