import backtrader as bt
import alpaca_trade_api as tradeapi
from alpaca_trade_api import TimeFrame
import alpacaconfig as config
from random_forest_model import train_random_forest_model
from Strategies.MLStrategy import MLStrategy

alpaca_api = tradeapi.REST(config.ALPACA_KEY, config.ALPACA_SECRET_KEY, base_url=config.APCA_API_BASE_URL)

def backtest(ticker): # backtest function for an individual stock
    # get data from alpaca
    stock_data = alpaca_api.get_bars(ticker, TimeFrame.Hour, start="2020-01-01", end="2023-01-01").df
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
    closes = stock_data['close'].tolist()
    sma_short = [strategy.sma_short.lines.sma[index] for index in range(len(dates))]
    sma_long = [strategy.sma_long.lines.sma[index] for index in range(len(dates))]
    rsi = [strategy.rsi.lines.rsi[index] for index in range(len(dates))]
    macd = [strategy.macd.lines.macd[index] for index in range(len(dates))]

    # Extract account and trading values
    cash_values = strategy.cash_values
    account_values = strategy.account_values
    position_sizes = strategy.position_sizes

    # Extract buy and sell dates
    buys_x = [bt.num2date(dt) for dt in strategy.buy_dates]
    buys_y = [closes[dates.index(date)] for date in buys_x if date in dates]
    sells_x = [bt.num2date(dt) for dt in strategy.sell_dates]
    sells_y = [closes[dates.index(date)] for date in sells_x if date in dates]

    return dates, closes, sma_short, sma_long, rsi, macd, cash_values, account_values, position_sizes, buys_x, buys_y, sells_x, sells_y

class CustomData(bt.feeds.PandasData):
    lines = ('trade_count', 'vwap',)
    params = (
        ('trade_count', -1),
        ('vwap', -1),
    )