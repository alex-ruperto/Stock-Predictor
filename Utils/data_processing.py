# imports
import backtrader as bt
import alpaca_trade_api as tradeapi
import Utils.alpacaconfig as config
import pandas as pd
import logging
from Models.random_forest_model import RandomForestTrainer
from Preprocessors.rfc_preprocessor import RFCPreprocessor
from Strategies.RFCStrategy import RFCStrategy
from alpaca_trade_api import TimeFrame
from Utils.logger_config import configure_logger

logger = configure_logger("Backtest")

alpaca_api = tradeapi.REST(config.ALPACA_KEY, config.ALPACA_SECRET_KEY, base_url=config.APCA_API_BASE_URL)

def backtest(ticker): # backtest function for an individual stock
    logger.info("Collecting stock data...")
    # get data from alpaca
    stock_data = alpaca_api.get_bars(ticker, TimeFrame.Hour, start="2022-01-01", end="2023-01-01").df
    if stock_data.empty:
        raise ValueError("No data fetched from Alpaca.")
    
    # Prepare data for backtrader
    stock_data.columns = [col.lower() for col in stock_data.columns]  # Ensure column names are in lowercase
    
    # Preprocess data with indicators
    preprocessor = RFCPreprocessor()
    preprocessed_data=  preprocessor.preprocess(stock_data)

    data_feed = CustomDataWithIndicators(dataname=preprocessed_data)
    
    # Initialize Cerebro engine
    cerebro = bt.Cerebro()
    cerebro.adddata(data_feed)
    cerebro.broker.set_cash(100)

    # Train model and add strategy to Cerebro
    rfc_trainer = RandomForestTrainer()
    rfc_trainer.train(preprocessed_data, 'target')

    # Evaluate the model performance during training
    try:
        evaluation_metrics = rfc_trainer.evaluate(rfc_trainer.model)
        logger.info(f"Training set evaluation metrics: {evaluation_metrics}")
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
    
    # Run backtest
    try:
        cerebro.addstrategy(RFCStrategy, model=rfc_trainer.model)
        logger.info("Backtesting Random Forest Classifier Model for " + ticker + "...")
        strategies = cerebro.run() # return a list of strategies
        strategy = strategies[0] # extract first strategy instance 
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")

    predictions = strategy.predictions
    actual_movements = strategy.actual_movements
    bt_accuracy = sum([1 if p == a else 0 for p, a in zip(predictions, actual_movements)]) / len(predictions)
    logger.info(f"Backtest Accuracy: {bt_accuracy}")

    # Extract strategy data for analysis
    dates = stock_data.index.tolist()
    closes = strategy.data.close.array
    cash_values = strategy.cash_values
    account_values = strategy.account_values
    position_sizes = strategy.position_sizes

    # Extract and convert buy and sell datetime objects to timezone-aware Timestamps in UTC
    buys_x = [pd.Timestamp(date).tz_localize('UTC') for date in strategy.buy_dates]
    sells_x = [pd.Timestamp(date).tz_localize('UTC') for date in strategy.sell_dates]

    buys_y = [closes[dates.index(date)] if date in dates else None for date in buys_x]
    sells_y = [closes[dates.index(date)] if date in dates else None for date in sells_x]

    # Extract feature importances
    feature_importances = rfc_trainer.model.feature_importances_

    logger.info("Backtest complete.")

    return dates, closes, cash_values, account_values, position_sizes, buys_x, buys_y, sells_x, sells_y, predictions, actual_movements, bt_accuracy, evaluation_metrics, feature_importances, preprocessed_data

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