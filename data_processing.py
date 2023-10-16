import yfinance as yf
import backtrader as bt
from ml_models import train_model
from ml_models import preprocess_data
from Strategies.MLStrategy import MLStrategy

def backtest(ticker): # function to backtest and plot individual ticker based on strategy
     # Fetch historical data
    cerebro = bt.Cerebro()
    print(f'Downloading data for: {ticker}.')
    raw_data = yf.download(ticker, '2019-01-01', '2023-09-01', auto_adjust=True)
    # 52 * 3 + 34 = 190 weeks. # 190c = total cash amount invested over 190 weeks where c is the amount invested per week.

    df = preprocess_data(raw_data) # df stands for dataframe
    clf = train_model(df) # train ML model based on df
    clf.eval() # set to evaluation mode.
    data = bt.feeds.PandasData(dataname=df)

    cerebro.adddata(data)  # add datafeed to cerebro
    cerebro.addstrategy(MLStrategy, model=clf)  # use SMACrossover strategy for the backtest.
    # Set our desired cash start
    cerebro.broker.set_cash(100.0)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    print(f'Running backtest on: {ticker}.')

    # Set the commision
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission on trades

    # run cerebro and store it into strategy.
    strategy = cerebro.run()[0]
    # Print out the final result
    print('Ending Portfolio Value: %.2f\n' % cerebro.broker.getvalue())
    print(f"Total Predictions: {len(strategy.predictions)}")
    total_correct_predictions = 0
    print(strategy.actual_movements[:30])
    print(strategy.predictions[:30])
    for prediction, actual in zip(strategy.predictions, strategy.actual_movements):
        if prediction == actual:
            total_correct_predictions += 1

    print(f"Total Correct Predictions: {total_correct_predictions}")
    accuracy = total_correct_predictions / len(strategy.predictions)
    print(f"Total Accuracy: {accuracy * 100:.2f}%")
    
    # extract backtrader data
    dates = df.index.tolist()
    closes = strategy.data.close.array
    sma_short = strategy.sma_short.array
    sma_long = strategy.sma_long.array
    rsi = strategy.rsi.array
    macd = strategy.macd.array
    cash_values = strategy.cash_values
    account_values = strategy.account_values
    position_sizes = strategy.position_sizes

    # For buy/sell markers
    buys_x = strategy.buy_dates # set buys_x to the buy dates. 
    # for each date in buy_dates, find the associated closing price from the 'closes' list and add it to buy_y
    buys_y = [closes[dates.index(date)] for date in buys_x if date in dates]
    sells_x = strategy.sell_dates
    # for each date in sell_dates, find the associated closing price from the 'closes' list and add it to sell_y
    sells_y = [closes[dates.index(date)] for date in sells_x if date in dates]

    return dates, closes, sma_short, sma_long, rsi, macd, cash_values, account_values, position_sizes, buys_x, buys_y, sells_x, sells_y