# imports
import yfinance as yf
import backtrader as bt
import plotly.graph_objects as go
import numpy as np


class SMACrossover(bt.Strategy):  # class definition with bt.Strategy as the parent
    # these two periods will refer to how many days the moving average will be calculated.
    # one moving average for a short period and one for a long period.
    params = dict(
        short_period=20,
        long_period=100
    )

    def __init__(self):  # constructor for when the SMACrossover class is called.
        # create short simple moving average using short_period and long simple moving average using long_period
        # parameters. The crossover indicator is checking for crossovers between two data lines.
        # + 1 for upward, -1 for downward crossover
        self.position_sizes = [self.position.size] * self.p.long_period
        self.cash_value = [self.broker.get_cash()] * self.p.long_period  # multiply this cash value list by the long period
        self.account_values = [self.broker.get_value()] * self.p.long_period  # multiply this asset value list by the long period
        self.sma_short = bt.ind.SMA(period=self.p.short_period)  # initialize sma_short as a bt SMA indicator using the short period
        self.sma_long = bt.ind.SMA(period=self.p.long_period)  # initialize sma_long as a bt SMA indicator using the long period
        self.crossover = bt.ind.CrossOver(self.sma_short, self.sma_long)  # use crossover indicator with sma_short and sma_long

    # call next method for each bar/candle in backtest.
    def next(self):
        self.cash_value.append(self.broker.get_cash())  # append the current cash value to the list.
        self.account_values.append(self.broker.get_value())
        size_to_buy = (self.broker.get_cash() * 0.1) // self.data.close[0]  # take 10% of your cash and then perform floor division by the closing price of the stock.
        if size_to_buy < 1:  # if the size_to_buy value is less than one
            size_to_buy = 1  # buy one

        if self.crossover > 0:  # Golden Cross
            self.buy(size=size_to_buy)
        elif self.crossover < 0:  # Death Cross
            self.sell()

        self.position_sizes.append(self.position.size)


def main():
    cerebro = bt.Cerebro()
    # create a "Cerebro engine." Used for running backtest, managing data feeds, strategies,
    # broker simulation, etc/

    # Fetch historical data
    data = yf.download('MRNA', start='2016-01-01', end='2023-01-01')  # collect data from MRNA at given time range
    datafeed = bt.feeds.PandasData(dataname=data)  # convert data into format that cerebro can understand
    cerebro.adddata(datafeed)  # add datafeed to cerebro

    cerebro.addstrategy(SMACrossover)  # use SMACrossover strategy for the backtest.

    # Set our desired cash start
    cerebro.broker.set_cash(1000.0)

    # Set the commission
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission on trades

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # run cerebro and store it into strategy.
    strategy = cerebro.run()[0]

    # Print out the final result
    print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # extract backtrader data
    dates = [bt.num2date(x) for x in strategy.data.datetime.array]
    closes = strategy.data.close.array
    sma_short = strategy.sma_short.array
    sma_long = strategy.sma_long.array
    cash_value = strategy.cash_value
    account_values = strategy.account_values
    position_sizes = strategy.position_sizes

    print(len(dates))
    print(len(cash_value))
    print(len(account_values))
    print(len(position_sizes))

    # For buy/sell markers
    buys_x = [dates[i] for i, val in enumerate(strategy.crossover.array) if val > 0]
    buys_y = [closes[i] for i, val in enumerate(strategy.crossover.array) if val > 0]
    sells_x = [dates[i] for i, val in enumerate(strategy.crossover.array) if val < 0]
    sells_y = [closes[i] for i, val in enumerate(strategy.crossover.array) if val < 0]

    # create plotly plot
    figure_trades = go.Figure()

    # add close prices
    figure_trades.add_trace(go.Scatter(x=dates, y=np.array(closes).tolist(), mode='lines', name='Close Price'))

    # Add SMAs
    figure_trades.add_trace(go.Scatter(x=dates, y=np.array(sma_short).tolist(), mode='lines', name='20-day SMA'))
    figure_trades.add_trace(go.Scatter(x=dates, y=np.array(sma_long).tolist(), mode='lines', name='100-day SMA'))

    # Add buy/sell markers
    figure_trades.add_trace(
        go.Scatter(x=buys_x, y=buys_y, mode='markers', marker=dict(color='green', size=10), name='Buy Signal'))
    figure_trades.add_trace(
        go.Scatter(x=sells_x, y=sells_y, mode='markers', marker=dict(color='red', size=10), name='Sell Signal'))

    # Add cash over time and account value over time
    figure_trades.add_trace(go.Scatter(x=dates, y=np.array(cash_value).tolist(), mode='lines', name='Cash Over Time'))
    figure_trades.add_trace(
        go.Scatter(x=dates, y=np.array(account_values).tolist(), mode='lines', name='Account Value over Time'))

    # Add position size over time
    figure_trades.add_trace(
        go.Scatter(x=dates, y=np.array(position_sizes).tolist(), mode='lines', name='Position over Time'))

    figure_trades.show()


if __name__ == '__main__':
    main()
