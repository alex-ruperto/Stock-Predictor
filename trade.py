# imports
import yfinance as yf
import backtrader as bt


class SMACrossover(bt.Strategy):  # class definition with bt.Strategy as the parent
    # these two periods will refer to how many days the moving average will be calculated.
    # one moving average for a short period and one for a long period.
    params = (('short_period', 20), ('long_period', 100))

    def __init__(self):  # constructor for when the SMACrossover class is called.
        # create short simple moving average using short_period and long simple moving average using long_period
        # parameters. The crossover indicator is checking for crossovers between two data lines.
        # + 1 for upward, -1 for downward crossover
        self.sma_short = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.short_period)
        self.sma_long = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.long_period)
        self.crossover = bt.indicators.CrossOver(self.sma_short, self.sma_long)

    # call next method for each bar/candle in backtest.
    def next(self):
        if self.crossover > 0:  # if short crosses above long. Bullish signal known as "Golden Cross"
            if not self.position:  # if not in the market
                self.buy()  # buy the stock.
        elif self.crossover < 0:  # if short crosses below long. Bearish signal known as "Death Cross"
            if self.position:  # if in the market
                self.sell()  # sell the stock.


if __name__ == '__main__':
    cerebro = bt.Cerebro()
    # create a "Cerebro engine." Used for running backtest, managing data feeds, strategies,
    # broker simulation, etc.

    # Fetch historical data
    data = yf.download('MRNA', start='2020-01-01', end='2022-01-01')  # collect data from MRNA stock: 2020-01-01 to
    # 2022-01-01
    datafeed = bt.feeds.PandasData(dataname=data)  # convert data into format that cerebro can understand
    cerebro.adddata(datafeed)  # add datafeed to cerebro

    cerebro.addstrategy(SMACrossover)  # use SMACrossover strategy for the backtest.

    # Set our desired cash start
    cerebro.broker.set_cash(1000.0)

    # Set the commission
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission on trades

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Plot the result
    cerebro.plot()
