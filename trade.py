# imports
import yfinance as yf
import backtrader as bt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


class SMACrossover(bt.Strategy):  # class definition with bt.Strategy as the parent
    # these two periods will refer to how many days the moving average will be calculated.
    # one moving average for a short period and one for a long period.
    params = dict(
        short_period=50,
        long_period=200
    )

    def __init__(self):  # constructor for when the SMACrossover class is called.
        # create short simple moving average using short_period and long simple moving average using long_period
        # parameters. The crossover indicator is checking for crossovers between two data lines.
        # + 1 for upward, -1 for downward crossover
        self.position_sizes = [self.position.size] * self.p.long_period
        self.cash_value = [
                              self.broker.get_cash()] * self.p.long_period  # multiply this cash value list by the long period
        self.account_values = [
                                  self.broker.get_value()] * self.p.long_period  # multiply this asset value list by the long period
        self.sma_short = bt.ind.SMA(
            period=self.p.short_period)  # initialize sma_short as a bt SMA indicator using the short period
        self.sma_long = bt.ind.SMA(
            period=self.p.long_period)  # initialize sma_long as a bt SMA indicator using the long period
        self.crossover = bt.ind.CrossOver(self.sma_short,
                                          self.sma_long)  # use crossover indicator with sma_short and sma_long

    # call next method for each bar/candle in backtest.
    def next(self):
        self.cash_value.append(self.broker.get_cash())  # append the current cash value to the list.
        self.account_values.append(self.broker.get_value())
        size_to_buy = (self.broker.get_cash() * 0.1) // self.data.close[0]  # take 10% of your cash and then perform floor division by the closing price of the stock.
        if size_to_buy < 1:  # if the size_to_buy value is less than one
            size_to_buy = 1  # buy one

        # Add a 3-day window to confirm trend
        confirmation_days = 3

        if self.crossover > 0 and sum([self.crossover[i] for i in range(-confirmation_days, 0)]) == confirmation_days:  # Golden Cross. Short SMA Crosses over Long SMA
            # this is a check to see short SMA crossed over.
            self.buy(size=size_to_buy)
        elif self.crossover < 0 < self.position.size:  # Death Cross. Short SMA Crosses under Long SMA
            self.sell()

        self.position_sizes.append(self.position.size)


def main():
    cerebro = bt.Cerebro()
    # create a "Cerebro engine." Used for running backtest, managing data feeds, strategies,
    # broker simulation, etc/

    # Fetch historical data
    data = yf.download('MRNA', start='2019-01-01', end='2023-09-01')  # collect data from MRNA at given time range
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

    # For buy/sell markers
    buys_x = [dates[i] for i, val in enumerate(strategy.crossover.array) if
              val > 0]  # only considers values that are positive. for each one, store corresponding date from the dates list using i. (buy signal)
    buys_y = [closes[i] for i, val in enumerate(strategy.crossover.array) if
              val > 0]  # only considers values that are positive. for each one, store corresponding closing price from the dates list using i. (buy signal)
    sells_x = [dates[i] for i, val in enumerate(strategy.crossover.array) if
               val < 0]  # only considers values that are negative. for each one, store corresponding date from the dates list using i. (sell signal)
    sells_y = [closes[i] for i, val in enumerate(strategy.crossover.array) if
               val < 0]  # only considers values that are negative. for each one, store corresponding closing price from the dates list using i. (sell signal)

    # create plotly plot
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=('Trading Data', 'Portfolio Value', 'Position Over Time'))

    # First plot (Trading Data)
    # Side note: The legend tag is simply to help with te alignment in fig.update_layout.
    fig.add_trace(go.Scatter(x=dates, y=np.array(closes).tolist(), mode='lines', name='Close Price', legend='legend1'),
                  row=1, col=1)  # Plot close price on row 1 col 1
    fig.add_trace(
        go.Scatter(x=dates, y=np.array(sma_short).tolist(), mode='lines', name='50-day SMA', legend='legend1'), row=1,
        col=1)  # Plot 20-day SMA on row 1 col 1
    fig.add_trace(
        go.Scatter(x=dates, y=np.array(sma_long).tolist(), mode='lines', name='200-day SMA', legend='legend1'), row=1,
        col=1)  # Plot 100-day SMA on row 1 col 1
    fig.add_trace(go.Scatter(x=buys_x, y=buys_y, mode='markers', marker=dict(color='green', size=15), name='Buy Signal',
                             legend='legend1'), row=1, col=1)  # Plot the buys on row 1 col 1
    fig.add_trace(
        go.Scatter(x=sells_x, y=sells_y, mode='markers', marker=dict(color='red', size=15), name='Sell Signal',
                   legend='legend1'), row=1, col=1)  # Plot the sells on row 1 col 1

    # Second Plot (Portfolio Value)
    fig.add_trace(
        go.Scatter(x=dates, y=np.array(cash_value).tolist(), mode='lines', name='Cash Over Time', legend='legend2'),
        row=2, col=1)  # plot the cash value on row 2 col 1
    fig.add_trace(go.Scatter(x=dates, y=np.array(account_values).tolist(), mode='lines', name='Account Value Over Time',
                             legend='legend2'), row=2, col=1)  # plot the account values on row 2 col 1

    # Third Plot (Position over Time)
    fig.add_trace(go.Scatter(x=dates, y=np.array(position_sizes).tolist(), mode='lines', name='Position Over Time',
                             legend='legend3'), row=3, col=1)  # plot the position over time on row 3 col 1.

    # Update xaxis properties
    fig.update_layout(
        xaxis=dict(rangeselector=dict(buttons=list([dict(count=1, label="1m", step="month", stepmode="backward", ),
                                                    dict(count=6, label="6m", step="month", stepmode="backward"),
                                                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                                                    dict(count=1, label="1y", step="year", stepmode="backward"),
                                                    dict(step="all")])),
                   showline=True,  # Shows the x-axis line
                   showgrid=True,  # Shows the x-axis grid
                   showticklabels=True),  # Shows the x-axis tick labels
        xaxis2=dict(showline=True, showgrid=True, showticklabels=True),
        xaxis3=dict(showline=True, showgrid=True, showticklabels=True),
        height=2500,
        template="plotly_dark",
        legend1={"y": 1},
        legend2={"y": 0.62},
        legend3={"y": 0.25},
        xaxis_rangeselector_font_color='white',
        xaxis_rangeselector_activecolor='red',
        xaxis_rangeselector_bgcolor='black',
    )

    fig.show()
    # fig.write_html("MRNA Stock.html") # this line allows user to download it as a html file


if __name__ == '__main__':
    main()
