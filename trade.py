# imports
import yfinance as yf
import backtrader as bt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from SMACrossoverStrategy import SMACrossoverStrategy
from MLStrategy import MLStrategy
from ml_models import train_model
import pandas as pd


def main():
    cerebro = bt.Cerebro()
    # create a "Cerebro engine." Used for running backtest, managing data feeds, strategies,
    # broker simulation, etc/

    # Fetch historical data
    ticker = 'MRNA'
    raw_data = yf.download(ticker, '2019-01-01', '2023-09-01', auto_adjust=True)

    df = preprocess_data(raw_data) # df stands for dataframe
    clf = train_model(df) # train ML model based on df
    data = bt.feeds.PandasData(dataname=df)

    cerebro.adddata(data)  # add datafeed to cerebro
    cerebro.addstrategy(MLStrategy, model=clf)  # use SMACrossover strategy for the backtest.
    # Set our desired cash start
    cerebro.broker.set_cash(10000.0)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Set the commis
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission on trades

    # print("Last date in data feed:", data.index[-1])
    # print("Length of data feed in cerebro:", len(data))
    # Print out the starting conditions
    # print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # run cerebro and store it into strategy.
    strategy = cerebro.run()[0]
    

    # Print out the final result
    print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # extract backtrader data
    dates = [bt.num2date(x) for x in strategy.data.datetime.array]
    closes = strategy.data.close.array
    sma_short = strategy.sma1.array
    sma_long = strategy.sma2.array
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

    # create plotly plot
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=(f'Trading Data for {ticker}', 'Portfolio Value', 'Position Over Time'))

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
    fig.add_trace(go.Scatter(x=buys_x, y=buys_y, mode='markers', marker=dict(color='green', size=15), name='Buy Order',
                             legend='legend1'), row=1, col=1)  # Plot the buys on row 1 col 1
    fig.add_trace(
        go.Scatter(x=sells_x, y=sells_y, mode='markers', marker=dict(color='red', size=15), name='Sell Order',
                   legend='legend1'), row=1, col=1)  # Plot the sells on row 1 col 1

    # Second Plot (Portfolio Value)
    fig.add_trace(
        go.Scatter(x=dates, y=np.array(cash_values).tolist(), mode='lines', name='Cash Over Time', legend='legend2'),
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

def preprocess_data(df):
    df['SMA1'] = df['Close'].rolling(window=50).mean()
    df['SMA2'] = df['Close'].rolling(window=200).mean()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df['RSI'] = rsi_formula(df['Close'], window=14)
    df['MACD_Line'], df['Signal_Line'] = macd_formula(df['Close'], short_window=12, long_window=26, signal_window=9)
    return df

def rsi_formula(data, window):
    """Compute the RSI for a given data series."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def macd_formula(data, short_window, long_window, signal_window):
    """Compute the MACD for a given data series."""
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()

    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    
    return macd_line, signal_line

if __name__ == '__main__':
    main()