import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from data_processing import backtest

def generate_figures_for_tickers(tickers):
    figures = {}
    for ticker in tickers:
        data = backtest(ticker)
        fig = generate_figure_for_ticker(ticker, *data) # pass in ticker along with all the returned 'data' from the backtest.
        figures[ticker] = fig
    return figures

def generate_figure_for_ticker(ticker, dates, closes, sma_short, sma_long, rsi, macd, cash_values, account_values, position_sizes, buys_x, buys_y, sells_x, sells_y):

    # create plotly plot
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=(f'Trading Data for {ticker}','RSI', 'MACD', 'Portfolio Value', 'Position Over Time'))

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
    # Second Plot (RSI)
    fig.add_trace(go.Scatter(x=dates, y=np.array(closes).tolist(), mode='lines', name='Close Price', legend='legend2'),
                  row=2, col=1)  # Plot close price on row 2 col 1
    fig.add_trace(go.Scatter(x=dates, y=np.array(rsi).tolist(), mode='lines', name='RSI', legend='legend2'),
                  row=2, col=1)  # Plot close price on row 2 col 1
    fig.add_trace(go.Scatter(x=buys_x, y=buys_y, mode='markers', marker=dict(color='green', size=15), name='Buy Order',
                             legend='legend2'), row=2, col=1)  # Plot the buys on row 2 col 1
    fig.add_trace(
        go.Scatter(x=sells_x, y=sells_y, mode='markers', marker=dict(color='red', size=15), name='Sell Order',
                   legend='legend2'), row=2, col=1)  # Plot the sells on row 2 col 1
    
    # Third Plot (MACD)
    fig.add_trace(go.Scatter(x=dates, y=np.array(closes).tolist(), mode='lines', name='Close Price', legend='legend3'),
                  row=3, col=1)  # Plot close price on row 3 col 1
    fig.add_trace(go.Scatter(x=dates, y=np.array(macd).tolist(), mode='lines', name='MACD', legend='legend3'),
                  row=3, col=1)  # Plot close price on row 3 col 1
    fig.add_trace(go.Scatter(x=buys_x, y=buys_y, mode='markers', marker=dict(color='green', size=15), name='Buy Order',
                             legend='legend3'), row=3, col=1)  # Plot the buys on row 3 col 1
    fig.add_trace(
        go.Scatter(x=sells_x, y=sells_y, mode='markers', marker=dict(color='red', size=15), name='Sell Order',
                   legend='legend3'), row=3, col=1)  # Plot the sells on row 2 col 1

    # Fourth Plot (Portfolio Value)
    fig.add_trace(
        go.Scatter(x=dates, y=np.array(cash_values).tolist(), mode='lines', name='Cash Over Time', legend='legend4'),
        row=4, col=1)  # plot the cash value on row 4 col 1
    fig.add_trace(go.Scatter(x=dates, y=np.array(account_values).tolist(), mode='lines', name='Account Value Over Time',
                             legend='legend4'), row=4, col=1)  # plot the account values on row 4 col 1

    # Fifth Plot (Position over Time)
    fig.add_trace(go.Scatter(x=dates, y=np.array(position_sizes).tolist(), mode='lines', name='Position Over Time',
                             legend='legend5'), row=5, col=1)  # plot the position over time on row 5 col 1.

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
        legend2={"y": 0.77},
        legend3={"y": 0.56},
        legend4={"y": 0.33},
        legend5={"y": 0.10},
        xaxis_rangeselector_font_color='white',
        xaxis_rangeselector_activecolor='red',
        xaxis_rangeselector_bgcolor='black',
    )
    return fig