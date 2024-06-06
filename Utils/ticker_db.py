
import shelve
import yfinance as yf
from data_processing import backtest
import plotly.graph_objects as go
from Pages.UI.figures import generate_figure_for_ticker

# "Database" for holding ticker data
DB_NAME = "ticker_db"

# Add a single ticker's figure and data
def add_ticker(ticker):
    figure, data = generate_figure_and_data_for_ticker(ticker)
    with shelve.open(DB_NAME) as db:
        db[ticker] = {'figure': figure, 'data': data}

# Add multiple tickers
def add_multiple_tickers(ticker_list):
    for ticker in ticker_list:
        add_ticker(ticker)

# Remove a single ticker
def remove_ticker(ticker):
    with shelve.open(DB_NAME) as db:
        if ticker in db:
            del db[ticker]

# Remove multiple tickers
def remove_multiple_tickers(ticker_list):
    for ticker in ticker_list:
        remove_ticker(ticker)

# Retrieve data for a specific ticker
def get_ticker_data(ticker):
    with shelve.open(DB_NAME) as db:
        return db.get(ticker)

# Get a list of all tickers in the "database"
def get_all_tickers():
    with shelve.open(DB_NAME) as db:
        return list(db.keys())

# Logic to generate figure and data for a single ticker
def generate_figure_and_data_for_ticker(ticker):
    # call backtest function to pre-process then backtest on ticker. this collects "data" of ticker.
    data = backtest(ticker) 
    figure = generate_figure_for_ticker(ticker, *data)
    return figure, data
