import dash
from dash import html, dcc, callback
from dash.dependencies import Input, Output, State
from Utils.ticker_db import get_all_tickers, get_ticker_data
import Utils.ticker_db as ticker_db
import dash_bootstrap_components as dbc
from Utils.logger_config import configure_logger

logger = configure_logger('Backtest Page')
dash.register_page(
    __name__, 
    title="Backtest",
    path='/backtest',
    description="This is the backtest data."
)

# TICKERS is a list of ticker symbols
TICKERS = get_all_tickers()

# Dropdown component for ticker selection
ticker_dropdown = dcc.Dropdown(
    id='ticker-dropdown',
    options=[{'label': ticker, 'value': ticker} for ticker in TICKERS],
    value=TICKERS[0] if TICKERS else None  # Default to the first ticker or none
)

# Layout for backtest_page
layout = dbc.Container([
    dbc.Row([
        dcc.Dropdown(
            id='ticker-dropdown',
            options=[{'label': ticker, 'value': ticker} for ticker in ticker_db.get_all_tickers()],
            value=ticker_db.get_all_tickers()[0] if ticker_db.get_all_tickers() else None, # Default value.
            clearable=False
        ),
        
    ], className='mt-5'), #bootstrap's margin-top
    dbc.Row([
        dcc.Graph(id='ticker-graph')
    ], className='mt-5'),  # Ensure there's space between the dropdown and graph
], fluid=True)  # Ensuring the container is fluid for full-width usage

# Callback to update dropdown options. In case something was added or removed.
@callback(
    Output('ticker-dropdown', 'options'),
    Output('ticker-dropdown', 'value'),
    Input('ticker-dropdown', 'value')
)
def update_dropdown_options(current_value): 
    tickers = get_all_tickers()
    options = [{'label': ticker, 'value': ticker} for ticker in tickers]
    # If the current value is not in the updated ticker list, reset the dropdown value
    value = current_value if current_value in tickers else (tickers[0] if tickers else None)
    return options, value

# Callback to change the selected graph
@callback(
    Output('ticker-graph', 'figure'),
    [Input('ticker-dropdown', 'value')]
)
def update_graph(selected_ticker):
    if selected_ticker:
        ticker_data = get_ticker_data(selected_ticker)
        logger.info(f"Ticker data was retrieved for {selected_ticker}: {ticker_data}")
        return ticker_data['figure'] if ticker_data else {}
    return {}