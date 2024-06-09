import dash
from dash import html, dcc, callback
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from Utils.ticker_db import add_ticker, remove_ticker, get_all_tickers
import Utils.ticker_db as ticker_db
from io import StringIO
import logging

# setup logging and StringIO
log_stream = StringIO()
logging.basicConfig(
    level=logging.INFO, # all messages (Info, Warning, Error, and Critical) will be logged
    format='Time: %(asctime)s  \nLevel: %(levelname)s \n%(message)s \n',
    # %(asctime)s = current time when the log message is generated. e.g., ‘2003-07-08 16:49:45,896’ 
    # (the numbers after the comma are millisecond portion of the time).
    # %(levelname)s = the text logging level, whether it's DEBUG, INFO, WARNING, ERROR, or CRITICAL
    # %(message)s = the logged message
    handlers=[logging.StreamHandler(log_stream)] # write logging records to a stream
)

logger = logging.getLogger()

dash.register_page(
    __name__, 
    title="Tickers",
    path='/tickers',
    description="This is the page to handle tickers"
)

layout = html.Div([
    html.H3("Manage Tickers", className="text-center mb-4"),
    dbc.Row(
        dbc.Col(
            dbc.Input(id='ticker-input', type='', placeholder='Enter ticker symbol...', className="mb-2"),
            width=6,
        ),
        justify="center"
    ),

    dbc.Row(
        [
            dbc.Col(dbc.Button("Add Ticker", id='add-ticker-button', className="custom-button-add mr-2"), width="auto"),
            dbc.Col(dbc.Button("Remove Ticker", id='remove-ticker-button', className="custom-button-remove mr-2"), width="auto"),
            dbc.Col(dbc.Button("Display All Tickers", id='display-tickers-button', className="custom-button-info mr-2"), width="auto"),
            dbc.Col(dbc.Button("Clear Log", id='clear-log-button', className="custom-button-info mr-2"), width="auto")
        ],
        justify="center", # center the buttons
        className="mb-4" # margin bottom for spacing from content
    ),

    dbc.Row(
        dbc.Col(
            dcc.Textarea(id='log-display', style={'width': '100%', 'height': 300}),
            width=12
        )
    ),

    dcc.Interval(
        id='interval-component',
        interval = 1 * 1000, # interval in milliseconds (1000 milliseconds = 1 second)
        n_intervals = 0
    ),

])

# Register the callbacks for ticker_page
@callback(
    Output('log-display', 'value'),
    [
        Input('interval-component', 'n_intervals'),
        Input('add-ticker-button', 'n_clicks'),
        Input('remove-ticker-button', 'n_clicks'),
        Input('display-tickers-button', 'n_clicks'),
        Input('clear-log-button', 'n_clicks'),
    ],
    State('ticker-input', 'value'),
    State('log-display', 'value')
)
def update_db(n_intervals, add_click, remove_click, display_click, clear_click, ticker_value, current_log):
    ctx = dash.callback_context
    if not current_log:
        current_log = ""
    
    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'add-ticker-button' and ticker_value:
            ticker_db.add_ticker(ticker_value)
            logger.info(f"Added {ticker_value}")

        elif button_id == 'remove-ticker-button' and ticker_value:
            ticker_db.remove_ticker(ticker_value)
            logger.info(f"Removed {ticker_value}")

        elif button_id == 'display-tickers-button':
            all_tickers = ticker_db.get_all_tickers()
            for ticker in all_tickers:
                logger.info(f"Ticker: {ticker}")

        elif button_id == 'clear-log-button':
            log_stream.truncate(0) # clear the log stream
            log_stream.seek(0) # reset stream position
            return "" # clear the text area

    # get current log messages
    log_stream.seek(0)
    log_contents = log_stream.read()
    log_stream.truncate(0) # clear the log stream after reading it
    log_stream.seek(0) # reset stream position

    updated_log = current_log + log_contents

    return updated_log