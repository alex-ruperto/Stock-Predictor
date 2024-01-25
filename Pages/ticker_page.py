import dash
from dash import html, dcc, callback
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from ticker_db import add_ticker, remove_ticker, get_all_tickers
import ticker_db


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
            dbc.Col(dbc.Button("Display All Tickers", id='display-tickers-button', className="custom-button-info mr-2"), width="auto")
        ],
        justify="center", # center the buttons
        className="mb-4" # margin bottom for spacing from content
    ),

    dbc.Row(
        dbc.Col(
            dbc.ListGroup(id='tickers-display', className="mt-4 ticker-list-group"),
            width=12
        ),
        justify="center"
    )
])

# Register the callbacks for ticker_page
@callback(
    Output('tickers-display', 'children'),
    [
        Input('add-ticker-button', 'n_clicks'),
        Input('remove-ticker-button', 'n_clicks'),
        Input('display-tickers-button', 'n_clicks')
    ],
    State('ticker-input', 'value')
)
def update_db(add_click, remove_click, display_click, ticker_value):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'add-ticker-button':
        ticker_db.add_ticker(ticker_value)
        return f"Added {ticker_value}"

    elif button_id == 'remove-ticker-button':
        ticker_db.remove_ticker(ticker_value)
        return f"Removed {ticker_value}"

    elif button_id == 'display-tickers-button':
        all_tickers = ticker_db.get_all_tickers()
        return [dbc.ListGroupItem(ticker) for ticker in all_tickers]

    return ""