import dash
from dash import html, dcc, callback
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from ticker_db import add_ticker, remove_ticker, get_all_tickers
import ticker_db


dash.register_page(
    __name__, 
    title="Page 2",
    path='/page-2',
    description="This is page 2"
)

layout = html.Div([
    html.H3("Manage Tickers"),
    
    # Input for ticker symbol
    dbc.Input(id='ticker-input', type='text', placeholder='Enter ticker symbol...'),
    
    # Buttons
    dbc.Button("Add Ticker", id='add-ticker-button', color="success", className="mr-2"),
    dbc.Button("Remove Ticker", id='remove-ticker-button', color="danger", className="mr-2"),
    dbc.Button("Display All Tickers", id='display-tickers-button', className="mr-2"),
    
    # Placeholder for displaying list of tickers
    html.Div(id='tickers-display', className="mt-4")
])

# Register the callbacks for page_2
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
        return ", ".join(all_tickers)

    return ""