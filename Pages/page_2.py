import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from ticker_db import add_ticker, remove_ticker, get_all_tickers


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
