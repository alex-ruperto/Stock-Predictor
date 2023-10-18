import dash
from dash import html, dcc, callback
from dash.dependencies import Input, Output, State
from ticker_db import get_all_tickers, get_ticker_data
from Pages.UI.figures import generate_figure_for_ticker
import ticker_db
import dash_bootstrap_components as dbc

dash.register_page(
    __name__, 
    title="Page 3",
    path='/page-3',
    description="This is page 3"
)

# Assuming TICKERS is a list of ticker symbols
TICKERS = get_all_tickers()

# Dropdown component for ticker selection
ticker_dropdown = dcc.Dropdown(
    id='ticker-dropdown',
    options=[{'label': ticker, 'value': ticker} for ticker in TICKERS],
    value=TICKERS[0] if TICKERS else None  # Default to the first ticker or none
)

# Layout for page_3
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
    ])
])

@callback(
    Output('ticker-graph', 'figure'),
    [Input('ticker-dropdown', 'value')]
)
def update_graph(selected_ticker):
    ticker_data = get_ticker_data(selected_ticker)
    return ticker_data['figure'] if ticker_data else {}