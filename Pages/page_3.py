import dash
from dash import html, dcc
from Pages.UI.figures import generate_figures_for_tickers
from dash.dependencies import Input, Output, State

dash.register_page(
    __name__, 
    title="Page 3",
    path='/page-3',
    description="This is page 3"
)

# Assuming TICKERS is a list of ticker symbols
TICKERS = ["AAPL", "MSFT", "GOOGL"]  # Example tickers, replace with your list
figures = generate_figures_for_tickers(TICKERS)

# Dropdown component for ticker selection
ticker_dropdown = dcc.Dropdown(
    id='ticker-dropdown',
    options=[{'label': ticker, 'value': ticker} for ticker in TICKERS],
    value=TICKERS[0]  # Default to the first ticker
)

# Layout for page_3
layout = html.Div([
    html.H3('Stock Analysis Page'),
    ticker_dropdown,
    dcc.Graph(id='ticker-graph', figure=figures[TICKERS[0]])  # Default figure
])