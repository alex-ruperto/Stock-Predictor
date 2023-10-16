# imports
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from ui.figures import generate_figures_for_tickers

TICKERS = ['NVDA', 'AAPL', 'GOOG']
app = dash.Dash(__name__)

app.layout=html.Div([
    dcc.Dropdown(
        id='ticker-dropdown',
        options=[{'label': ticker, 'value': ticker} for ticker in TICKERS],
        value=TICKERS[0] # default
    ),
    dcc.Graph(id='stock-graph')
])

# update the graph based on the dropdown on selection
@app.callback(
        Output('stock-graph', 'figure'), # update the figure property of the component ID stock-graph
        [Input('ticker-dropdown', 'value')] # input to this function is the value property of ticker-dropdown
)

def update_graph(selected_ticker):
    return figures[selected_ticker]

if __name__ == '__main__':
    figures = generate_figures_for_tickers(TICKERS)
    app.run_server(debug=True, use_reloader=False)