# imports
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from UI.home_page import create_page_home
from UI.page_2 import create_page_2
from UI.page_3 import create_page_3




TICKERS = ['NVDA', 'AAPL', 'GOOG']
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.LUX])



app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])


def display_page(pathname):
    if pathname == '/page-2':
        return create_page_2()
    if pathname == '/page-3':
        return create_page_3()
    else:
        return create_page_home()

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)