# imports
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from app import app
import dash_bootstrap_components as dbc
from UI.home_page import create_page_home
from UI.page_2 import create_page_2
from UI.page_3 import create_page_3

TICKERS = ['NVDA', 'AAPL', 'GOOG']
server = app.server
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])


def display_page(pathname):
    if pathname == '/2':
        return create_page_2()
    elif pathname == '/3':
        return create_page_3()
    elif pathname == '/':
        return create_page_home()
    else:
        return create_page_home()  # default to home page for any other path
    
if __name__ == '__main__':
    app.run_server(debug=True)