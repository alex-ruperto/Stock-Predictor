# imports
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from UI.figures import generate_figures_for_tickers
from UI.navbar import layout as navbar_layout
from UI.home_page import layout as home_page_layout


TICKERS = ['NVDA', 'AAPL', 'GOOG']
external_stylesheets = ['/assets/styles.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout=html.Div([
    dcc.Location(id='url', refresh=False),
    navbar_layout(),
    html.Div(id='page-content')
])

# update the graph based on the dropdown on selection
@app.callback(
        Output('page-content', 'children'),
        [Input('url', 'pathname')]
)

def update_page(pathname):
    if pathname == '/' or pathname == '/home':  # Default or home page
        return home_page_layout()
    else:
        return '404 Page Not Found'  # Optionally, a 404 page

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)