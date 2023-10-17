# imports 
import dash_bootstrap_components as dbc
import dash
from dash import Input, Output, State, html, Dash, html, dcc


app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home Page", href="/")),
        dbc.NavItem(dbc.NavLink("Page 2", href="/page-2")),
        dbc.NavItem(dbc.NavLink("Page 3", href="/page-3")),
    ],
    brand="Trading Bot",
    color="primary",
    dark=True,
)

app.layout = dbc.Container(
    [
        navbar,
        dash.page_container
    ],
    className="dbc",
    fluid=True
)

if __name__ == '__main__':
    app.run(debug=True)