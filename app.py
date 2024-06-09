# imports 
from Utils.app_instance import app
import dash_bootstrap_components as dbc
import dash 

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home Page", href="/")),
        dbc.NavItem(dbc.NavLink("Tickers", href="/tickers")),
        dbc.NavItem(dbc.NavLink("Backtest", href="/backtest")),
    ],
    brand="Trading Bot",
    color="primary",
    dark=True,
)

app.layout = dbc.Container(
    [
        dbc.Row([navbar]),
        dbc.Row([dash.page_container], className="mt-5")
    ],
    className="dbc",
    fluid=True
)


if __name__ == '__main__':
    app.run(debug=True)