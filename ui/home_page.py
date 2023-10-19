from dash import html, dcc
def layout():
    return html.Div([
        html.H1('Welcome to the Trading Bot dashboard'),
        html.P('Input a ticker to perform a backtest on using a LSTM Machine Learning Model:')
    ])