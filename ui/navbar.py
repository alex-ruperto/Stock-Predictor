from dash import html

def layout():
    return html.Div([
        html.A('Home', href='/home', className='navbar-link'),
    ], className='navbar')