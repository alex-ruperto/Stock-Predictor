from dash import html
from UI.navbar import create_navbar

header = html.H3('Welcome to home page!')
nav = create_navbar()

def create_page_home():
    
    layout = html.Div([
        nav,
        header,
    ])
    return layout