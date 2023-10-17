from dash import html
from UI.navbar import create_navbar


header = html.H3('Welcome to page 3!')
nav = create_navbar()


def create_page_3():
    
    layout = html.Div([
        nav,
        header,
    ])
    return layout