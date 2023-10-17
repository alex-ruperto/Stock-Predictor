import dash
from dash import html

dash.register_page(
    __name__, 
    title="Page 2",
    path='/page-2',
    description="This is page 2"
)

layout = html.Div(
    children="Page 2",
    style={
        "display": "flex",
        "justifyContent": "center",  # Center horizontally
        "alignItems": "center",      # Center vertically
        "height": "100vh"            # Take full viewport height
    }
)