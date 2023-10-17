import dash
from dash import html

dash.register_page(
    __name__, 
    title="Home Page",
    path='/',
    description="This is the home page"
)

layout = html.Div(
    children="Home Page",
    style={
        "display": "flex",
        "justifyContent": "center",  # Center horizontally
        "alignItems": "center",      # Center vertically
        "height": "100vh"            # Take full viewport height
    }
)