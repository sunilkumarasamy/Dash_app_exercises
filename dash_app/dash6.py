import dash
import dash_html_components as html
import dash_design_kit as ddk

# Your Dash app instance
app = dash.Dash(__name__)

# Layout using dash-design-kit components
app.layout = html.Div([
    ddk.Card(
        children=[
            ddk.CardHeader("Example Card"),
            ddk.CardBody("This is the body of the card."),
        ]
    )
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
