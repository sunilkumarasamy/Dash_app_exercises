import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Sample DataFrame
mp = pd.DataFrame({'value': np.random.randn(100), 'center_dist': np.random.randn(100)})

app = dash.Dash(__name__)

def remove_outliers_by_kurtosis(x, threshold=3):
    kurt = x.kurtosis()
    mask = (x - x.mean()).abs() < threshold * x.std()
    x_no_outliers = x[mask]
    return x_no_outliers

app.layout = html.Div([
    dcc.Graph(id='kde-plot'),
    dcc.Graph(id='qq-plot'),
    dcc.Graph(id='scatter-plot'),
    html.Div(id='descriptive-stats'),
    html.Div(id='regression-stats')
])

@app.callback(
    Output('kde-plot', 'figure'),
    Output('qq-plot', 'figure'),
    Output('scatter-plot', 'figure'),
    Output('descriptive-stats', 'children'),
    Output('regression-stats', 'children'),
    Input('kde-plot', 'hoverData')
)
def update_plots(hoverData):
    # Assuming mp is your DataFrame
    x = mp
    outlier_threshold = 3

    # Plot KDE
    kde_fig = px.histogram(x, x='value', nbins=30, marginal='kde', title='Kernel Density Estimate (KDE)')

    # QQ plot
    qq_fig = px.scatter(x, x='value', title='Quantile-Quantile (QQ) Plot')
    qq_fig.update_traces(marker=dict(color='red'))

    # Remove outliers based on kurtosis
    x_no_outliers = remove_outliers_by_kurtosis(x['value'], outlier_threshold)
    print(x_no_outliers.shape)

    # Scatter plot with linear regression line (after removing outliers)
    scatter_fig = px.scatter(x, x='center_dist', y=x_no_outliers, title='Scatter Plot with Linear Regression Line (Outliers Removed by Kurtosis)')
    model = LinearRegression().fit(x['center_dist'].values.reshape(-1, 1), x_no_outliers.values.reshape(-1, 1))
    scatter_fig.add_trace(go.Scatter(x=x['center_dist'], y=model.predict(np.array(x['center_dist']).reshape(-1, 1)).squeeze(),
                                     mode='lines', line=dict(color='red')))

    # Descriptive statistics
    desc_stats = html.Div(x_no_outliers.describe())

    # Linear regression coefficients and R-squared
    regression_stats = html.Div([
        html.Div(f"Linear Regression Coefficients: {model.coef_}"),
        html.Div(f"R-squared: {r2_score(x_no_outliers, model.predict(x['center_dist'].values.reshape(-1, 1)).squeeze())}")
    ])

    return kde_fig, qq_fig, scatter_fig, desc_stats, regression_stats

if __name__ == '__main__':
    app.run_server(port=8046,debug=True)
