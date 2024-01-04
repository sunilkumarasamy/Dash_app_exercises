import dash
from dash import dcc, html
import numpy as np
import pandas as pd

# Create Dash app
app = dash.Dash(__name__)

# Create sample data
np.random.seed(42)
data = np.random.normal(loc=0, scale=1, size=(100, 2))
df = pd.DataFrame(data, columns=['X', 'Y'])
df['R'] = np.sqrt(df['X']**2 + df['Y']**2)

# Convert to polar coordinates
df['Theta'] = np.arctan2(df['Y'], df['X'])

# Calculate mean and standard deviation of radial distances
mean_r = df['R'].mean()
std_r = df['R'].std()

# Set a dynamic threshold (e.g., 2 times the standard deviation)
dynamic_threshold = 2 * std_r

# Identify outliers
outlier_indices = df[df['R'] > dynamic_threshold].index

# Define layout
app.layout = html.Div([
    dcc.Graph(
        id='polar-plot',
        figure={
            'data': [
                {'type': 'scatterpolar', 'r': df['R'], 'theta': df['Theta'], 'mode': 'markers', 'name': 'Data Points'},
                {'type': 'scatterpolar', 'r': df.loc[outlier_indices, 'R'], 'theta': df.loc[outlier_indices, 'Theta'], 'mode': 'markers', 'marker': {'color': 'red'}, 'name': 'Outliers'},
                {'type': 'scatterpolar', 'r': [dynamic_threshold, dynamic_threshold], 'theta': [0, 2*np.pi], 'mode': 'lines', 'line': {'dash': 'dash', 'color': 'green'}, 'name': 'Dynamic Threshold'}
            ],
            'layout': {
                'polar': {'radialaxis': {'visible': True}},
                'title': 'Polar Plot with Outlier Detection'
            }
        }
    )
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True,port=8051)
