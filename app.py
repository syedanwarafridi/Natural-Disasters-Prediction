import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Load the dataset
df = pd.read_csv('./datasets/earthquake data.csv')

# Data cleaning
df['Date & Time'] = pd.to_datetime(df['Date & Time'])

# Handle negative depth values
df['Depth'] = df['Depth'].apply(lambda x: x if x > 0 else 1)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Earthquake Data Dashboard", style={'textAlign': 'center', 'color': '#0074D9', 'fontWeight': 'bold'}),

    html.P("This dashboard presents visualizations of earthquake data including their locations, magnitudes, and depths. You can explore the data through various interactive graphs and maps. Use the dropdown menu to filter the data by country.",
           style={'textAlign': 'center', 'color': '#4B4B4B', 'fontSize': '18px', 'maxWidth': '800px', 'margin': 'auto'}),

    dcc.Graph(
        id='scatter-plot',
        figure=px.scatter(
            df,
            x='Longitude',
            y='Latitude',
            color='Magnitude',
            size='Depth',
            hover_name='Country',
            title='Earthquakes by Location'
        ).update_layout(title={'text': 'Earthquakes by Location', 'x': 0.5}, title_font={'size': 24, 'color': '#0074D9', 'family': 'Arial'}),
        style={'padding': '20px'}
    ),

    dcc.Graph(
        id='time-series',
        figure=px.line(
            df.sort_values('Date & Time'),
            x='Date & Time',
            y='Magnitude',
            title='Earthquake Magnitude Over Time'
        ).update_layout(title={'text': 'Earthquake Magnitude Over Time', 'x': 0.5}, title_font={'size': 24, 'color': '#0074D9', 'family': 'Arial'}),
        style={'padding': '20px'}
    ),

    html.Div([
        html.Label('Select Country:', style={'fontSize': '20px', 'color': '#0074D9'}),
        dcc.Dropdown(
            id='country-dropdown',
            options=[{'label': country, 'value': country} for country in df['Country'].unique()],
            value=df['Country'].unique()[0],  # default value
            style={'fontSize': '16px'}
        )
    ], style={'width': '50%', 'padding': '10px', 'margin': 'auto'}),

    html.Div([
        dcc.Graph(
            id='depth-histogram',
            style={'padding': '20px'}
        ),
        dcc.Graph(
            id='magnitude-histogram',
            style={'padding': '20px'}
        )
    ], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-around'}),

    html.Div([
        dcc.Graph(
            id='country-plot',
            style={'padding': '20px'}
        ),
        dcc.Graph(
            id='country-earthquakes',
            style={'padding': '20px'}
        )
    ], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-around'}),

    # dcc.Graph(
    #     id='earth-map',
    #     style={'padding': '20px'}
    # ),

    dcc.Graph(
        id='region-pie-chart',
        style={'padding': '20px'}
    )
], 
    style={
        'fontFamily': 'Arial, sans-serif', 
        'backgroundColor': '#F7F7F7',
        'backgroundImage': 'url("https://www.transparenttextures.com/patterns/diagonal-stripes.png")',
        'backgroundSize': 'cover',
        'padding': '20px'
    }
)

# Callback to update the histograms based on selected country
@app.callback(
    [Output('depth-histogram', 'figure'),
     Output('magnitude-histogram', 'figure')],
    [Input('country-dropdown', 'value')]
)
def update_histograms(selected_country):
    filtered_df = df[df['Country'] == selected_country]
    depth_hist = px.histogram(filtered_df, x='Depth', title='Distribution of Earthquake Depths')
    magnitude_hist = px.histogram(filtered_df, x='Magnitude', title='Distribution of Earthquake Magnitudes')
    
    depth_hist.update_layout(title={'text': 'Distribution of Earthquake Depths', 'x': 0.5}, title_font={'size': 24, 'color': '#0074D9', 'family': 'Arial'})
    magnitude_hist.update_layout(title={'text': 'Distribution of Earthquake Magnitudes', 'x': 0.5}, title_font={'size': 24, 'color': '#0074D9', 'family': 'Arial'})
    
    return depth_hist, magnitude_hist

# Callback to update the country plot and earthquake count
@app.callback(
    [Output('country-plot', 'figure'),
     Output('country-earthquakes', 'figure')],
    [Input('country-dropdown', 'value')]
)
def update_country_plots(selected_country):
    country_count = df['Country'].value_counts().reset_index()
    country_count.columns = ['Country', 'Count']

    country_plot = px.bar(country_count, x='Country', y='Count', title='Earthquakes by Country')
    country_plot.update_layout(title={'text': 'Earthquakes by Country', 'x': 0.5}, title_font={'size': 24, 'color': '#0074D9', 'family': 'Arial'})

    filtered_df = df[df['Country'] == selected_country]
    country_earthquakes = px.scatter(filtered_df, x='Longitude', y='Latitude', color='Magnitude', size='Depth',
                                     title=f'Earthquakes in {selected_country}', hover_name='Country')
    country_earthquakes.update_layout(title={'text': f'Earthquakes in {selected_country}', 'x': 0.5}, title_font={'size': 24, 'color': '#0074D9', 'family': 'Arial'})

    return country_plot, country_earthquakes

# # Callback to update the Earth map
# @app.callback(
#     Output('earth-map', 'figure'),
#     [Input('country-dropdown', 'value')]
# )
# def update_earth_map(selected_country):
#     fig = px.scatter_geo(
#         df[df['Country'] == selected_country],
#         lat='Latitude',
#         lon='Longitude',
#         color='Magnitude',
#         size='Depth',
#         hover_name='Country',
#         title=f'Earthquakes in {selected_country}'
#     )
#     fig.update_geos(showcoastlines=True, coastlinecolor="Black", showland=True, landcolor="lightgray")
#     fig.update_layout(title={'text': f'Global Earthquake Map', 'x': 0.5}, title_font={'size': 24, 'color': '#0074D9', 'family': 'Arial'})
#     return fig

# Callback to update the region pie chart
@app.callback(
    Output('region-pie-chart', 'figure'),
    [Input('country-dropdown', 'value')]
)
def update_region_pie_chart(selected_country):
    region_counts = df['Lands'].value_counts().reset_index()
    region_counts.columns = ['Lands', 'Count']

    fig = px.pie(region_counts, values='Count', names='Lands', title='Earthquakes by Region')
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(title={'text': '', 'x': 0.5}, title_font={'size': 24, 'color': '#0074D9', 'family': 'Arial'})
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
