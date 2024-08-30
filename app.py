import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import plotly.express as px
from dash.exceptions import PreventUpdate
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the dataset
df = pd.read_csv('./datasets/earthquake data.csv')

# Data cleaning
df['Date & Time'] = pd.to_datetime(df['Date & Time'])

# Handle negative depth values
df['Depth'] = df['Depth'].apply(lambda x: x if x > 0 else 1)

# Load the trained model with custom loss
custom_objects = {"mse": "mean_squared_error"}
model = load_model('earthquake_depth_prediction_model.h5', custom_objects=custom_objects)

# Initialize encoders and scaler
lands_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
country_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
scaler = StandardScaler()

# Fit encoders and scaler on training data
df['year'] = df['Date & Time'].dt.year
df['month'] = df['Date & Time'].dt.month
df['day'] = df['Date & Time'].dt.day
df['hour'] = df['Date & Time'].dt.hour
df['minute'] = df['Date & Time'].dt.minute

lands_encoded = lands_encoder.fit_transform(df[['Lands']])
country_encoded = country_encoder.fit_transform(df[['Country']])

training_data_points = np.hstack([
    df[['Latitude', 'Longitude', 'Magnitude', 'year', 'month', 'day', 'hour', 'minute']].values,
    lands_encoded,
    country_encoded
])

scaler.fit(training_data_points)

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
            value=df['Country'].unique()[0],
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

    dcc.Graph(
        id='earth-map',
        style={'padding': '20px'}
    ),

    dcc.Graph(
        id='region-pie-chart',
        style={'padding': '20px'}
    ),

    html.Div([
        html.H2("Prediction", style={'textAlign': 'center', 'color': '#0074D9', 'fontWeight': 'bold'}),

        html.Div([
            html.Label('Latitude:', style={'fontSize': '18px', 'color': '#0074D9'}),
            dcc.Input(id='input-latitude', type='number', value=0.0, style={'marginBottom': '10px', 'width': '97%', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #0074D9'}),
            
            html.Label('Longitude:', style={'fontSize': '18px', 'color': '#0074D9'}),
            dcc.Input(id='input-longitude', type='number', value=0.0, style={'marginBottom': '10px', 'width': '97%', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #0074D9'}),
            
            html.Label('Magnitude:', style={'fontSize': '18px', 'color': '#0074D9'}),
            dcc.Input(id='input-magnitude', type='number', value=0.0, style={'marginBottom': '10px', 'width': '97%', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #0074D9'}),
            
            html.Label('Date & Time:', style={'fontSize': '18px', 'color': '#0074D9'}),
            dcc.Input(id='input-date-time', type='text', placeholder='YYYY-MM-DD HH:MM', style={'marginBottom': '10px', 'width': '97%', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #0074D9'}),
            
            html.Label('Lands:', style={'fontSize': '18px', 'color': '#0074D9'}),
            dcc.Input(id='input-lands', type='text', style={'marginBottom': '10px', 'width': '97%', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #0074D9'}),
            
            html.Label('Country:', style={'fontSize': '18px', 'color': '#0074D9'}),
            dcc.Dropdown(
                id='input-country',
                options=[{'label': country, 'value': country} for country in df['Country'].unique()],
                value=df['Country'].unique()[0],
                style={'marginBottom': '10px', 'width': '97%', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #0074D9'}
            ),
        ], style={'display': 'grid', 'gap': '10px', 'padding': '20px', 'backgroundColor': '#E5F6FE', 'borderRadius': '10px', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 'maxWidth': '600px', 'margin': 'auto'}),
        
        html.Button('Predict Depth', id='submit-button', n_clicks=0, style={'display': 'block', 'margin': '20px auto', 'padding': '15px 30px', 'backgroundColor': '#0074D9', 'color': 'white', 'fontSize': '18px', 'fontWeight': 'bold', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
        
        html.Div(id='prediction-output', style={'textAlign': 'center', 'color': '#0074D9', 'fontSize': '20px', 'fontWeight': 'bold', 'marginTop': '20px'})
    ], style={'maxWidth': '800px', 'margin': 'auto'})
],

style={
    'fontFamily': 'Arial, sans-serif', 
    'backgroundColor': '#F7F7F7',
    'backgroundImage': 'url("https://www.transparenttextures.com/patterns/diagonal-stripes.png")',
    'backgroundSize': 'cover',
    'padding': '20px'
})

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

# Callback to update the Earth map
@app.callback(
    Output('earth-map', 'figure'),
    [Input('country-dropdown', 'value')]
)
def update_earth_map(selected_country):
    fig = px.scatter_geo(
        df[df['Country'] == selected_country],
        lat='Latitude',
        lon='Longitude',
        color='Magnitude',
        size='Depth',
        hover_name='Country',
        title=f'Earthquakes in {selected_country}'
    )
    fig.update_geos(showcoastlines=True, coastlinecolor="Black", showland=True, landcolor="lightgray")
    fig.update_layout(title={'text': f'Global Earthquake Map', 'x': 0.5}, title_font={'size': 24, 'color': '#0074D9', 'family': 'Arial'})
    return fig

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


# Callback to update the prediction output
@app.callback(
    Output('prediction-output', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('input-latitude', 'value'),
     State('input-longitude', 'value'),
     State('input-magnitude', 'value'),
     State('input-date-time', 'value'),
     State('input-lands', 'value'),
     State('input-country', 'value')]
)
def update_prediction(n_clicks, latitude, longitude, magnitude, date_time, lands, country):
    if n_clicks > 0:
        # Create new_data dictionary and process the inputs
        new_data = {
            "Date & Time": [date_time],
            "Latitude": [latitude],
            "Longitude": [longitude],
            "Magnitude": [magnitude],
            "Lands": [lands],
            "Country": [country]
        }

        new_df = pd.DataFrame(new_data)

        # Preprocess the new data
        new_df['Date & Time'] = pd.to_datetime(new_df['Date & Time'])
        new_df['year'] = new_df['Date & Time'].dt.year
        new_df['month'] = new_df['Date & Time'].dt.month
        new_df['day'] = new_df['Date & Time'].dt.day
        new_df['hour'] = new_df['Date & Time'].dt.hour
        new_df['minute'] = new_df['Date & Time'].dt.minute

        # Transform the new data using fitted encoders
        lands_encoded_new = lands_encoder.transform(new_df[['Lands']])
        country_encoded_new = country_encoder.transform(new_df[['Country']])

        # Prepare new data for prediction
        new_data_points = np.hstack([
            new_df[['Latitude', 'Longitude', 'Magnitude', 'year', 'month', 'day', 'hour', 'minute']].values,
            lands_encoded_new,
            country_encoded_new
        ])

        # Scale the new data using the fitted scaler
        new_data_points_scaled = scaler.transform(new_data_points)

        # Reshape the data to fit the model input
        new_data_points_scaled = new_data_points_scaled.reshape((new_data_points_scaled.shape[0], 1, new_data_points_scaled.shape[1]))

        # Make predictions
        predictions = model.predict(new_data_points_scaled)

        # Output the predictions
        prediction_output = f"Predicted Depth: {predictions[0][0]}"

        return prediction_output

    return ''

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
