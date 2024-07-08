import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import datetime

# Load the trained model with the correct custom_objects
custom_objects = {"mse": "mean_squared_error"}
model = load_model('earthquake_depth_prediction_model.h5', custom_objects=custom_objects)

# Load the original training data to fit encoders and scalers
# Replace this path with the path to your original training data
training_df = pd.read_csv('./datasets/earthquake data.csv')
training_df['Date & Time'] = pd.to_datetime(training_df['Date & Time'])

# Preprocess the training data
training_df['year'] = training_df['Date & Time'].dt.year
training_df['month'] = training_df['Date & Time'].dt.month
training_df['day'] = training_df['Date & Time'].dt.day
training_df['hour'] = training_df['Date & Time'].dt.hour
training_df['minute'] = training_df['Date & Time'].dt.minute

# One-hot encode Lands and Country
lands_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
country_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

lands_encoded = lands_encoder.fit_transform(training_df[['Lands']])
country_encoded = country_encoder.fit_transform(training_df[['Country']])

# Prepare the training data for scaling
training_data_points = np.hstack([
    training_df[['Latitude', 'Longitude', 'Magnitude', 'year', 'month', 'day', 'hour', 'minute']].values,
    lands_encoded,
    country_encoded
])

# Scale the training data
scaler = StandardScaler()
scaler.fit(training_data_points)

# Dummy data for prediction (for June 2024)
# Replace these values with actual new data
new_data = {
    "Date & Time": ["2024-06-15 12:00", "2024-06-20 14:30"],
    "Latitude": [35.6895, -23.5505],
    "Longitude": [139.6917, -46.6333],
    "Magnitude": [5.2, 4.8],
    "Lands": ["JAPAN", "BRAZIL"],
    "Country": ["TOKYO", "SAO PAULO"]
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
for i, prediction in enumerate(predictions):
    print(f"Prediction for data point {i+1}: Depth = {prediction[0]}")
