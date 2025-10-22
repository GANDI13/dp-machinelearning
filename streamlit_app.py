import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.title('🤖 Machine Learning App')
st.info('This app predicts the likely city based on power outage patterns in Nigeria.')

# Load data
with st.expander('Data'):
    st.write('**Raw Data**')
    df = pd.read_csv('https://raw.githubusercontent.com/GANDI13/dp-machinelearning/refs/heads/master/synthetic_power_outage_data.csv')
    st.write(df)

# Separate features and target
X_Raw = df[['duration_minutes', 'time_since_last_outage', 'status']]
Y_Raw = df['city']

# Encode categorical variables
X_encoded = pd.get_dummies(X_Raw, columns=['status'])
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y_Raw)

# Sidebar input
with st.sidebar:
    st.header('Input Features')
    status = st.selectbox('Status', ('ON', 'OFF'))
    duration_minutes = st.slider('Duration Minutes (mins)', 0.0, 179.0, 26.58)
    time_since_last_outage = st.slider('Time Since Last Outage (hrs)', 0.0, 2026.0, 356.12)

# Input DataFrame
input_df = pd.DataFrame({
    'duration_minutes': [duration_minutes],
    'time_since_last_outage': [time_since_last_outage],
    'status': [status]
})

# Match encoding format
input_encoded = pd.get_dummies(input_df, columns=['status'])
input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# Train model
clf = RandomForestClassifier()
clf.fit(X_encoded, Y_encoded)

# Predict city
prediction = clf.predict(input_encoded)
predicted_city = label_encoder.inverse_transform(prediction)[0]

# Display results
st.subheader('Predicted City')
st.success(f'The predicted city is: **{predicted_city}**')

with st.expander('Prediction Details'):
    prediction_proba = clf.predict_proba(input_encoded)
    df_prediction_proba = pd.DataFrame(prediction_proba, columns=label_encoder.classes_)
    st.write('**Prediction Probabilities:**')
    st.write(df_prediction_proba)

with st.expander('Input Summary'):
    st.write(input_df)
