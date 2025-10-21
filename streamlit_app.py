import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.title('🤖 Machine Learning App')
st.info('This app builds a machine learning model for synthetic power outage prediction in Nigeria.')

# Load data
with st.expander('Data'):
    st.write('**Raw Data**')
    df = pd.read_csv('https://raw.githubusercontent.com/GANDI13/dp-machinelearning/refs/heads/master/synthetic_power_outage_data.csv')
    st.write(df)

# Separate X and y
st.write('**X_Raw**')
X_Raw = df.drop('home_id', axis=1)
st.write(X_Raw)

st.write('**Y_Raw**')
Y_Raw = df['home_id']
st.write(Y_Raw)

# Visualization
with st.expander('Data Visualization'):
    st.scatter_chart(data=df, x='city', y='time_since_last_outage', color='home_id')

# Sidebar input
with st.sidebar:
    st.header('Input Features')
    city = st.selectbox('Select City', ('Abuja', 'Lagos', 'Kano', 'Port Harcourt', 'Enugu'))
    status = st.selectbox('Status', ('ON', 'OFF'))
    duration_minutes = st.slider('Duration Minutes (mins)', 0.0, 179.0, 26.58)
    time_since_last_outage = st.slider('Time Since Last Outage (hrs)', 0.0, 2026.0, 356.12)

# Input DataFrame
data = {
    'city': [city],
    'duration_minutes': [duration_minutes],
    'time_since_last_outage': [time_since_last_outage],
    'status': [status]
}
input_df = pd.DataFrame(data)

# Combine input with dataset for consistent encoding
input_power_outage = pd.concat([input_df, X_Raw], axis=0)

# Encode categorical variables
encode = ['city', 'status']
df_encoded = pd.get_dummies(input_power_outage, columns=encode)
X_encoded = pd.get_dummies(X_Raw, columns=encode)
df_encoded = df_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# Select input row
input_row = df_encoded[:1]

# Ensure numeric data
X_encoded = X_encoded.apply(pd.to_numeric, errors='coerce').fillna(0)
input_row = input_row.apply(pd.to_numeric, errors='coerce').fillna(0)

# Encode target labels
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y_Raw.astype(str))

# Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_encoded, Y_encoded)

# Make prediction
prediction = clf.predict(input_row)
prediction_label = label_encoder.inverse_transform(prediction)
prediction_proba = clf.predict_proba(input_row)

# Create probability DataFrame
df_prediction_proba = pd.DataFrame(prediction_proba, columns=label_encoder.classes_)

# Display predicted city
st.subheader('🏙️ Predicted City')
power_outage_city = np.array(['Abuja', 'Lagos', 'Kano', 'Port Harcourt', 'Enugu'])
st.success(f"Predicted city (from input): {city}")

# Display results
st.subheader('🔮 Prediction Results')
st.write(f'**Predicted Outage Class / Home ID:** {prediction_label[0]}')
st.write('**Prediction Probabilities:**')
st.dataframe(df_prediction_proba)

# Display summary
with st.expander('Input Feature Summary'):
    st.write('**Input Data**')
    st.write(input_df)
    st.write('**Encoded Input Data**')
    st.write(input_row)


