import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('🤖 Machine Learning App')
st.info('This app builds a machine learning model for synthetic_power_outage in Nigeria')

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

# Combine input with full dataset for consistent encoding
input_power_outage = pd.concat([input_df, X_Raw], axis=0)

# Encode categorical variables
encode = ['city', 'status']
df_encoded = pd.get_dummies(input_power_outage, columns=encode)

# Ensure same columns between train and input
X_encoded = pd.get_dummies(X_Raw, columns=encode)
df_encoded = df_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# Select only the first row (our input)
input_row = df_encoded[:1]

# Encode Y (for demo — assumes numeric IDs map to 3 outage categories)
target_mapper = {
    0: 'No Outage',
    1: 'Short Outage',
    2: 'Long Outage'
}
def target_encode(vl):
    return target_mapper.get(vl, 'Unknown')

# Try encoding the target if possible
try:
    Y = Y_Raw.apply(target_encode)
except Exception:
    Y = Y_Raw  # fallback

# Train model
clf = RandomForestClassifier()
clf.fit(X_encoded, Y_Raw)  # use numeric Y_Raw for training

# Predict
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

# Display predictions
st.subheader('Prediction Results')
st.write(f'**Predicted Home ID / Outage Class:** {prediction[0]}')
st.write('**Prediction Probability:**')
st.write(prediction_proba)

# Summary
with st.expander('Input Feature Summary'):
    st.write('**Input Data**')
    st.write(input_df)
    st.write('**Encoded Input Data**')
    st.write(input_row)

        
