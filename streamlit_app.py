import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('🤖 Machine Learning App')
st.info('This app builds a machine learning model for synthetic_power_outage in Nigeria')

# Loading data
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
input_power_outage = pd.concat([input_df, X_Raw], axis=0)

# Encode categorical variables
encode = ['city', 'status']
df_power_outage = pd.get_dummies(input_power_outage, columns=encode)
input_row = df_power_outage[:1]

# Target mapping
target_mapper = {
    0: 'No Outage',
    1: 'Short Outage',
    2: 'Long Outage'
}

def target_encode(vl):
    return target_mapper.get(vl, 'Unknown')

# Numeric labels (0,1,2) in Y_Raw
try:
    Y = Y_Raw.apply(target_encode)
    st.write('**Encoded Target (Y)**')
    st.write(Y)
except Exception as e:
    st.warning(f'Skipping target encoding: {e}')

# Display results
with st.expander('Input Feature Summary'):
    st.write('**Input Power Outage**')
    st.write(input_df)
    st.write('**Combined Input Data**')
    st.write(input_power_outage)
    st.write('**Encoded Input Power Outage**')
    st.write(input_row)
    
# Model training and Inference
# Train ML Model
clf = RandomForestClassifier() 
clf.fit(X_Raw, Y)

# Apply Model to make predictions
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

prediction_proba

        
