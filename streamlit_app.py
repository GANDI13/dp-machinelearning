import streamlit as st
import pandas as pd
st.title('🤖 Machine Learning App')

st.info('This app builds a machine learning model for synthetic_power_outage in Nigeria') 

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/GANDI13/dp-machinelearning/refs/heads/master/synthetic_power_outage_data.csv')
  df

st.write('**X**')
x = df.drop('home_id', axis=1)
x

st.write('**y**')
y = df.home_id
y

# home_id,city,latitude,longitude,status,timestamp,duration_minutes,time_since_last_outage
with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='city', y='time_since_last_outage', color='home_id')

# Data preparations
with st.sidebar:
  st.header('Input Features')
  city = st.selectbox('City',('Abuja', 'Lagos', 'Kano', 'Port Harcourt', 'Enugu'))
  status = st.selectbox('Status',('ON', 'OFF'))
  duration_minutes = st.slider('Duration Minutes (mins)', 0.00, 179.00, 26.58)
  time_since_last_outage = st.slider('Time Since Last Outage (mins)', 0.00, 2026.0, 356.12)

# Create a DataFrame for model input
data = {
    'city': [city],
    'duration_minutes': [duration_minutes],
    'time_since_last_outage': [time_since_last_outage],
    'status': [status]
}

input_df = pd.DataFrame(data)
input_synthetic_power_outage_data = pd.concat([input_df, X], axis=0)

st.write("Input preview:")
st.dataframe(input_df)
        
  
