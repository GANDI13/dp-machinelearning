import streamlit as st
import pandas as pd
st.title('🤖 Machine Learning App')

st.info('This app builds a machine learning model for synthetic_power_outage in Nigeria') 

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/GANDI13/dp-machinelearning/refs/heads/master/synthetic_power_outage_data.csv')
  df

st.write('**X**')
x = df.drop('city', axis=1)
x

st.write('**y**')
y = df.city
y

# home_id,city,latitude,longitude,status,timestamp,duration_minutes,time_since_last_outage
with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='home_id', y='time_since_last_outage', color='city')

# Data preparations
with st.sidebar:
  st.header('Input Features')
  city = st.selectbox('city','Abuja', 'Lagos', 'Kano', 'Port Harcourt', 'Enugu'))
  status = st.selectbox('status','ON', 'OFF'))
  time_since_last_outage = st.slider('Time_Since_Last_Outage', 0.00, 2026.00, 356.12)
  
                   
