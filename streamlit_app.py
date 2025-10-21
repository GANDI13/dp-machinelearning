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

                   
