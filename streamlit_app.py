import streamlit as st
import pandas as pd
st.title('🤖 Machine Learning App')

st.info('This app builds a machine learning model for synthetic_power_outage in Nigeria') 

df = pd.read_csv('https://raw.githubusercontent.com/GANDI13/dp-machinelearning/refs/heads/master/synthetic_power_outage_data.csv')
df
