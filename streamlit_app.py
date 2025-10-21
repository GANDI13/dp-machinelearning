import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.title('🤖 Machine Learning App')
st.info('This app predicts power outage trends across Nigerian cities using synthetic data.')

# Load dataset
with st.expander('Dataset Overview'):
    df = pd.read_csv('https://raw.githubusercontent.com/GANDI13/dp-machinelearning/refs/heads/master/synthetic_power_outage_data.csv')
    st.write('**Raw Data**', df)

# Separate features and target
X_Raw = df.drop('home_id', axis=1)
Y_Raw = df['home_id']

# Quick Visualization
with st.expander('Visualize Data'):
    st.scatter_chart(data=df, x='city', y='time_since_last_outage', color='home_id')

# Sidebar user inputs
with st.sidebar:
    st.header('Enter Input Features')
    city = st.selectbox('Select City', ('Abuja', 'Lagos', 'Kano', 'Port Harcourt', 'Enugu'))
    status = st.selectbox('Status', ('ON', 'OFF'))
    duration_minutes = st.slider('Duration (mins)', 0.0, 179.0, 26.58)
    time_since_last_outage = st.slider('Time Since Last Outage (hrs)', 0.0, 2026.0, 356.12)

# Prepare input data
input_data = {
    'city': [city],
    'duration_minutes': [duration_minutes],
    'time_since_last_outage': [time_since_last_outage],
    'status': [status]
}
input_df = pd.DataFrame(input_data)

# Combine input with full dataset for consistent encoding
input_power_outage = pd.concat([input_df, X_Raw], axis=0)

# One-hot encode categorical features
encode_cols = ['city', 'status']
df_encoded = pd.get_dummies(input_power_outage, columns=encode_cols)
X_encoded = pd.get_dummies(X_Raw, columns=encode_cols)

# Align columns
df_encoded = df_encoded.reindex(columns=X_encoded.columns, fill_value=0)
input_row = df_encoded[:1]

# Encode target variable
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y_Raw.astype(str))

# Train the Random Forest model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_encoded, Y_encoded)

# Predict
prediction = clf.predict(input_row)
prediction_label = label_encoder.inverse_transform(prediction)
prediction_proba = clf.predict_proba(input_row)

# Display prediction results
st.subheader('🔮 Prediction Results')
st.success(f"**Predicted City:** {city}")
st.info(f"**Predicted Home ID / Outage Category:** {prediction_label[0]}")
st.write('**Prediction Probabilities:**', prediction_proba)

# Summary of input and encoding
with st.expander('Input Summary'):
    st.write('**User Input Data**', input_df)
    st.write('**Encoded Input (Model Input)**', input_row)

