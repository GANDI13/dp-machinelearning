import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.title('🤖 Machine Learning App')
st.info('This app builds a machine learning model for synthetic power outage prediction in Nigeria.')

# Load dataset
with st.expander('Dataset Overview'):
    df = pd.read_csv('https://raw.githubusercontent.com/GANDI13/dp-machinelearning/refs/heads/master/synthetic_power_outage_data.csv')
    st.write('**Raw Data:**')
    st.dataframe(df)

# --- Prepare features and target ---
X_Raw = df[['city', 'duration_minutes', 'time_since_last_outage']]
Y_Raw = df['status']

# One-hot encode city column
X_encoded = pd.get_dummies(X_Raw, columns=['city'], drop_first=False)
X_encoded = X_encoded.apply(pd.to_numeric, errors='coerce')  # ensure numeric
X_encoded = X_encoded.fillna(0)

# Encode target
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y_Raw.astype(str))

# --- Sidebar user input ---
with st.sidebar:
    st.header('Input Features')
    city = st.selectbox('Select City', ('Abuja', 'Lagos', 'Kano', 'Port Harcourt', 'Enugu'))
    duration_minutes = st.slider('Duration Minutes (mins)', 0.0, 179.0, 26.58)
    time_since_last_outage = st.slider('Time Since Last Outage (hrs)', 0.0, 2026.0, 356.12)

# Create input DataFrame
input_df = pd.DataFrame({
    'city': [city],
    'duration_minutes': [duration_minutes],
    'time_since_last_outage': [time_since_last_outage]
})

# Match encoding format
input_encoded = pd.get_dummies(input_df, columns=['city'], drop_first=False)
input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)
input_encoded = input_encoded.apply(pd.to_numeric, errors='coerce')

# --- Train model ---
clf = RandomForestClassifier(random_state=42)
clf.fit(X_encoded, Y_encoded)

# --- Make prediction ---
prediction = clf.predict(input_encoded)
prediction_label = label_encoder.inverse_transform(prediction)[0]
prediction_proba = clf.predict_proba(input_encoded)[0]

# --- Display prediction ---
st.subheader('Predicted Power Status')
if prediction_label == 'ON':
    st.success(f"The predicted status is: **{prediction_label}**")
else:
    st.error(f"The predicted status is: **{prediction_label}** ")

confidence = np.max(prediction_proba) * 100
st.metric(label="Model Confidence", value=f"{confidence:.2f}%")

# --- Show probabilities with progress columns ---
with st.expander('Prediction Probabilities'):
    df_prediction_proba = pd.DataFrame([prediction_proba], columns=label_encoder.classes_)
    st.dataframe(
        df_prediction_proba,
        column_config={
            label: st.column_config.ProgressColumn(
                label,
                help=f"Probability of status '{label}'",
                format="%.2f",
                min_value=0,
                max_value=1
            ) for label in label_encoder.classes_
        },
        use_container_width=True
    )

# --- Show input summary ---
with st.expander('Input Summary'):
    st.write('**User Input:**')
    st.write(input_df)
    st.write('**Encoded Model Input:**')
    st.write(input_encoded)
