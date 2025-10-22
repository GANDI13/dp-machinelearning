import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.title('🤖 Machine Learning App')
st.info('This app predicts the power outage status (ON/OFF) across Nigerian cities using synthetic data.')

# Load dataset
with st.expander('Dataset Overview'):
    df = pd.read_csv('https://raw.githubusercontent.com/GANDI13/dp-machinelearning/refs/heads/master/synthetic_power_outage_data.csv')
    st.write('**Raw Data**', df)

# Separate features (X) and target (y)
X_Raw = df.drop('status', axis=1)
Y_Raw = df['status']

# Visualization
with st.expander('Visualize Data'):
    st.scatter_chart(data=df, x='city', y='time_since_last_outage', color='status')

# Sidebar input
with st.sidebar:
    st.header('Enter Input Features')
    city = st.selectbox('Select City', ('Abuja', 'Lagos', 'Kano', 'Port Harcourt', 'Enugu'))
    duration_minutes = st.slider('Duration (mins)', 0.0, 179.0, 26.58)
    time_since_last_outage = st.slider('Time Since Last Outage (hrs)', 0.0, 2026.0, 356.12)

# Prepare input
input_data = {
    'city': [city],
    'duration_minutes': [duration_minutes],
    'time_since_last_outage': [time_since_last_outage]
}
input_df = pd.DataFrame(input_data)

# Combine input + dataset for consistent encoding
input_power_outage = pd.concat([input_df, X_Raw], axis=0)

# One-hot encode categorical variables
encode_cols = ['city']
df_encoded = pd.get_dummies(input_power_outage, columns=encode_cols)
X_encoded = pd.get_dummies(X_Raw, columns=encode_cols)

# Align columns
df_encoded = df_encoded.reindex(columns=X_encoded.columns, fill_value=0)
input_row = df_encoded[:1]

# Encode target
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y_Raw.astype(str))

# Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_encoded, Y_encoded)

# Predict
prediction = clf.predict(input_row)
prediction_label = label_encoder.inverse_transform(prediction)
prediction_proba = clf.predict_proba(input_row)

# Convert probabilities to DataFrame
df_prediction_proba = pd.DataFrame(prediction_proba, columns=label_encoder.classes_)

# Display prediction results
st.subheader('🔌 Prediction Results')
st.success(f"**Predicted Status:** {prediction_label[0]}")
st.write('**Prediction Probabilities:**')

# Progress bars for each class probability
st.dataframe(
    df_prediction_proba,
    column_config={
        'OFF': st.column_config.ProgressColumn(
            'OFF Probability',
            format="%.2f",
            width="medium",
            min_value=0,
            max_value=1
        ),
        'ON': st.column_config.ProgressColumn(
            'ON Probability',
            format="%.2f",
            width="medium",
            min_value=0,
            max_value=1
        ),
    },
    hide_index=True
)

# Summary of input
with st.expander('Input Summary'):
    st.write('**User Input Data**', input_df)
    st.write('**Encoded Input (Model Input)**', input_row)



