import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Load model and scaler ---
os.chdir(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')

# --- Page config ---
st.set_page_config(page_title="Heart Disease Predictor", page_icon=None)

st.title("Heart Disease Prediction")
st.write("Enter patient details below to predict the likelihood of heart disease.")

# --- Input Form ---
st.subheader("Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=20, max_value=100, value=52)
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=125)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=212)
    fbs = st.selectbox("Fasting Blood Sugar > 120?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=168)

with col2:
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                      format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"][x])
    restecg = st.selectbox("Resting ECG", options=[0, 1, 2],
                           format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])
    exang = st.selectbox("Exercise Induced Angina?", options=[0, 1],
                         format_func=lambda x: "Yes" if x == 1 else "No")

with col3:
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of ST Segment", options=[0, 1, 2],
                         format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    ca = st.selectbox("Major Vessels Colored (0-4)", options=[0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3],
                        format_func=lambda x: ["Unknown", "Normal", "Fixed Defect", "Reversible Defect"][x])

# --- Predict Button ---
if st.button("Predict"):

    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs,
                                 restecg, thalach, exang, oldpeak, slope, ca, thal]],
                               columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                                        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

    # ✅ CHANGE: added .values to convert DataFrame to numpy array
    # XGBoost was trained on numpy array (no column names), so feeding a
    # DataFrame with column names causes a mismatch error
    prediction = model.predict(input_data.values)[0]
    probability = model.predict_proba(input_data.values)[0][1]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("High Risk of Heart Disease detected")
        st.metric("Probability of Disease", f"{probability:.1%}")
        st.warning("Please consult a cardiologist immediately.")
    else:
        st.success("Low Risk — No Heart Disease detected")
        st.metric("Probability of Disease", f"{probability:.1%}")
        st.info("Maintain a healthy lifestyle and regular checkups.")

    with st.expander("View Input Summary"):
        st.dataframe(input_data)