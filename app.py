import streamlit as st
import pickle
import numpy as np

# -------- Load model & scaler --------
@st.cache_resource
def load_model():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)

    return scaler, model

scaler, model = load_model()

# -------- UI --------
st.title("🌳 Loan Approval Prediction App")
st.write("Enter feature values to get prediction:")

f1 = st.number_input("Applicant Income", min_value=0.0, max_value=19988.0, value=0.0)
f2 = st.number_input("Credit Score", min_value=0.0, max_value=799.0, value=0.0)
f3 = st.number_input("DTI Ratio", min_value=0.0, max_value=0.6, value=0.0)



f4 = st.number_input("Loan Amount", min_value=0.0, max_value=39995.0, value=0.0)


# -------- Prediction --------
if st.button("Predict"):
    input_data = np.array([[f1, f2, f3, f4]])
    scaled_data = scaler.transform(input_data)

    # get probability of class 1 (Approved)
    proba = model.predict_proba(scaled_data)[0][1]

    # Hardcoded thresholds
    if proba >= 0.8:
        st.success(f"✅ Approved (Probability: {proba:.2f})")
    elif proba >= 0.5:
        st.warning(f"⚠️ Moderate (Probability: {proba:.2f})")
    else:
        st.error(f"❌ Not Approved (Probability: {proba:.2f})")