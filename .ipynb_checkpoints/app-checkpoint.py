import streamlit as st
import pickle
import nu py as np


def load_model():
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return scaler, model

scaler, model = load_model()

st.title("Random Forest Prediction App 🌳")

f1 = st.number_input("Feature 1")
f2 = st.number_input("Feature 2")
f3 = st.number_input("Feature 3")
f4 = st.number_input("Feature 4")

if st.button("Predict"):
    data = np.array([[f1, f2, f3,f4]])
    result = model.predict(data)

    if result[0] == 1:
        st.success("✅ Positive Class")
    else:
        st.error("❌ Negative Class")