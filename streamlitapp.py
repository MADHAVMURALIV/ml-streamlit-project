import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")

st.title("Titanic Survival Prediction")

age = st.slider("Select Age", 0, 100, 25)
fare = st.number_input("Enter Fare", 0.0, 500.0, 50.0)

if st.button("Predict"):
    features = np.array([[age, fare]])
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        st.success("SURVIVED")
    else:
        st.error("DID NOT SURVIVE")
# runs from -> python -m streamlit run streamlitapp.py