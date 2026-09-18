import streamlit as st
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np

# Page config
st.set_page_config(page_title="Churn Predictor", layout="wide")

st.title("Customer Churn Prediction")
st.write("Enter customer details to see churn probability.")

# Load model
model = tf.keras.models.load_model('churn_ann_model.h5')
scaler = joblib.load('scaler.pkl')

# Layout (2 columns)
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Information")

    age = st.number_input("Age", 18, 100, 35)
    gender = st.selectbox("Gender", ["Female", "Male"])
    income = st.number_input("Estimated Salary ($)", 0, 200000, 50000)
    credit_score = st.number_input("Credit Score", 300, 900, 650)

with col2:
    st.subheader("Account Details")

    balance = st.number_input("Account Balance ($)", 0, 200000, 50000)
    products = st.slider("Number of Products", 1, 4, 2)
    tenure = st.slider("Tenure (years)", 0, 10, 5)
    credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    active_member = st.selectbox("Is Active Member?", ["Yes", "No"])

# Button
if st.button("Predict Churn Probability"):

    # Encoding
    gender_val = 1 if gender == "Male" else 0
    credit_card_val = 1 if credit_card == "Yes" else 0
    active_val = 1 if active_member == "Yes" else 0

    # Dummy geography (to match 9 features)
    geo_germany = 0
    geo_spain = 0

    # Create dataframe
    input_df = pd.DataFrame([[
        age, gender_val, income, credit_score, balance,
        products, tenure,
        geo_germany, geo_spain
    ]],
    columns=[
        'Age','Gender','Income','CreditScore','Balance',
        'NumOfProducts','Tenure',
        'Geography_Germany','Geography_Spain'
    ])

    # Scale
    num_cols = ['Age','Income','CreditScore','Balance','NumOfProducts','Tenure']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Predict
    prob = model.predict(input_df, verbose=0)[0][0]

    st.success(f"Churn Probability: {prob:.2%}")