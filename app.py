import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ===== LOAD MODEL =====
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Credit Card Default Prediction", layout="centered")

st.title("💳 Credit Card Default Prediction")
st.write("Prediksi apakah nasabah berpotensi **gagal bayar** menggunakan XGBoost.")

st.divider()

# ===== INPUT FEATURES =====
limit_bal = st.number_input("Limit Balance", min_value=0.0, value=200000.0)
sex = st.selectbox("Gender", [1, 2])  # 1=Male, 2=Female
education = st.selectbox("Education Level", [1, 2, 3, 4])
marriage = st.selectbox("Marital Status", [1, 2, 3])
age = st.number_input("Age", min_value=18, max_value=100, value=30)

bill_amt1 = st.number_input("Bill Amount Month 1", value=50000.0)
pay_amt1 = st.number_input("Payment Amount Month 1", value=20000.0)

# ===== PREDICTION =====
if st.button("Predict"):
    input_data = np.array([[
        limit_bal, sex, education, marriage, age,
        bill_amt1, pay_amt1
    ]])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Default Risk (Probability: {prob:.2f})")
    else:
        st.success(f"✅ No Default Risk (Probability: {prob:.2f})")
