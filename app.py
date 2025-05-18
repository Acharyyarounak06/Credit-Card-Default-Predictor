import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Credit Card Default Prediction")

st.title("🔍 Credit Card Default Prediction")
st.markdown("Enter customer details below to predict the likelihood of credit card default.")

def user_input_features():
    LIMIT_BAL = st.number_input('Credit Limit (LIMIT_BAL)', min_value=1000, max_value=1000000, value=20000)
    SEX = st.selectbox('Sex (1 = Male, 2 = Female)', [1, 2])
    EDUCATION = st.selectbox('Education (1 = Graduate, 2 = University, 3 = High School, 4 = Others)', [1, 2, 3, 4])
    MARRIAGE = st.selectbox('Marriage Status (1 = Married, 2 = Single, 3 = Others)', [1, 2, 3])
    AGE = st.slider('Age', 21, 79, 35)
    PAY_0 = st.selectbox('Repayment Status - Sept (PAY_0)', list(range(-2, 9)))
    PAY_2 = st.selectbox('Repayment Status - Aug (PAY_2)', list(range(-2, 9)))
    PAY_3 = st.selectbox('Repayment Status - July (PAY_3)', list(range(-2, 9)))
    PAY_4 = st.selectbox('Repayment Status - June (PAY_4)', list(range(-2, 9)))
    PAY_5 = st.selectbox('Repayment Status - May (PAY_5)', list(range(-2, 9)))
    PAY_6 = st.selectbox('Repayment Status - April (PAY_6)', list(range(-2, 9)))
    BILL_AMT = [st.number_input(f'Bill Amount {i}', value=5000) for i in range(1, 7)]
    PAY_AMT = [st.number_input(f'Payment Amount {i}', value=2000) for i in range(1, 7)]
    
    data = [LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
            PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6] + BILL_AMT + PAY_AMT
    return np.array(data).reshape(1, -1)

input_data = user_input_features()

@st.cache_resource
def load_model():
    import joblib
    import os
    from sklearn.datasets import load_iris
    model_path = "model_rf.pkl"
    scaler_path = "scaler.pkl"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    else:
        df = pd.read_csv("UCI_Credit_Card.csv").drop("ID", axis=1)
        X = df.drop("default.payment.next.month", axis=1)
        y = df["default.payment.next.month"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
    return model, scaler

model, scaler = load_model()
scaled_input = scaler.transform(input_data)

if st.button("🔮 Predict"):
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]
    
    st.subheader("🧾 Result")
    if prediction == 1:
        st.error(f"⚠️ Likely to Default with probability {probability:.2f}")
    else:
        st.success(f"✅ Not Likely to Default with probability {1 - probability:.2f}")
