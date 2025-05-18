import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="Credit Card Default Prediction")

# Optional logo
st.markdown("""
    <div style="text-align:center;">
        <img src="logo.png" alt="Bank Logo" width="120"/>
    </div>
""", unsafe_allow_html=True)

# Dark mode toggle
dark_mode = st.toggle("🌙 Enable Dark Mode", value=True)

# Dynamic CSS injection
if dark_mode:
    st.markdown("""
        <style>
        body, .stApp {
            background-color: #121212;
            color: #FF3C38;
        }
        h1, h2, h3, h4 {
            color: #FF3C38;
        }
        .stButton>button {
            background-color: #FF3C38;
            color: #FFFFFF;
            border-radius: 8px;
            padding: 0.5em 1em;
        }
        .stButton>button:hover {
            background-color: #e60000;
        }
        input, select, textarea {
            background-color: #1c1c1c !important;
            color: #FF3C38 !important;
        }
        .stNumberInput, .stSelectbox, .stSlider {
            background-color: #1c1c1c;
            color: #FF3C38;
        }
        .st-success, .st-error {
            background-color: #2b2b2b;
            border-left: 5px solid #FF3C38;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body, .stApp {
            background-color: #FFFFFF;
            color: #000000;
        }
        h1, h2, h3, h4 {
            color: #B00020;
        }
        .stButton>button {
            background-color: #B00020;
            color: #FFFFFF;
            border-radius: 8px;
            padding: 0.5em 1em;
        }
        .stButton>button:hover {
            background-color: #8c0016;
        }
        input, select, textarea {
            background-color: #f2f2f2 !important;
            color: #000000 !important;
        }
        .stNumberInput, .stSelectbox, .stSlider {
            background-color: #f9f9f9;
            color: #000000;
        }
        .st-success, .st-error {
            background-color: #f9f9f9;
            border-left: 5px solid #B00020;
        }
        </style>
    """, unsafe_allow_html=True)

# Animated Title
st.markdown("""
    <h1 style='text-align: center; color: #FF3C38; animation: fadeIn 2s ease-out;'>🔍 Credit Card Default Predictor</h1>
    <style>
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("Enter customer details below to predict the likelihood of credit card default.")

# Input features
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

# Load model & scaler
@st.cache_resource
def load_model():
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

# Prediction
if st.button("🔮 Predict"):
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.markdown("""
        <style>
        .fade-result {
            animation: fadeInUp 1s ease-out;
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="fade-result">', unsafe_allow_html=True)

    st.subheader("🧾 Result")
    if prediction == 1:
        st.error(f"⚠️ Likely to Default with probability {probability:.2f}")
    else:
        st.success(f"✅ Not Likely to Default with probability {1 - probability:.2f}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Probability Chart
    fig, ax = plt.subplots()
    ax.bar(['Not Default', 'Default'], [1 - probability, probability], color=['green', 'red'])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probability")
    st.pyplot(fig)
    st.markdown("### Feature Importance")
