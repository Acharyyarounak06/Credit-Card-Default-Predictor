# 🧱 1. Architecture Document (System Design)
# 🔷 System Overview
A machine learning pipeline that takes user demographic and financial data to predict credit card default risk. Deployed via a Streamlit web app.

# 🗂️ Architecture Components

[User Input Form] --> [Streamlit App] --> [Preprocessing] --> [ML Model]
                                          |                   ↓
                                          +-----> [Prediction Output]


# 🧩 Components Explained
# User Interface (Streamlit):

1. Collects input data (age, limit balance, bill amounts, etc.)
2. Sends data for prediction.

# Preprocessing Layer:
1. Standardizes input using the same scaler used during training.

# Model Layer:
1. L1oads a RandomForestClassifier model (or best-performing model).
2. Outputs a binary prediction (Default: Yes/No).

# Output Layer:
1. Displays prediction on the Streamlit page.
