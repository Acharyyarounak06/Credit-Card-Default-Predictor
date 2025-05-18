# 🔸 Low-Level Design (LLD)

# File Structure Overview:

📁 Credit-Card-Default-Predictor
├── model.pkl → Serialized ML model
├── credit_card_default.ipynb → Notebook for data exploration, training, and evaluation
├── requirements.txt → Python dependencies
├── app.py → Streamlit web app
├── utils.py (optional) → Helper functions (e.g., data cleaning, prediction)
├── dataset.csv → Input dataset

# Module Details:

# data_preprocessing():

Drop irrelevant columns, handle nulls, encode categorical data

Standardize numerical features

# train_model():

Split dataset, train models, evaluate accuracy and ROC-AUC

Save best-performing model as model.pkl

# predict_default(input_data):

Load model.pkl

Apply preprocessing pipeline

Return default prediction (Yes/No)

# Streamlit UI:

Sidebar inputs for age, bill amount, repayment status, etc.

Submit button triggers prediction

Display result with success/warning message

# Model Pipeline:
Raw Data → Preprocessing → Model Training → Serialization → User Input via Streamlit → Model Inference → Output Display
