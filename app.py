import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------- Load model ----------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("Loan Approval Prediction App")
st.write("Enter applicant information to predict whether the loan is likely to be approved.")

# ---------- Input form ----------
with st.form("loan_form"):
    st.subheader("Numeric Information")
    applicant_income = st.number_input("ApplicantIncome", min_value=0.0, step=500.0)
    coapplicant_income = st.number_input("CoapplicantIncome", min_value=0.0, step=500.0)
    loan_amount = st.number_input("LoanAmount (in thousands)", min_value=0.0, step=10.0)
    loan_term = st.number_input("Loan_Amount_Term (in days)", min_value=0.0, step=30.0)
    credit_history = st.selectbox("Credit_History", [1.0, 0.0], format_func=lambda x: "Good (1.0)" if x == 1.0 else "Bad (0.0)")

    st.subheader("Categorical Information")
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self_Employed", ["No", "Yes"])
    property_area = st.selectbox("Property_Area", ["Urban", "Semiurban", "Rural"])

    submitted = st.form_submit_button("Predict Loan Status")

# ---------- Make prediction ----------
if submitted:
    input_data = {
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "Property_Area": property_area,
    }

    X_input = pd.DataFrame([input_data])

    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][1]  # probability of approval (class 1)

    if pred == 1:
        st.success(f"✅ Prediction: LOAN APPROVED (probability: {proba:.2%})")
    else:
        st.error(f"❌ Prediction: LOAN NOT APPROVED (probability of approval: {proba:.2%})")

    st.caption("Model: Logistic Regression with preprocessing pipeline (trained in your final project).")
