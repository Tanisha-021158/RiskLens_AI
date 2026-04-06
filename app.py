import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="RiskLens AI", page_icon="🔍")

# --- 2. LOAD ALL ASSETS ---
@st.cache_resource
def load_assets():
    model = joblib.load('loan_model.joblib')
    scaler = joblib.load('scaler.joblib')
    num_imputer = joblib.load('num_imputer.joblib')
    cat_imputer = joblib.load('cat_imputer.joblib')
    return model, scaler, num_imputer, cat_imputer

model, scaler, num_imputer, cat_imputer = load_assets()

# --- 3. UI ---
st.title("🔍 RiskLens AI")
st.markdown("### Intelligent Credit Risk Assessment")

# Sidebar inputs
st.sidebar.header("Applicant Profile")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
income = st.sidebar.number_input("Applicant Income ($)", value=5000)
co_income = st.sidebar.number_input("Co-applicant Income ($)", value=0)
loan_amt = st.sidebar.number_input("Loan Amount ($)", value=150)
term = st.sidebar.number_input("Term (Days)", value=360)
credit = st.sidebar.selectbox("Credit History", [1.0, 0.0])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.sidebar.button("Analyze Risk"):
    # 1. Feature Engineering (Log Transform on Total Income)
    total_income_log = np.log1p(income + co_income)

    # 2. Apply Clipping (Using your calculated Raw Log/Numerical Limits)
    # This keeps inputs within the mathematical "knowledge" of your model
    total_income_log_clipped = np.clip(total_income_log, 7.48288, 9.72513)
    loan_amt_clipped = np.clip(loan_amt, -2.0, 270.0)

    # 3. Create Dataframe 
    # Order matched to: ['Dependents', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 
    # 'Gender_Male', 'Married_Yes', 'Education_Not Graduate', 'Self_Employed_Yes', 
    # 'Property_Area_Semiurban', 'Property_Area_Urban', 'Total_Income_log']
    input_df = pd.DataFrame({
        'Dependents': [3.0 if dependents == "3+" else float(dependents)],
        'LoanAmount': [loan_amt_clipped],
        'Loan_Amount_Term': [term],
        'Credit_History': [credit],
        'Gender_Male': [1 if gender == "Male" else 0],
        'Married_Yes': [1 if married == "Yes" else 0],
        'Education_Not Graduate': [1 if education == "Not Graduate" else 0],
        'Self_Employed_Yes': [1 if self_employed == "Yes" else 0],
        'Property_Area_Semiurban': [1 if property_area == "Semiurban" else 0],
        'Property_Area_Urban': [1 if property_area == "Urban" else 0],
        'Total_Income_log': [total_income_log_clipped]
    })

    # 4. Scaling
    # Applying the scaler to the specific columns it was trained on
    cols_to_scale = ['Total_Income_log', 'LoanAmount', 'Loan_Amount_Term']
    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

    # 5. Prediction
    # Getting the probability of class 1 (Approved)
    prob = model.predict_proba(input_df)[:, 1][0]
    threshold = 0.6

    # --- DISPLAY RESULT ---
    st.divider()
    if prob >= threshold:
        st.success(f"### ✅ Approved \n**Confidence Score:** {prob:.2%}")
    else:
        st.error(f"### ❌ Rejected \n**Confidence Score:** {prob:.2%}")

    # --- SIGMOID PLOT ---
    st.subheader("Risk Mapping")
    
    # Calculate Log-Odds (Z) for the scatter point
    # We use a tiny epsilon to prevent math errors at exactly 0 or 1
    eps = 1e-9
    user_z = np.log((prob + eps) / (1 - prob + eps))
    
    z_curve = np.linspace(-6, 6, 100)
    sigmoid_curve = 1 / (1 + np.exp(-z_curve))
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(z_curve, sigmoid_curve, color='gray', alpha=0.3, label="Model Probability Curve")
    ax.axhline(y=threshold, color='red', linestyle='--', label=f'Decision Threshold ({threshold})')
    ax.scatter(user_z, prob, color='teal', s=150, zorder=5, label="Applicant Score")
    
    # Visual cues for zones
    ax.fill_between(z_curve, threshold, 1, color='green', alpha=0.05)
    ax.fill_between(z_curve, 0, threshold, color='red', alpha=0.05)
    
    ax.set_xlabel("Internal Log-Odds")
    ax.set_ylabel("Probability")
    ax.legend()
    st.pyplot(fig)
    
    st.caption("The graph shows where the applicant sits on the probability curve relative to the risk threshold.")