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
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"]) # <-- Added this
income = st.sidebar.number_input("Applicant Income ($)", value=5000)
co_income = st.sidebar.number_input("Co-applicant Income ($)", value=0)
loan_amt = st.sidebar.number_input("Loan Amount ($)", value=150)
term = st.sidebar.number_input("Term (Days)", value=360)
credit = st.sidebar.selectbox("Credit History", [1.0, 0.0])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.sidebar.button("Analyze Risk"):
    # 1. Feature Engineering
    total_income_log = np.log1p(income + co_income)

    # 2. Create Dataframe 
    # CRITICAL: These must be in the EXACT same order as your X_train.columns
    input_df = pd.DataFrame({
        'Dependents': [3.0 if dependents == "3+" else float(dependents)],
        'LoanAmount': [loan_amt],
        'Loan_Amount_Term': [term],
        'Credit_History': [credit],
        'Gender_Male': [1 if gender == "Male" else 0],
        'Married_Yes': [1 if married == "Yes" else 0],
        'Education_Not Graduate': [1 if education == "Not Graduate" else 0],
        'Self_Employed_Yes': [1 if self_employed == "Yes" else 0], # <-- Added this
        'Property_Area_Semiurban': [1 if property_area == "Semiurban" else 0],
        'Property_Area_Urban': [1 if property_area == "Urban" else 0],
        'Total_Income_log': [total_income_log]
    })

    # 3. Scaling
    cols_to_scale = ['Total_Income_log', 'LoanAmount', 'Loan_Amount_Term']
    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

    # 4. Predict
    prob = model.predict_proba(input_df)[:, 1][0]
    threshold = 0.6

    # Display Result
    st.divider()
    if prob >= threshold:
        st.success(f"### ✅ Approved \n**Confidence Score:** {prob:.2%}")
    else:
        st.error(f"### ❌ Rejected \n**Confidence Score:** {prob:.2%}")

    # 5. Sigmoid Plot
    st.subheader("Risk Mapping")
    user_z = np.log(prob / (1 - prob))
    z_curve = np.linspace(-6, 6, 100)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(z_curve, 1/(1+np.exp(-z_curve)), color='gray', alpha=0.3, label="Model Probability Curve")
    ax.axhline(y=threshold, color='red', linestyle='--', label=f'Decision Threshold ({threshold})')
    ax.scatter(user_z, prob, color='teal', s=150, zorder=5, label="Applicant Score")
    
    ax.set_xlabel("Internal Log-Odds")
    ax.set_ylabel("Probability")
    ax.legend()
    st.pyplot(fig)