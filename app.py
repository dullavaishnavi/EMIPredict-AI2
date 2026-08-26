import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="EMIPredict AI", page_icon="🏦", layout="wide")

BASE = Path(__file__).resolve().parent
CLASS_MODEL = joblib.load(BASE / "models" / "best_classification_model.joblib")
REG_MODEL = joblib.load(BASE / "models" / "best_regression_model.joblib")

FEATURE_COLUMNS = list(CLASS_MODEL.feature_names_in_)

st.title("🏦 EMIPredict AI")
st.caption("Intelligent Financial Risk Assessment — EMI Eligibility + Maximum Monthly EMI")


def build_features(x: dict) -> pd.DataFrame:
    """Build the exact 72-column matrix expected by the exported XGBoost models."""
    d = pd.DataFrame([x])
    income = d["monthly_salary"].clip(lower=1)

    total_expenses = (
        d["monthly_rent"].fillna(0)
        + d["school_fees"].fillna(0)
        + d["college_fees"].fillna(0)
        + d["travel_expenses"].fillna(0)
        + d["groceries_utilities"].fillna(0)
        + d["other_monthly_expenses"].fillna(0)
        + d["current_emi_amount"].fillna(0)
    )

    d["total_monthly_expenses"] = total_expenses
    d["disposable_income"] = d["monthly_salary"] - total_expenses
    d["dti_ratio"] = (d["current_emi_amount"] / income).clip(0, 5)
    d["expense_to_income_ratio"] = (total_expenses / income).clip(0, 10)
    d["emi_to_income_ratio"] = (d["current_emi_amount"] / income).clip(0, 5)
    d["requested_to_income_ratio"] = (d["requested_amount"] / income).clip(0, 100)
    d["bank_balance_to_income"] = (d["bank_balance"] / income).clip(0, 100)
    d["emergency_fund_months"] = (d["emergency_fund"] / income).clip(0, 120)
    d["affordable_emi_estimate"] = (d["disposable_income"] * 0.40).clip(lower=0)

    credit_penalty = np.clip((700 - d["credit_score"]) / 200, 0, 3)
    d["financial_risk_score"] = (
        0.40 * d["dti_ratio"].clip(0, 1)
        + 0.25 * d["expense_to_income_ratio"].clip(0, 2)
        + 0.20 * credit_penalty
        + 0.15 * (1 / (1 + d["emergency_fund_months"]))
    )

    # Exact one-hot columns used by the exported model.
    one_hot_groups = {
        "gender": ["F", "FEMALE", "Female", "M", "MALE", "Male", "female", "male", "nan"],
        "marital_status": ["Married", "Single", "nan"],
        "education": ["Graduate", "High School", "Post Graduate", "Professional", "nan"],
        "employment_type": ["Government", "Private", "Self-employed", "nan"],
        "company_type": ["Large Indian", "MNC", "Mid-size", "Small", "Startup", "nan"],
        "house_type": ["Family", "Own", "Rented", "nan"],
        "existing_loans": ["No", "Yes", "nan"],
        "emi_scenario": ["E-commerce Shopping EMI", "Education EMI", "Home Appliances EMI", "Personal Loan EMI", "Vehicle EMI", "nan"],
    }

    # Add selected category indicator and zero-fill all expected one-hot fields.
    for group, values in one_hot_groups.items():
        value = str(x[group])
        for v in values:
            d[f"{group}_{v}"] = 1 if value == v else 0

    # Credit score band indicators.
    score = float(x["credit_score"])
    if score <= 580:
        band = "Poor"
    elif score <= 670:
        band = "Fair"
    elif score <= 740:
        band = "Good"
    else:
        band = "Excellent"
    for b in ["Excellent", "Fair", "Good", "Poor", "nan"]:
        d[f"credit_score_band_{b}"] = 1 if band == b else 0

    # Keep exactly the columns the model was trained on, in the correct order.
    for col in FEATURE_COLUMNS:
        if col not in d.columns:
            d[col] = 0
    return d[FEATURE_COLUMNS].astype(float)


def build_engineered_raw(x: dict) -> pd.DataFrame:
    """Build the engineered-but-not-one-hot dataframe used by the Colab pipelines."""
    d = pd.DataFrame([x])
    income = d["monthly_salary"].clip(lower=1)
    total_expenses = (
        d["monthly_rent"].fillna(0) + d["school_fees"].fillna(0)
        + d["college_fees"].fillna(0) + d["travel_expenses"].fillna(0)
        + d["groceries_utilities"].fillna(0) + d["other_monthly_expenses"].fillna(0)
        + d["current_emi_amount"].fillna(0)
    )
    d["total_monthly_expenses"] = total_expenses
    d["disposable_income"] = d["monthly_salary"] - total_expenses
    d["dti_ratio"] = (d["current_emi_amount"] / income).clip(0, 5)
    d["expense_to_income_ratio"] = (total_expenses / income).clip(0, 10)
    d["emi_to_income_ratio"] = (d["current_emi_amount"] / income).clip(0, 5)
    d["requested_to_income_ratio"] = (d["requested_amount"] / income).clip(0, 100)
    d["bank_balance_to_income"] = (d["bank_balance"] / income).clip(0, 100)
    d["emergency_fund_months"] = (d["emergency_fund"] / income).clip(0, 120)
    d["affordable_emi_estimate"] = (d["disposable_income"] * 0.40).clip(lower=0)
    credit_penalty = np.clip((700 - d["credit_score"]) / 200, 0, 3)
    d["financial_risk_score"] = (
        0.40 * d["dti_ratio"].clip(0, 1)
        + 0.25 * d["expense_to_income_ratio"].clip(0, 2)
        + 0.20 * credit_penalty
        + 0.15 * (1 / (1 + d["emergency_fund_months"]))
    )
    d["credit_score_band"] = pd.cut(
        d["credit_score"], bins=[0, 580, 670, 740, 850],
        labels=["Poor", "Fair", "Good", "Excellent"], include_lowest=True
    ).astype("string")
    return d

def predict_with_model(model, raw: dict, model_name: str):
    """Support both the exported raw XGBoost models and sklearn Pipelines."""
    if hasattr(model, "named_steps"):
        engineered = build_engineered_raw(raw)
        expected = list(getattr(model, "feature_names_in_", engineered.columns))
        for col in expected:
            if col not in engineered.columns:
                engineered[col] = np.nan
        engineered = engineered[expected]
        return model.predict(engineered), model.predict_proba(engineered) if hasattr(model, "predict_proba") else None

    features = build_features(raw)
    pred = model.predict(features)
    proba = model.predict_proba(features) if hasattr(model, "predict_proba") else None
    return pred, proba


with st.form("emi_form"):
    st.subheader("Applicant Information")
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", 21, 80, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
        employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
        years_of_employment = st.number_input("Years of Employment", 0.0, 50.0, 5.0, step=0.5)

    with c2:
        monthly_salary = st.number_input("Monthly Salary (₹)", 1000.0, 5000000.0, 60000.0, step=1000.0)
        company_type = st.selectbox("Company Type", ["MNC", "Large Indian", "Mid-size", "Small", "Startup"])
        house_type = st.selectbox("House Type", ["Own", "Rented", "Family"])
        monthly_rent = st.number_input("Monthly Rent (₹)", 0.0, 500000.0, 12000.0, step=500.0)
        family_size = st.number_input("Family Size", 1, 15, 4)
        dependents = st.number_input("Dependents", 0, 14, 1)

    with c3:
        credit_score = st.number_input("Credit Score", 300, 850, 700)
        bank_balance = st.number_input("Bank Balance (₹)", 0.0, 10000000.0, 150000.0, step=5000.0)
        emergency_fund = st.number_input("Emergency Fund (₹)", 0.0, 10000000.0, 100000.0, step=5000.0)
        existing_loans = st.selectbox("Existing Loans", ["No", "Yes"])
        current_emi_amount = st.number_input("Current EMI (₹)", 0.0, 1000000.0, 8000.0, step=500.0)
        emi_scenario = st.selectbox("EMI Scenario", ["Personal Loan EMI", "Education EMI", "Vehicle EMI", "Home Appliances EMI", "E-commerce Shopping EMI"])

    st.subheader("Loan & Monthly Expenses")
    c4, c5, c6 = st.columns(3)
    with c4:
        requested_amount = st.number_input("Requested Loan Amount (₹)", 10000.0, 10000000.0, 500000.0, step=10000.0)
        requested_tenure = st.selectbox("Requested Tenure (months)", [12, 18, 24, 36, 48, 60, 72])
        school_fees = st.number_input("School Fees (₹/month)", 0.0, 200000.0, 3000.0, step=500.0)
    with c5:
        college_fees = st.number_input("College Fees (₹/month)", 0.0, 300000.0, 2000.0, step=500.0)
        travel_expenses = st.number_input("Travel Expenses (₹/month)", 0.0, 200000.0, 4000.0, step=500.0)
        groceries_utilities = st.number_input("Groceries & Utilities (₹/month)", 0.0, 300000.0, 10000.0, step=500.0)
    with c6:
        other_monthly_expenses = st.number_input("Other Expenses (₹/month)", 0.0, 300000.0, 5000.0, step=500.0)

    submitted = st.form_submit_button("🔍 Assess EMI Eligibility", use_container_width=True)

if submitted:
    raw = {
        "age": age, "gender": gender, "marital_status": marital_status,
        "education": education, "monthly_salary": monthly_salary,
        "employment_type": employment_type, "years_of_employment": years_of_employment,
        "company_type": company_type, "house_type": house_type,
        "monthly_rent": monthly_rent, "family_size": family_size,
        "dependents": dependents, "school_fees": school_fees,
        "college_fees": college_fees, "travel_expenses": travel_expenses,
        "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses,
        "existing_loans": existing_loans, "current_emi_amount": current_emi_amount,
        "credit_score": credit_score, "bank_balance": bank_balance,
        "emergency_fund": emergency_fund, "emi_scenario": emi_scenario,
        "requested_amount": requested_amount, "requested_tenure": requested_tenure,
    }

    class_pred, class_proba = predict_with_model(CLASS_MODEL, raw, "classification")
    reg_pred, _ = predict_with_model(REG_MODEL, raw, "regression")

    # The stored classification models use 1 = Eligible and 0 = Not Eligible.
    eligible = int(class_pred[0])
    probability = float(class_proba[0, 1]) if class_proba is not None else None
    max_emi = float(reg_pred[0])

    st.divider()
    a, b, c = st.columns(3)
    with a:
        st.metric("EMI Eligibility", "Eligible" if eligible else "Not Eligible")
    with b:
        st.metric("Maximum Monthly EMI", f"₹{max_emi:,.0f}")
    with c:
        st.metric("Eligibility Probability", f"{probability:.1%}" if probability is not None else "N/A")

    if eligible:
        st.success("The model predicts that this applicant is eligible under the trained model.")
    else:
        st.warning("The model predicts that this applicant is not eligible under the trained model.")

    st.info("This is an educational ML demonstration and should not be used as a real lending decision without appropriate validation, governance, fairness testing, and regulatory review.")
