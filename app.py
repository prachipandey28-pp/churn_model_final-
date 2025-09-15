# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("churn_model.pkl")

# Load datasets
df = pd.read_excel("churn.xlsx")
banks = pd.read_csv("Bank_Comparison.csv")

# Sidebar Menu
st.sidebar.title("Navigation")
menu = ["Dashboard", "Predict Churn", "View Customers", "Bank Comparison", "About"]
choice = st.sidebar.selectbox("Go to", menu)

# Dashboard
if choice == "Dashboard":
    st.title("📊 Bank Churn Prediction Dashboard")
    st.metric("Total Customers", len(df))
    st.metric("Churned Customers", df["Exited"].sum())
    st.metric("Churn Rate", f"{df['Exited'].mean() * 100:.2f}%")

# Predict Churn
elif choice == "Predict Churn":
    st.title("🔮 Predict Customer Churn & Loan Eligibility")

    # Form Inputs
    age = st.number_input("Age", 18, 100, 30)
    balance = st.number_input("Balance", 0, 1000000, 50000)
    salary = st.number_input("Estimated Salary", 0, 500000, 50000)
    credit_score = st.number_input("Credit Score", 300, 900, 650)
    tenure = st.number_input("Tenure (Years)", 0, 20, 5)
    active = st.selectbox("Active Member", ["Yes", "No"])
    products = st.number_input("Num of Products", 1, 4, 1)

    if st.button("Predict"):
        # Input dataframe
        input_data = pd.DataFrame([[
            credit_score, age, tenure, balance, products,
            1 if active == "Yes" else 0, salary
        ]], columns=["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "IsActiveMember", "EstimatedSalary"])

        # Prediction
        pred = model.predict(input_data)[0]

        if pred == 1:
            st.error("❌ This customer is likely to CHURN!")
        else:
            st.success("✅ This customer will STAY with the bank.")

            # Loan eligibility
            if credit_score > 700 and salary > 20000 and balance > 10000:
                st.info("💰 Eligible for Loan")
                st.write("### 🏦 Available Bank Offers")
                st.table(banks)
            else:
                st.warning("⚠️ Not Eligible for Loan")

# View Customers
elif choice == "View Customers":
    st.title("👥 Customer Database")
    st.dataframe(df.head(50))

# Bank Comparison
elif choice == "Bank Comparison":
    st.title("🏦 Bank Loan Comparison")
    st.table(banks)

# About
elif choice == "About":
    st.title("ℹ️ About This App")
    st.write("""
    **AI-Powered Bank Churn Prediction System**  
    - Predicts whether a customer will churn  
    - Checks loan eligibility  
    - Shows customer database  
    - Compares bank loans  
    """)