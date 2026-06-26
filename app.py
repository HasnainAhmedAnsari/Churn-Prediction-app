import streamlit as st
import pandas as pd
import pickle

# ==========================
# Load Models
# ==========================

logistic_model = pickle.load(
    open("models/logistic_regression_model.pkl", "rb")
)

decision_tree_model = pickle.load(
    open("models/decision_tree_model.pkl", "rb")
)

# ==========================
# Load Encoders
# ==========================

contract_encoder = pickle.load(
    open("models/contract_encoder.pkl", "rb")
)

internet_encoder = pickle.load(
    open("models/internet_encoder.pkl", "rb")
)

security_encoder = pickle.load(
    open("models/security_encoder.pkl", "rb")
)

support_encoder = pickle.load(
    open("models/support_encoder.pkl", "rb")
)

payment_encoder = pickle.load(
    open("models/payment_encoder.pkl", "rb")
)

# ==========================
# Load Scaler
# ==========================

scaler = pickle.load(
    open("models/scaler.pkl", "rb")
)

# ==========================
# Page Title
# ==========================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊"
)

st.title("📊 Customer Churn Prediction")

st.write(
    "Predict whether a customer is likely to churn using Machine Learning."
)

# ==========================
# Model Selection
# ==========================

model_choice = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree"
    ]
)

# ==========================
# Inputs
# ==========================

contract = st.selectbox(
    "Contract",
    contract_encoder.classes_
)

internet_service = st.selectbox(
    "Internet Service",
    internet_encoder.classes_
)

online_security = st.selectbox(
    "Online Security",
    security_encoder.classes_
)

tech_support = st.selectbox(
    "Tech Support",
    support_encoder.classes_
)

payment_method = st.selectbox(
    "Payment Method",
    payment_encoder.classes_
)

tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=100,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=500.0
)

# ==========================
# Predict Button
# ==========================

if st.button("Predict Churn"):

    contract_encoded = contract_encoder.transform(
        [contract]
    )[0]

    internet_encoded = internet_encoder.transform(
        [internet_service]
    )[0]

    security_encoded = security_encoder.transform(
        [online_security]
    )[0]

    support_encoded = support_encoder.transform(
        [tech_support]
    )[0]

    payment_encoded = payment_encoder.transform(
        [payment_method]
    )[0]

    input_data = pd.DataFrame(
        [[
            contract_encoded,
            internet_encoded,
            security_encoded,
            support_encoded,
            payment_encoded,
            tenure,
            monthly_charges,
            total_charges
        ]],
        columns=[
            'Contract',
            'InternetService',
            'OnlineSecurity',
            'TechSupport',
            'PaymentMethod',
            'tenure',
            'MonthlyCharges',
            'TotalCharges'
        ]
    )

    numerical_cols = [
        'tenure',
        'MonthlyCharges',
        'TotalCharges'
    ]

    input_data[numerical_cols] = scaler.transform(
        input_data[numerical_cols]
    )

    if model_choice == "Logistic Regression":

        prediction = logistic_model.predict(
            input_data
        )[0]

        probability = logistic_model.predict_proba(
            input_data
        )[0][1]

        accuracy = "77.33%"

    else:

        prediction = decision_tree_model.predict(
            input_data
        )[0]

        probability = decision_tree_model.predict_proba(
            input_data
        )[0][1]

        accuracy = "72.35%"

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            "⚠ Customer is likely to Churn"
        )

    else:

        st.success(
            "✅ Customer is likely to Stay"
        )

    st.write(
        f"Prediction Confidence: {probability*100:.2f}%"
    )

    st.write(
        f"Model Accuracy: {accuracy}"
    )