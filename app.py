import streamlit as st
import pandas as pd
import pickle

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="TeleChurn AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# LOAD MODELS
# ======================================================

with open("models/logistic_regression_model.pkl", "rb") as f:
    logistic_model = pickle.load(f)

with open("models/decision_tree_model.pkl", "rb") as f:
    decision_tree_model = pickle.load(f)

# ======================================================
# LOAD ENCODERS
# ======================================================

with open("models/contract_encoder.pkl", "rb") as f:
    contract_encoder = pickle.load(f)

with open("models/internet_encoder.pkl", "rb") as f:
    internet_encoder = pickle.load(f)

with open("models/security_encoder.pkl", "rb") as f:
    security_encoder = pickle.load(f)

with open("models/support_encoder.pkl", "rb") as f:
    support_encoder = pickle.load(f)

with open("models/payment_encoder.pkl", "rb") as f:
    payment_encoder = pickle.load(f)

# ======================================================
# LOAD SCALER
# ======================================================

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown("""
<style>

.main{
    background-color:#F5F7FA;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#0F172A;
}

.subtitle{
    text-align:center;
    font-size:18px;
    color:#475569;
    margin-bottom:30px;
}

.section{
    # background:white;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 0px 15px rgba(0,0,0,0.08);
    margin-bottom:20px;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================

st.markdown(
    '<h1 class="title">📡 TeleChurn AI</h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Customer Churn Prediction using Machine Learning & DevOps</p>',
    unsafe_allow_html=True
)

st.divider()

# ======================================================
# SIDEBAR
# ======================================================

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/3062/3062634.png",
        width=100
    )

    st.title("TeleChurn AI")

    st.write(
        "Predict whether a telecom customer is likely to leave the service."
    )

    st.divider()

    model_choice = st.radio(

        "Choose Machine Learning Model",

        [
            "Logistic Regression",
            "Decision Tree"
        ]

    )

    st.divider()

    if model_choice == "Logistic Regression":

        st.metric(
            "Model Accuracy",
            "77.33%"
        )

    else:

        st.metric(
            "Model Accuracy",
            "72.35%"
        )

    st.divider()

    st.success("Project Status: Ready")

    st.caption(
        "Built using Python, Scikit-Learn, Streamlit, Docker and GitHub Actions."
    )

st.markdown("""
<div class="section">

<h3>📌 Project Overview</h3>

This application predicts whether a telecom customer is likely to churn based on customer subscription details, billing information, and service usage.

The prediction is generated using two supervised machine learning models:

<ul>

<li>Logistic Regression</li>

<li>Decision Tree Classifier</li>

</ul>

Select your preferred model from the sidebar, enter customer details, and click <b>Predict Churn</b>.

</div>
""", unsafe_allow_html=True)


left, right = st.columns(2)

with left:

    st.subheader("👤 Customer Information")

    contract = st.selectbox(
        "Contract Type",
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
        "Technical Support",
        support_encoder.classes_
    )

    payment_method = st.selectbox(
        "Payment Method",
        payment_encoder.classes_
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


with right:


    st.subheader("💳 Billing Information")

    tenure = st.slider(

        "Customer Tenure (Months)",

        0,

        72,

        12

    )

    monthly_charges = st.number_input(

        "Monthly Charges",

        min_value=0.0,

        value=70.0

    )

    total_charges = st.number_input(

        "Total Charges",

        min_value=0.0,

        value=1200.0

    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

st.divider()

predict = st.button(

    "🚀 Predict Customer Churn",

    use_container_width=True

)

# ======================================================
# PREDICTION
# ======================================================


if predict:

    # -------------------------------
    # Encode categorical values
    # -------------------------------

    contract_value = contract_encoder.transform([contract])[0]

    internet_value = internet_encoder.transform([internet_service])[0]

    security_value = security_encoder.transform([online_security])[0]

    support_value = support_encoder.transform([tech_support])[0]

    payment_value = payment_encoder.transform([payment_method])[0]

    # -------------------------------
    # Create DataFrame
    # -------------------------------

    input_df = pd.DataFrame({

        "Contract":[contract_value],

        "InternetService":[internet_value],

        "OnlineSecurity":[security_value],

        "TechSupport":[support_value],

        "PaymentMethod":[payment_value],

        "tenure":[tenure],

        "MonthlyCharges":[monthly_charges],

        "TotalCharges":[total_charges]

    })

    # -------------------------------
    # Scale Numerical Features
    # -------------------------------

    input_df[

        [

            "tenure",

            "MonthlyCharges",

            "TotalCharges"

        ]

    ] = scaler.transform(

        input_df[

            [

                "tenure",

                "MonthlyCharges",

                "TotalCharges"

            ]

        ]

    )
    st.write("Encoded Input")

    st.dataframe(input_df)

    # -------------------------------
    # Select Model
    # -------------------------------

    if model_choice == "Logistic Regression":

        model = logistic_model

        model_accuracy = 77.33

    else:

        model = decision_tree_model

        model_accuracy = 72.35

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0]

    confidence = probability.max() * 100

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 0:

        st.success(

            "Customer is likely to Stay."

        )

        st.progress(

            confidence / 100

        )

        st.metric(

            "Confidence",

            f"{confidence:.2f}%"

        )

        st.info(

            """
            Recommendation

            Customer appears satisfied with current services.

            Continue providing quality service and maintain engagement.
            """

        )

    else:

        st.error(

            "Customer is likely to Churn."

        )

        st.progress(

            confidence / 100

        )

        st.metric(

            "Confidence",

            f"{confidence:.2f}%"

        )

        st.warning(

            """
            Recommendation

            Customer has a high probability of leaving.

            Consider offering promotional discounts, loyalty rewards, or personalized support.
            """

        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(

            "Model Used",

            model_choice

        )

    with col2:

        st.metric(

            "Model Accuracy",

            f"{model_accuracy}%"

        )

    st.subheader("Prediction Probabilities")

    st.write(probability)
    
    probability_df = pd.DataFrame({

        "Outcome":[

            "Stay",

            "Churn"

        ],

        "Probability":[

            probability[0] * 100,

            probability[1] * 100

        ]

    })

    st.dataframe(

        probability_df,

        use_container_width=True,

        hide_index=True

    )

    st.bar_chart(

        probability_df.set_index(

            "Outcome"

        )

    )

    st.divider()

    st.subheader("Business Insight")

    if prediction == 1:

        st.write(

            """
            This customer exhibits characteristics commonly associated with churn.

            Retention strategies such as discounts, contract upgrades, or enhanced customer support may help reduce churn risk.
            """

        )

    else:

        st.write(

            """
            This customer demonstrates a low risk of churn.

            Maintaining service quality and customer satisfaction should help preserve long-term loyalty.
            """

        )

else:

    st.info(

        "Enter customer details and click the button above to predict churn."

    )

    st.divider()

    st.subheader("📘 About TeleChurn AI")

st.write("""
**TeleChurn AI** is a machine learning application developed to predict customer churn in the telecommunications industry.

The application compares two supervised learning algorithms:

- Logistic Regression
- Decision Tree Classifier

Users can choose either model and compare their prediction results and overall accuracy.
""")

st.subheader("🛠 Technology Stack")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
**Machine Learning**

- Scikit-Learn
- Logistic Regression
- Decision Tree
""")

with col2:
    st.info("""
**Data Processing**

- Pandas
- NumPy
- StandardScaler
""")

with col3:
    st.info("""
**DevOps**

- Git
- Docker
- GitHub Actions
- Streamlit
""")
    

st.subheader("✨ Application Features")

st.success("""
✔ Customer Churn Prediction

✔ Two Machine Learning Models

✔ Automatic Data Preprocessing

✔ Confidence Score

✔ Probability Distribution

✔ Interactive Dashboard

✔ Dockerized Application

✔ CI Pipeline using GitHub Actions
""")

st.subheader("📊 Model Performance")

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree"
    ],
    "Accuracy": [
        "77.33%",
        "72.35%"
    ],
    "Remarks": [
        "Higher overall accuracy",
        "Easy to interpret"
    ]
})

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)

st.divider()

st.markdown(
"""
<div style='text-align:center; color:gray;'>

<h4>📡 TeleChurn AI</h4>

Customer Churn Prediction using Machine Learning & DevOps

Developed using

Python • Streamlit • Scikit-Learn • Docker • GitHub Actions

<hr>

© All Rights reserved to Hasnain Ahmed Ansari and team since 2026

</div>
""",
unsafe_allow_html=True
)

st.divider()

st.caption("Version 1.0")

st.caption("Built for Introduction to Data Science & DevOps")

st.caption("TeleChurn AI")