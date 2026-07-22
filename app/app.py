from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

model = joblib.load(MODEL_PATH)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a telecom customer is likely to churn.")

st.divider()

# -----------------------------
# Mapping Dictionaries
# -----------------------------

yes_no = {"No": 0, "Yes": 1}

gender_map = {
    "Female": 0,
    "Male": 1
}

internet_map = {
    "DSL": 0,
    "Fiber Optic": 1,
    "No Internet": 2
}

contract_map = {
    "Month-to-Month": 0,
    "One Year": 1,
    "Two Year": 2
}

payment_map = {
    "Electronic Check": 0,
    "Mailed Check": 1,
    "Bank Transfer": 2,
    "Credit Card": 3
}

# -----------------------------
# Layout
# -----------------------------

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        list(gender_map.keys())
    )

    senior = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    tenure = st.slider(
        "Tenure (Months)",
        0,
        72,
        12
    )

    phone = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    multiple = st.selectbox(
        "Multiple Lines",
        ["No", "Yes"]
    )

    internet = st.selectbox(
        "Internet Service",
        list(internet_map.keys())
    )

    security = st.selectbox(
        "Online Security",
        ["No", "Yes"]
    )

    backup = st.selectbox(
        "Online Backup",
        ["No", "Yes"]
    )

with col2:

    device = st.selectbox(
        "Device Protection",
        ["No", "Yes"]
    )

    support = st.selectbox(
        "Tech Support",
        ["No", "Yes"]
    )

    tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes"]
    )

    movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes"]
    )

    contract = st.selectbox(
        "Contract",
        list(contract_map.keys())
    )

    paper = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )

    payment = st.selectbox(
        "Payment Method",
        list(payment_map.keys())
    )

    monthly = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    total = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=1500.0
    )

st.divider()

# -----------------------------
# Prediction
# -----------------------------

if st.button("🔍 Predict Churn", use_container_width=True):

    data = pd.DataFrame([{

        "gender": gender_map[gender],
        "SeniorCitizen": yes_no[senior],
        "Partner": yes_no[partner],
        "Dependents": yes_no[dependents],
        "tenure": tenure,
        "PhoneService": yes_no[phone],
        "MultipleLines": yes_no[multiple],
        "InternetService": internet_map[internet],
        "OnlineSecurity": yes_no[security],
        "OnlineBackup": yes_no[backup],
        "DeviceProtection": yes_no[device],
        "TechSupport": yes_no[support],
        "StreamingTV": yes_no[tv],
        "StreamingMovies": yes_no[movies],
        "Contract": contract_map[contract],
        "PaperlessBilling": yes_no[paper],
        "PaymentMethod": payment_map[payment],
        "MonthlyCharges": monthly,
        "TotalCharges": total

    }])

    prediction = model.predict(data)[0]

    st.subheader("Prediction")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is likely to stay.")

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(data)[0][1]

        st.progress(float(probability))

        st.metric(
            "Churn Probability",
            f"{probability*100:.2f}%"
        )