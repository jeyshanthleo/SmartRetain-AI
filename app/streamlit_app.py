import streamlit as st
import pandas as pd
import json

from predictor import predict_churn
from nova_agent import generate_retention_strategy


st.set_page_config(page_title="SmartRetain AI", page_icon="📉", layout="centered")

st.title("📉 SmartRetain AI")
st.caption("Churn Prediction + AI Retention Strategy (Amazon Nova)")


# -----------------------------
# Helper: validate required fields for your pipeline
# -----------------------------
REQUIRED_KEYS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]


def validate_customer_info(customer_info: dict):
    missing = [k for k in REQUIRED_KEYS if k not in customer_info]
    return missing


def normalize_strategy(strategy):
    """
    Nova may return:
    - dict
    - JSON string
    - plain text
    This function always returns a dict with a safe shape.
    """
    # If it's already a dict, keep it
    if isinstance(strategy, dict):
        return strategy

    # If it's a string, try JSON parse
    if isinstance(strategy, str):
        try:
            parsed = json.loads(strategy)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # Plain string fallback
        return {
            "title": "AI Retention Strategy",
            "churn_reasons": [],
            "recommended_strategy": [],
            "expected_impact": "",
            "customer_message": {
                "subject": "AI Strategy",
                "body": strategy
            }
        }

    # Any other type fallback
    return {
        "title": "AI Retention Strategy",
        "churn_reasons": [],
        "recommended_strategy": [],
        "expected_impact": "",
        "customer_message": {
            "subject": "AI Strategy",
            "body": str(strategy)
        }
    }


# -----------------------------
# UI Form
# -----------------------------
st.header("1️⃣ Enter Customer Details")

name = st.text_input("Customer name", value="John")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=5)

with col2:
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
    internet = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes"])

col3, col4 = st.columns(2)

with col3:
    tech_support = st.selectbox("Tech Support", ["No", "Yes"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])

with col4:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=1.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=350.0, step=1.0)

st.divider()

run_btn = st.button("🚀 Predict & Generate Strategy")


if run_btn:
    customer_info = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,

        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,

        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,

        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,

        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    # Validate keys
    missing = validate_customer_info(customer_info)
    if missing:
        st.error(f"Error: columns are missing: {missing}")
        st.stop()

    st.header("2️⃣ Prediction Results")

    # predict_churn returns dict => read keys properly
    pred = predict_churn(customer_info)
    churn_prob = float(pred["churn_probability"])
    risk_level = pred["risk_level"]

    st.success(f"✅ Churn Probability: {churn_prob:.3f}")
    st.warning(f"⚠️ Risk Level: {risk_level}")

    st.header("3️⃣ AI Retention Strategy")

    with st.spinner("Generating strategy from Amazon Nova..."):
        raw_strategy = generate_retention_strategy(customer_info, risk_level)

    # ✅ NEW: normalize output (dict / JSON string / plain text)
    strategy = normalize_strategy(raw_strategy)

    # ✅ Display cleanly
    st.subheader("📌 Recommended Offer")
    st.success(strategy.get("title", "Retention Plan"))

    st.subheader("📊 Why Customer May Churn")
    reasons = strategy.get("churn_reasons", [])
    if isinstance(reasons, list) and reasons:
        for r in reasons:
            st.write(f"• {r}")
    else:
        st.write("• (No reasons provided)")

    st.subheader("🎯 Recommended Actions")
    actions = strategy.get("recommended_strategy", [])
    if isinstance(actions, list) and actions:
        for a in actions:
            st.write(f"• {a}")
    else:
        st.write("• (No actions provided)")

    st.subheader("💰 Estimated Business Impact")
    st.info(strategy.get("expected_impact", "—"))

    # ✅ Customer message: handle dict or string
    st.subheader("📩 Customer Message")
    customer_msg = strategy.get("customer_message", {})
    if isinstance(customer_msg, dict):
        subject = customer_msg.get("subject", "AI Strategy")
        body = customer_msg.get("body", "").replace("\\n", "\n")
        st.markdown(f"**Subject:** {subject}")
        st.write(body if body else "—")
    else:
        st.write(str(customer_msg))

    # Save as text for CSV
    strategy_text = json.dumps(strategy, ensure_ascii=False)

    st.header("4️⃣ Download / Save")

    results_df = pd.DataFrame([{
        "name": name,
        "churn_probability": churn_prob,
        "risk_level": risk_level,
        "strategy": strategy_text
    }])

    st.download_button(
        "⬇️ Download Result as CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="smartretain_result.csv",
        mime="text/csv"
    )