from predictor import predict_churn
from nova_agent import generate_retention_strategy


def run_pipeline():
    customer_info = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70,
        "TotalCharges": 350
    }

    # 1) Predict churn (works whether predict_churn returns dict OR tuple)
    pred = predict_churn(customer_info)

    if isinstance(pred, dict):
        proba = pred.get("churn_probability")
        risk_level = pred.get("risk_level")
    else:
        proba, risk_level = pred

    # Convert proba to float safely
    if isinstance(proba, (list, tuple)):
        proba = proba[0]

    try:
        proba = float(proba)
    except Exception:
        proba = float(str(proba).strip().split()[0])

    # 2) Get strategy from Nova
    strategy = generate_retention_strategy(customer_info, risk_level)

    # 3) Print output
    print("\n✅ Churn Probability:", round(proba, 3))
    print("✅ Risk Level:", risk_level)
    print("\n===== RETENTION STRATEGY OUTPUT =====\n")
    print(strategy)


if __name__ == "__main__":
    run_pipeline()