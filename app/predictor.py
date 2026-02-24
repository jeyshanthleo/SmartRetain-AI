import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "models/churn_model.joblib"
model = joblib.load(MODEL_PATH)


# These are the ORIGINAL Telco columns your pipeline was trained on
RAW_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]


def _prepare_raw_df(data: dict) -> pd.DataFrame:
    # Keep only expected columns (prevents extra keys breaking the pipeline)
    clean = {k: data.get(k, None) for k in RAW_COLUMNS}
    df = pd.DataFrame([clean])

    # Ensure numeric columns are numeric (Telco dataset often has TotalCharges as string/blank)
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Replace empty strings with NaN (important for imputers)
    df = df.replace(r"^\s*$", np.nan, regex=True)

    return df


def predict_churn(data: dict) -> dict:
    X = _prepare_raw_df(data)

    proba = model.predict_proba(X)[0][1]
    risk = "High" if proba >= 0.6 else ("Medium" if proba >= 0.3 else "Low")

    return {
        "churn_probability": float(proba),
        "risk_level": risk
    }