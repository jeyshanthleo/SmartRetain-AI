from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "telco_churn.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"

def main():
    df = pd.read_csv(DATA_PATH)

    # Clean TotalCharges (in this dataset it can contain spaces)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Target
    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn"])

    # Drop ID
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced")

    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    print("✅ ROC-AUC:", roc_auc_score(y_test, proba))
    print(classification_report(y_test, preds))

    joblib.dump(clf, MODEL_PATH)
    print(f"✅ Saved model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()