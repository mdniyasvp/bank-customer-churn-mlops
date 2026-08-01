import os
import joblib
import streamlit as st
from ui.api_client import predict_customer
from src.config.config import MODEL_DIR, MODEL_NAME

PREDICTION_MODE = st.secrets.get(
    "PREDICTION_MODE",
    os.getenv("PREDICTION_MODE", "api"),
)
PREDICTION_MODE = os.getenv("PREDICTION_MODE", "api")


def predict(api_url, payload):

    if PREDICTION_MODE == "api":
        return predict_customer(api_url, payload)

    pipeline = joblib.load(
        MODEL_DIR / f"{MODEL_NAME}.joblib"
    )

    import pandas as pd

    df = pd.DataFrame([payload])

    prediction = pipeline.predict(df)[0]

    probability = pipeline.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "label": "Exited" if prediction else "Stayed",
        "probability": float(probability),
    }