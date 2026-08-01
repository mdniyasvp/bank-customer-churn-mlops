from pathlib import Path

import joblib
import streamlit as st

from src.config.config import MODEL_DIR, MODEL_NAME

from ui.api_client import predict_customer
from ui.dashboard import show_dashboard
from ui.forms import render_customer_form
from ui.hero import render_hero
from ui.history import show_prediction_history
from ui.recomendation import get_recommendation
from ui.sidebar import render_sidebar

# Page Configuration

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="centered",
)

# Load CSS

css_file = Path("assets/style.css")

if css_file.exists():

    with open(css_file) as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True,
        )

# API Configuration

import os

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/predict"
)
# Load Trained Pipeline

MODEL_PATH = MODEL_DIR / f"{MODEL_NAME}.joblib"

@st.cache_resource
def load_pipeline():

    return joblib.load(MODEL_PATH)


pipeline = load_pipeline()

# Sidebar

render_sidebar()

# Hero Banner

render_hero()

# Project Information

st.markdown(
    '<div class="section-title">📊 Project Information</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    with st.container(border=True):

        st.caption("Algorithm")

        st.markdown("### XGBoost")

with col2:

    with st.container(border=True):

        st.caption("Backend")

        st.markdown("### FastAPI")

with col3:

    with st.container(border=True):

        st.caption("Deployment")

        st.markdown("### Local API")

with col4:

    with st.container(border=True):

        st.caption("Status")

        st.success("🟢 Online")

# =====================================================
# Customer Form
# =====================================================

submitted, payload, customer = render_customer_form()

# =====================================================
# Prediction Workflow
# =====================================================
if "history" not in st.session_state:
    st.session_state.history = []
if submitted:

    # -----------------------------------------
    # Call FastAPI
    # -----------------------------------------

    result = predict_customer(
        API_URL,
        payload,
    )

    # -----------------------------------------
    # Continue only if prediction succeeded
    # -----------------------------------------

    if result is not None:

        label = result["label"]

        probability = result["probability"]

        recommendation = get_recommendation(
            probability
        )

        # -----------------------------------------
        # Save Prediction History
        # -----------------------------------------

        st.session_state.history.append(
            {
                "Age": customer["age"],
                "Country": customer["geography"],
                "Gender": customer["gender"],
                "Prediction": label,
                "Probability": round(
                    probability * 100,
                    2,
                ),
            }
        )

        # -----------------------------------------
        # Prediction Dashboard
        # -----------------------------------------

        show_dashboard(
            label=label,
            probability=probability,
            recommendation=recommendation,
            customer=customer,
            pipeline=pipeline,
        )

        # =====================================================
        # Prediction History
        # =====================================================

        show_prediction_history()

        # =====================================================
        # Footer
        # =====================================================

        st.divider()

        st.caption(
        "Built with ❤️ using Streamlit, FastAPI and XGBoost."
)