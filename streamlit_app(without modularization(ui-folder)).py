from pathlib import Path

import joblib
from src.config.config import MODEL_DIR, MODEL_NAME
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

if "history" not in st.session_state:
    st.session_state.history = []

def plot_feature_importance(model, top_n=10):

    classifier = model.named_steps["classifier"]
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importance = classifier.feature_importances_
    importance_df = (
        pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importance,
            }
        )
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(top_n)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        importance_df["Feature"][::-1],
        importance_df["Importance"][::-1],
    )
    ax.set_title("Top Features Influencing Prediction")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    st.pyplot(fig)

# load css

css_file = Path("assets/style.css")
if css_file.exists():
   with open(css_file) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True,
        )


API_URL = "http://127.0.0.1:8000/predict"

MODEL_PATH = MODEL_DIR / f"{MODEL_NAME}.joblib"

@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


def create_gauge(probability):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Churn Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2563EB"},
                "steps": [
                    {"range": [0, 30], "color": "#DCFCE7"},
                    {"range": [30, 70], "color": "#FEF3C7"},
                    {"range": [70, 100], "color": "#FECACA"},
                ],
            },
        )
    )

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig

def get_recommendation(probability):

    if probability < 0.30:

        return {
            "risk": "🟢 LOW",
            "action": [
                "Customer appears loyal.",
                "No immediate action required.",
                "Continue normal engagement."
            ]
        }

    elif probability < 0.70:

        return {
            "risk": "🟡 MEDIUM",
            "action": [
                "Monitor customer activity.",
                "Offer personalized promotions.",
                "Encourage product usage."
            ]
        }

    else:

        return {
            "risk": "🔴 HIGH",
            "action": [
                "Contact customer immediately.",
                "Offer a retention incentive.",
                "Assign a relationship manager."
            ]
        }

pipeline = load_pipeline()

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="centered"
)
# Sidebar

with st.sidebar:

    st.title("🏦 Bank Churn")
    st.markdown("---")
    st.subheader("Project")

    st.write("**Model:** XGBoost")
    st.write("**Backend:** FastAPI")
    st.write("**Deployment:** Railway")
    st.write("**Frontend:** Streamlit")

    st.markdown("---")

    st.info(
        "This application predicts whether a customer is likely "
        "to leave the bank based on historical customer data."
    )

# Main Title

st.markdown(
    """
<div class="hero">
<h1>🏦 Bank Customer Churn Predictor</h1>
<p>
AI-Powered Customer Retention Dashboard
</p>
</div>
""",
    unsafe_allow_html=True
)

# Project Information

st.markdown(
    '<div class="section-title">📊 Project Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    with st.container(border=True):
        st.caption("Algorithm")
        st.subheader("XGBoost")

with col2:
    with st.container(border=True):
        st.caption("Backend")
        st.subheader("FastAPI")

with col3:
    with st.container(border=True):
        st.caption("Deployment")
        st.subheader("Local API")

with col4:
    with st.container(border=True):
        st.caption("Status")
        st.success("🟢 Online")


# Customer Information

st.divider()

st.markdown(
    '<div class="section-title">📝 Customer Information</div>',
    unsafe_allow_html=True
)
with st.form("prediction_form"):

    # -----------------------------------
    # Customer Profile
    # -----------------------------------

    st.markdown("### 👤 Customer Profile")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

    st.divider()

    # Banking Details

    st.markdown("### 💳 Banking Details")
    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=650
        )

    with col2:
        tenure = st.number_input(
            "Tenure",
            min_value=0,
            max_value=10,
            value=5
        )

    col1, col2 = st.columns(2)

    with col1:
        balance = st.number_input(
            "Balance",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

    with col2:
        num_products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=4,
            value=2
        )

    st.divider()

    # Account Status

    st.markdown("### 📈 Account Status")
    col1, col2 = st.columns(2)
    with col1:
        has_card = st.selectbox(
            "Has Credit Card",
            ["Yes", "No"]
        )

    with col2:
        active_member = st.selectbox(
            "Is Active Member",
            ["Yes", "No"]
        )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    st.divider()

    submitted = st.form_submit_button(
        "🔮 Predict Customer Churn",
        use_container_width=True
    )

# Prediction

if submitted:

    payload = {
        "CreditScore": int(credit_score),
        "Geography": geography,
        "Gender": gender,
        "Age": int(age),
        "Tenure": int(tenure),
        "Balance": float(balance),
        "NumOfProducts": int(num_products),
        "HasCrCard": 1 if has_card == "Yes" else 0,
        "IsActiveMember": 1 if active_member == "Yes" else 0,
        "EstimatedSalary": float(estimated_salary),
    }

    try:

        with st.spinner("🔄 Predicting customer churn..."):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=15
            )

            if response.status_code == 422:
                st.error("❌ Invalid input values.")
                st.json(response.json())
                st.stop()
            response.raise_for_status()
            result = response.json()
        label = result["label"]
        probability = result["probability"]

        recommendation = get_recommendation(probability)

        st.session_state.history.append(
            {
                "Age": age,
                "Country": geography,
                "Gender": gender,
                "Prediction": label,
                "Probability": round(probability * 100, 2),
            }
        )


        label = result["label"]
        probability = result["probability"]

        recommendation = get_recommendation(probability)
        st.session_state.history.append(
            {
                "Age": age,
                "Country": geography,
                "Gender": gender,
                "Prediction": label,
                "Probability": round(probability * 100, 2),
            }
        )

        # Prediction Dashboard

        st.divider()
        st.markdown("## 📊 Prediction Dashboard")
        left, right = st.columns([1, 1])

        # ---------------- LEFT CARD ----------------

        with left:
            with st.container(border=True):
                if label == "Stayed":
                    st.success("🟢 CUSTOMER WILL STAY")
                else:
                    st.error("🔴 CUSTOMER WILL EXIT")
                st.metric(
                    "Prediction",
                    label
                )
                st.metric(
                    "Churn Probability",
                    f"{probability*100:.2f}%"
                )
                if probability < 0.30:
                    st.success("LOW RISK")
                elif probability < 0.70:
                    st.warning("MEDIUM RISK")
                else:
                    st.error("HIGH RISK")

        # ---------------- RIGHT CARD ----------------
        with right:
            st.plotly_chart(
                create_gauge(probability),
                use_container_width=True
            )

        # Business Recommendation

        st.markdown("## 💼 Business Recommendation")
        with st.container(border=True):
            for action in recommendation["action"]:
                st.write(f"✅ {action}")

        # Customer Summary

        st.markdown("## 👤 Customer Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Age:** {age}")
            st.write(f"**Gender:** {gender}")
            st.write(f"**Country:** {geography}")
            st.write(f"**Credit Score:** {credit_score}")
        with col2:
            st.write(f"**Balance:** ₹{balance:,.2f}")
            st.write(f"**Products:** {num_products}")
            st.write(f"**Active Member:** {'Yes' if active_member == 'Yes' else 'No'}")
            st.write(f"**Estimated Salary:** ₹{estimated_salary:,.2f}")\
            
        # Feature Importance

            st.divider()
            st.subheader("📊 Model Feature Importance")
            plot_feature_importance(pipeline)

        
    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Unable to connect to the FastAPI server.\n\n"
            "Make sure FastAPI is running."
        )
    except requests.exceptions.Timeout:
        st.error("⏳ Request timed out.")
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")

if st.session_state.history:
    st.divider()
    st.subheader("📜 Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
    )
    csv = history_df.to_csv(index=False).encode("utf-8")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download CSV",
            csv,
            "prediction_history.csv",
            "text/csv",
            use_container_width=True,
        )

    with col2:
        if st.button(
            "🗑 Clear History",
            use_container_width=True,
        ):
            st.session_state.history = []
            st.rerun()

st.divider()
st.subheader("📊 Model Feature Importance")
plot_feature_importance(pipeline)

st.divider()
st.caption(
    "Built with using Streamlit, FastAPI and XGBoost."
)