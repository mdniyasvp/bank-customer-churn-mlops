import streamlit as st


def render_sidebar():

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
            "This application predicts whether a customer "
            "is likely to leave the bank based on "
            "historical customer data."
        )