import requests
import streamlit as st

def predict_customer(api_url: str, payload: dict):
    try:
        with st.spinner("🔄 Predicting customer churn..."):
            response = requests.post(api_url,json=payload,timeout=15,)

            if response.status_code == 422:
                st.error("❌ Invalid input values.")
                st.json(response.json())
                return None

            response.raise_for_status()
            return response.json()

    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Unable to connect to the FastAPI server.\n\n"
            "Make sure FastAPI is running."
        )

    except requests.exceptions.Timeout:
        st.error("⏳ Request timed out.")

    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")

    return None