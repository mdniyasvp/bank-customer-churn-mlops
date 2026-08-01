import streamlit as st

def render_hero():
    st.markdown(
        """
<div class="hero">
<h1>🏦 Bank Customer Churn Predictor</h1>

<p>
AI-Powered Customer Retention Dashboard
</p>

</div>
""",
        unsafe_allow_html=True,
    )