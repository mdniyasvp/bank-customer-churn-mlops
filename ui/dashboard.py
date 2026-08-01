import streamlit as st
from ui.charts import (create_gauge, plot_feature_importance,)


def show_dashboard(label, probability, recommendation, customer, pipeline,):
    st.divider()
    st.markdown("## 📊 Prediction Dashboard")
    left, right = st.columns(2)

    # Prediction Card

    with left:
        with st.container(border=True):
            if label == "Stayed":
                st.success("🟢 CUSTOMER WILL STAY")
            else:
                st.error("🔴 CUSTOMER WILL EXIT")
            st.metric("Prediction", label,)

            st.metric("Churn Probability", f"{probability*100:.2f}%")

            if probability < 0.30:
                st.success("LOW RISK")

            elif probability < 0.70:
                st.warning("MEDIUM RISK")

            else:
                st.error("HIGH RISK")

    # Gauge

    with right:
        st.plotly_chart(
            create_gauge(probability),
            use_container_width=True,
        )

    # Recommendation

    st.markdown("## 💼 Business Recommendation")
    with st.container(border=True):
        for action in recommendation["action"]:
            st.write(f"✅ {action}")

    # Customer Summary

    st.markdown("## 👤 Customer Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Age:** {customer['age']}")
        st.write(f"**Gender:** {customer['gender']}")
        st.write(f"**Country:** {customer['geography']}")
        st.write(f"**Credit Score:** {customer['credit_score']}")

    with col2:
        st.write(f"**Balance:** ₹{customer['balance']:,.2f}")
        st.write(f"**Products:** {customer['num_products']}")
        st.write(f"**Active Member:** {customer['active_member']}")
        st.write(f"**Estimated Salary:** ₹{customer['estimated_salary']:,.2f}")

    # Feature Importance

    st.divider()
    st.subheader("📊 Model Feature Importance")
    plot_feature_importance(pipeline)