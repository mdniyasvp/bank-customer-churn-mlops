import streamlit as st
def render_customer_form():
    st.divider()
    st.markdown(
        '<div class="section-title">📝 Customer Information</div>',
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        st.markdown("### 👤 Customer Profile")
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35,)

        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"],)

        geography = st.selectbox("Geography", ["France", "Germany", "Spain"],)

        st.divider()
        st.markdown("### 💳 Banking Details")
        col1, col2 = st.columns(2)

        with col1:
            credit_score = st.number_input("Credit Score",min_value=300,max_value=900,value=650,)

        with col2:
            tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5, )
        col1, col2 = st.columns(2)

        with col1:
            balance = st.number_input(
                "Balance",
                min_value=0.0,
                value=50000.0,
                step=1000.0,
            )

        with col2:

            num_products = st.number_input(
                "Number of Products",
                min_value=1,
                max_value=4,
                value=2,
            )

        st.divider()
        st.markdown("### 📈 Account Status")
        col1, col2 = st.columns(2)
        with col1:
            has_card = st.selectbox(
                "Has Credit Card",
                ["Yes", "No"],
            )

        with col2:
            active_member = st.selectbox(
                "Is Active Member",
                ["Yes", "No"],
            )

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
        )

        st.divider()
        submitted = st.form_submit_button(
            "🔮 Predict Customer Churn",
            use_container_width=True,
        )

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

    customer = {
        "age": age,
        "gender": gender,
        "geography": geography,
        "credit_score": credit_score,
        "balance": balance,
        "num_products": num_products,
        "active_member": active_member,
        "estimated_salary": estimated_salary,
    }

    return submitted, payload, customer