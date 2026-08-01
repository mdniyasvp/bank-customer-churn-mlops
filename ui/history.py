import pandas as pd
import streamlit as st


def show_prediction_history():
    if "history" not in st.session_state:
        st.session_state.history = []
    if st.session_state.history:
        st.divider()
        st.subheader("📜 Prediction History")
        history_df = pd.DataFrame(
            st.session_state.history
        )
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