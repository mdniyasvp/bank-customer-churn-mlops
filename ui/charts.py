import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


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
                    {
                        "range": [0, 30],"color": "#DCFCE7"
                    },
                    {
                        "range": [30, 70],"color": "#FEF3C7"
                    },
                    {
                        "range": [70, 100],"color": "#FECACA"
                    },
                ],
            },
        )
    )

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig

def plot_feature_importance(pipeline, top_n=10):
    classifier = pipeline.named_steps["classifier"]
    feature_names = (
        pipeline.named_steps[
            "preprocessor"
        ].get_feature_names_out()
    )

    importance = classifier.feature_importances_
    importance_df = (
        pd.DataFrame(
            {
                "Feature": feature_names,"Importance": importance,
            }
        )
        .sort_values("Importance",ascending=False,
        )
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1],)
    ax.set_title("Top Features Influencing Prediction")
    ax.set_xlabel( "Importance")
    plt.tight_layout()
    st.pyplot(fig)