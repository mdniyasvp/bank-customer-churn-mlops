from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

MODELS = {

    "logistic_regression": LogisticRegression(
        random_state=42,
        max_iter=1000
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

}