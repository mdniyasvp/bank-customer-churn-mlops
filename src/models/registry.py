from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


MODELS = {

    "logistic_regression": LogisticRegression(
        random_state=42,
        max_iter=1000
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "xgboost": XGBClassifier(

        n_estimators=300,

        learning_rate=0.1,

        max_depth=6,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=42,

        eval_metric="logloss"

    )

}