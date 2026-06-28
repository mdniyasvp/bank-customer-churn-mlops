from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.registry import MODELS

# -------------------------
# Select Model
# -------------------------

MODEL_NAME = "logistic_regression"

# -------------------------
# Feature Lists
# -------------------------

categorical_features = [
    "Geography",
    "Gender"
]

numerical_features = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]

# -------------------------
# Preprocessor
# -------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)

# -------------------------
# Model Pipeline
# -------------------------

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", MODELS[MODEL_NAME])
    ]
)