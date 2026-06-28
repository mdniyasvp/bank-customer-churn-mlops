import pandas as pd

from src.config.config import DROP_COLUMNS, TARGET_COLUMN

def preprocess_data(df: pd.DataFrame):

    # Drop unnecessary columns
    df = df.drop(columns=DROP_COLUMNS)

    # Convert binary columns
    df["HasCrCard"] = df["HasCrCard"].astype(int)
    df["IsActiveMember"] = df["IsActiveMember"].astype(int)

    # Features
    X = df.drop(columns=[TARGET_COLUMN])

    # Target
    y = df[TARGET_COLUMN]

    return X, y