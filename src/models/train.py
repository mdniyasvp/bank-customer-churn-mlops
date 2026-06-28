from sklearn.model_selection import train_test_split
from src.data.loader import load_data
from src.data.preprocessing import preprocess_data
from src.config.config import RANDOM_STATE, TEST_SIZE
from src.models.evaluate import evaluate_model

def prepare_data():
    """
    Load data, preprocess it, and split into train/test sets.
    """

    # Load raw data
    df = load_data()

    # Preprocess
    X, y = preprocess_data(df)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


from src.pipelines.model_pipeline import model_pipeline

...

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = prepare_data()

    model_pipeline.fit(X_train, y_train)

    print("Pipeline training completed successfully!")

    print("Training Features :", X_train.shape)
    print("Testing Features  :", X_test.shape)
    print("Training Target   :", y_train.shape)
    print("Testing Target    :", y_test.shape)
    
print("\nTraining Target Distribution")
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting Target Distribution")
print(y_test.value_counts(normalize=True) * 100)

import joblib
from pathlib import Path

from src.config.config import MODEL_DIR, MODEL_PATH
model_pipeline.fit(X_train, y_train)

metrics = evaluate_model(
    model_pipeline,
    X_test,
    y_test
)

print("\nTraining Complete")
print("-" * 40)

for metric, value in metrics.items():
    if isinstance(value, float):
        print(f"{metric:<12}: {value:.4f}")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    model_pipeline,
    MODEL_PATH
)

print(f"\nModel saved to:\n{MODEL_PATH}")