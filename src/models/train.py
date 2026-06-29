import time
import joblib

from sklearn.model_selection import train_test_split

from src.config.config import (
    RANDOM_STATE,
    TEST_SIZE,
    MODEL_DIR,
    MODEL_NAME,
)

from src.data.loader import load_data
from src.data.preprocessing import preprocess_data
from src.pipelines.model_pipeline import build_pipeline
from src.models.evaluate import evaluate_model
from src.utils.mlflow_logger import log_experiment

def prepare_data():
    """
    Load data, preprocess it, and split into train/test sets.
    """

    df = load_data()

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def main():

    print("=" * 60)
    print(f"Training Model : {MODEL_NAME}")
    print("=" * 60)

    X_train, X_test, y_train, y_test = prepare_data()

    pipeline = build_pipeline(MODEL_NAME)

    start_time = time.time()

    pipeline.fit(X_train, y_train)

    training_time = time.time() - start_time

    metrics = evaluate_model(
        pipeline,
        X_test,
        y_test,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = MODEL_DIR / f"{MODEL_NAME}.joblib"

    joblib.dump(
        pipeline,
        model_path,
    )

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)

    print(f"Training Time : {training_time:.2f} seconds")

    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric:<12}: {value:.4f}")

    print(f"\nModel saved to:\n{model_path}")

    log_experiment(
        model_name=MODEL_NAME,
        model=pipeline,
        metrics=metrics,
        params={
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
        }
    )
if __name__ == "__main__":
    main()
