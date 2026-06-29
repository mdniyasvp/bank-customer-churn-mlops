import time
import joblib

from sklearn.model_selection import train_test_split

from src.config.config import (
    RANDOM_STATE,
    TEST_SIZE,
    MODEL_DIR,
)
from src.utils.report import save_model_report
from src.data.loader import load_data
from src.data.preprocessing import preprocess_data
from src.models.registry import MODELS
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

    X_train, X_test, y_train, y_test = prepare_data()

    best_pipeline = None
    best_metrics = None
    best_model_name = None
    best_training_time = None

    best_f1 = -1

    print("\n" + "=" * 70)
    print("MODEL TRAINING & COMPARISON")
    print("=" * 70)

    results = []

    for model_name in MODELS.keys():

        print("\n" + "=" * 60)
        print(f"Training Model : {model_name}")
        print("=" * 60)

        pipeline = build_pipeline(model_name)

        start_time = time.time()

        pipeline.fit(
            X_train,
            y_train
        )

        training_time = time.time() - start_time

        metrics = evaluate_model(
            pipeline,
            X_test,
            y_test,
        )
        
        results.append({

        "Model": model_name,

        "Accuracy": metrics["accuracy"],

        "Precision": metrics["precision"],

        "Recall": metrics["recall"],

        "F1": metrics["f1"],

        "ROC-AUC": metrics["roc_auc"],

        "Training Time (s)": round(training_time, 2)

    })
        save_model_report(results)

        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)

        print(f"Training Time : {training_time:.2f} seconds")

        for metric, value in metrics.items():

            if isinstance(value, float):
                print(f"{metric:<12}: {value:.4f}")

        log_experiment(
            model_name=model_name,
            model=pipeline,
            metrics=metrics,
            params={
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
            }
        )

        if metrics["f1"] > best_f1:

            best_f1 = metrics["f1"]
            best_pipeline = pipeline
            best_metrics = metrics
            best_model_name = model_name
            best_training_time = training_time

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = MODEL_DIR / f"{best_model_name}.joblib"

    joblib.dump(
        best_pipeline,
        model_path
    )

    print("\n")
    print("=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(f"Model          : {best_model_name}")
    print(f"Training Time  : {best_training_time:.2f} seconds")

    for metric, value in best_metrics.items():

        if isinstance(value, float):
            print(f"{metric:<15}: {value:.4f}")

    print(f"\nBest model saved to:\n{model_path}")


if __name__ == "__main__":
    main()