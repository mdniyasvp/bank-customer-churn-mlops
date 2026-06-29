import time
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import GridSearchCV

from src.models.train import prepare_data
from src.models.evaluate import evaluate_model
from src.pipelines.model_pipeline import build_pipeline

from src.config.config import (
    MODEL_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)


MODEL_NAME = "random_forest"


def main():

    print("=" * 60)
    print("Hyperparameter Tuning")
    print("=" * 60)

    # ------------------------
    # Load Data
    # ------------------------

    X_train, X_test, y_train, y_test = prepare_data()

    # ------------------------
    # Build Pipeline
    # ------------------------

    pipeline = build_pipeline(MODEL_NAME)

    # ------------------------
    # Parameter Grid
    # ------------------------

    param_grid = {

        "classifier__n_estimators": [100, 200],

        "classifier__max_depth": [
            10,
            20,
            None,
        ],

        "classifier__min_samples_split": [
            2,
            5,
        ],

        "classifier__min_samples_leaf": [
            1,
            2,
        ],

    }

    # ------------------------
    # Grid Search
    # ------------------------

    grid_search = GridSearchCV(

        estimator=pipeline,

        param_grid=param_grid,

        scoring="f1",

        cv=5,

        n_jobs=-1,

        verbose=2,

    )

    start_time = time.time()

    grid_search.fit(
        X_train,
        y_train,
    )

    training_time = time.time() - start_time

    print("\nBest Parameters")
    print(grid_search.best_params_)

    print("\nBest Cross Validation F1")
    print(f"{grid_search.best_score_:.4f}")

    # ------------------------
    # Evaluate
    # ------------------------

    metrics = evaluate_model(
        grid_search.best_estimator_,
        X_test,
        y_test,
    )

    # ------------------------
    # Save Model
    # ------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = MODEL_DIR / "random_forest_tuned.joblib"

    joblib.dump(
        grid_search.best_estimator_,
        model_path,
    )

    print(f"\nModel saved to:\n{model_path}")

    # ------------------------
    # MLflow
    # ------------------------

    mlflow.set_experiment("Bank Customer Churn")

    with mlflow.start_run(run_name="Random Forest Tuned"):

        mlflow.log_param(
            "model",
            "random_forest_tuned",
        )

        mlflow.log_params(
            grid_search.best_params_
        )

        mlflow.log_param(
            "test_size",
            TEST_SIZE,
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE,
        )

        mlflow.log_metric(
            "training_time",
            training_time,
        )

        for key, value in metrics.items():

            if isinstance(value, float):

                mlflow.log_metric(
                    key,
                    value,
                )

        mlflow.sklearn.log_model(
            sk_model=grid_search.best_estimator_,
            name="model",
        )

    print("\nExperiment logged to MLflow.")

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)

    print(f"Training Time : {training_time:.2f} seconds")

    for key, value in metrics.items():

        if isinstance(value, float):

            print(f"{key:<12}: {value:.4f}")


if __name__ == "__main__":
    main()