import mlflow
import mlflow.sklearn
import mlflow.xgboost


def log_experiment(
    model_name,
    model,
    metrics,
    params=None
):
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Bank Customer Churn")

    with mlflow.start_run():

        mlflow.log_param("model", model_name)

        if params:
            mlflow.log_params(params)

        for key, value in metrics.items():
            if isinstance(value, float):
                mlflow.log_metric(key, value)

        if model_name == "xgboost":

            mlflow.xgboost.log_model(
                xgb_model=model.named_steps["classifier"],
                name="model"
            )

        else:

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model"
            )

    print("\nExperiment logged to MLflow.")