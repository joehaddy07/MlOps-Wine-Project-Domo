import os
import mlflow
import mlflow.sklearn

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)

MLFLOW_EXPERIMENT_NAME = "Wine_Quality_Predictions"

MODEL_NAME = "WineQualityModel"

def configure_mlflow():

    # --------------------------------------------------------
    # Configure MLflow Tracking Server
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    # --------------------------------------------------------
    # Create or select experiment
    # --------------------------------------------------------

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT_NAME
    )

    print(
        f"MLflow Tracking URI: "
        f"{mlflow.get_tracking_uri()}"
    )

    print(
        f"MLflow Experiment: "
        f"{MLFLOW_EXPERIMENT_NAME}"
    )


def log_dataset_information(wine_dataset):

    mlflow.log_param(
        "dataset_rows",
        wine_dataset.shape[0]
    )

    mlflow.log_param(
        "dataset_columns",
        wine_dataset.shape[1]
    )

    mlflow.log_param(
        "target_column",
        "quality"
    )

def log_training_parameters(
    test_size,
    random_state,
    n_estimators
):

    mlflow.log_param(
        "test_size",
        test_size
    )

    mlflow.log_param(
        "random_state",
        random_state
    )

    mlflow.log_param(
        "n_estimators",
        n_estimators
    )

def log_metrics(accuracy):

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

def register_model(model):

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="wine_quality_model",
        registered_model_name=MODEL_NAME
    )

    return model_info