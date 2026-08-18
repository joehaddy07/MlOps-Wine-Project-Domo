# ============================================================
# MLFLOW TRACKING
# ============================================================

import mlflow
import mlflow.sklearn


# ============================================================
# CONFIGURATION
# ============================================================

MLFLOW_TRACKING_URI = "http://localhost:5000"

EXPERIMENT_NAME = "Wine_Quality_Predictions"

MODEL_NAME = "WineQualityModel"


# ============================================================
# CONFIGURE MLFLOW
# ============================================================

def configure_mlflow():
    """
    Configure MLflow to use the MLflow Tracking Server.
    """

    print()
    print("=" * 60)
    print("MLFLOW CONFIGURATION")
    print("=" * 60)

    print(
        f"MLflow Tracking URI: "
        f"{MLFLOW_TRACKING_URI}"
    )

    # --------------------------------------------------------
    # Set tracking server
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    # --------------------------------------------------------
    # Verify tracking URI
    # --------------------------------------------------------

    active_uri = mlflow.get_tracking_uri()

    print(
        f"Active MLflow Tracking URI: "
        f"{active_uri}"
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if not active_uri.startswith("http://") and not active_uri.startswith("https://"):

        raise RuntimeError(
            "MLflow is NOT configured to use the "
            "MLflow Tracking Server. "
            f"Current URI: {active_uri}"
        )

    # --------------------------------------------------------
    # Configure experiment
    # --------------------------------------------------------

    print(
        f"MLflow Experiment: "
        f"{EXPERIMENT_NAME}"
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print(
        "MLflow experiment configured successfully."
    )

    print("=" * 60)


# ============================================================
# LOG DATASET INFORMATION
# ============================================================

def log_dataset_information(wine_dataset):
    """
    Log dataset information to MLflow.
    """

    mlflow.log_param(
        "dataset_name",
        "wine.csv"
    )

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


# ============================================================
# LOG TRAINING PARAMETERS
# ============================================================

def log_training_parameters(
    test_size,
    random_state,
    n_estimators
):
    """
    Log training parameters.
    """

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

    mlflow.log_param(
        "model_type",
        "RandomForestClassifier"
    )


# ============================================================
# LOG METRICS
# ============================================================

def log_metrics(accuracy):
    """
    Log model metrics.
    """

    mlflow.log_metric(
        "accuracy",
        float(accuracy)
    )


# ============================================================
# REGISTER MODEL
# ============================================================

def register_model(model):
    """
    Log and register the trained model.
    """

    print()
    print("Logging model to MLflow...")

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="wine_quality_model",
        registered_model_name=MODEL_NAME
    )

    print(
        "Model logged and registered successfully."
    )

    print(
        f"Model URI: "
        f"{model_info.model_uri}"
    )

    return model_info