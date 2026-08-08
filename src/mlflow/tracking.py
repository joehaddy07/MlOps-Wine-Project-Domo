# ============================================================
# IMPORT LIBRARIES
# ============================================================

import mlflow
import mlflow.sklearn

# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

TRACKING_URI = "http://localhost:5000"

EXPERIMENT_NAME = "Wine_Quality_Predictions"

REGISTERED_MODEL_NAME = "WineQualityModel"

# ============================================================
# CONFIGURE MLFLOW
# ============================================================

def configure_mlflow():
    """
    Configure the MLflow Tracking Server.
    """

    mlflow.set_tracking_uri(TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)


# ============================================================
# LOG DATASET INFORMATION
# ============================================================

def log_dataset_information(wine_dataset):

    mlflow.log_param(
        "dataset_rows",
        wine_dataset.shape[0]
    )

    mlflow.log_param(
        "dataset_columns",
        wine_dataset.shape[1]
    )


# ============================================================
# LOG TRAINING PARAMETERS
# ============================================================

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

    mlflow.log_param(
        "model_type",
        "RandomForestClassifier"
    )


# ============================================================
# LOG MODEL METRICS
# ============================================================

def log_metrics(accuracy):

    mlflow.log_metric(
        "wine_accuracy",
        accuracy
    )


# ============================================================
# LOG ARTIFACTS
# ============================================================

def log_artifacts(results):

    mlflow.log_artifact(
        "wine_dataset_logged.csv",
        artifact_path="datasets"
    )

    mlflow.log_artifact(
        results["classification_report_file"],
        artifact_path="reports"
    )

    mlflow.log_artifact(
        results["confusion_matrix_file"],
        artifact_path="plots"
    )

    for plot in results["plots"]:

        mlflow.log_artifact(
            plot,
            artifact_path="plots"
        )


# ============================================================
# REGISTER MODEL
# ============================================================

def register_model(model):

    model_info = mlflow.sklearn.log_model(

        sk_model=model,

        artifact_path="wine_model",

        registered_model_name=REGISTERED_MODEL_NAME
    )

    print("\nRegistered Model URI")

    print("---------------------------")

    print(model_info.model_uri)

    return model_info