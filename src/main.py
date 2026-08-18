# ============================================================
# WINE QUALITY MLOPS PIPELINE
# ============================================================

import os
import mlflow

from src.data.load_data import load_dataset
from src.features.preprocessing import preprocess_data
from src.models.train import train_model
from src.evaluation.evaluate import evaluate_model

from src.mlflow.tracking import (
    configure_mlflow,
    log_dataset_information,
    log_training_parameters,
    log_metrics,
    register_model,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "data/raw/wine.csv"

TEST_SIZE = 0.20

RANDOM_STATE = 42

N_ESTIMATORS = 100

EXPERIMENT_NAME = "Wine_Quality_Predictions"

RUN_NAME = "Wine_Quality_RandomForest"

MODEL_NAME = "WineQualityModel"


# ============================================================
# HELPER FUNCTION
# ============================================================

def log_artifact_safely(file_path, artifact_path):
    """
    Log an artifact through the MLflow Tracking Server.

    The GitHub Actions runner should NOT attempt to write
    directly to /mlflow.

    Instead, mlflow.log_artifact() sends the artifact to
    the configured MLflow Tracking Server.
    """

    print()
    print("-" * 60)
    print("MLFLOW ARTIFACT")
    print("-" * 60)

    print(f"Local file: {file_path}")
    print(f"Artifact folder: {artifact_path}")

    if not os.path.exists(file_path):
        print(
            f"WARNING: Artifact does not exist: {file_path}"
        )
        return False

    try:

        print(
            "Uploading artifact through MLflow Tracking Server..."
        )

        mlflow.log_artifact(
            file_path,
            artifact_path=artifact_path
        )

        print(
            f"Artifact uploaded successfully: {file_path}"
        )

        return True

    except Exception as error:

        print(
            "ERROR: MLflow artifact upload failed."
        )

        print(
            f"Exception type: {type(error).__name__}"
        )

        print(
            f"Exception message: {error}"
        )

        return False


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("WINE QUALITY MLOPS PIPELINE")
    print("=" * 60)

    # ========================================================
    # 1. CONFIGURE MLFLOW
    # ========================================================

    print()
    print("[1/8] Configuring MLflow...")

    configure_mlflow()

    print(
        f"MLflow Tracking URI: "
        f"{mlflow.get_tracking_uri()}"
    )

    print(
        f"MLflow Experiment: "
        f"{EXPERIMENT_NAME}"
    )

    # ========================================================
    # 2. LOAD DATASET
    # ========================================================

    print()
    print("[2/8] Loading dataset...")

    absolute_dataset_path = os.path.abspath(
        DATASET_PATH
    )

    print(
        f"Loading dataset from: "
        f"{absolute_dataset_path}"
    )

    wine_dataset = load_dataset(
        DATASET_PATH
    )

    print(
        f"Dataset loaded successfully: "
        f"{wine_dataset.shape}"
    )

    print(
        f"Dataset shape: "
        f"{wine_dataset.shape}"
    )

    # ========================================================
    # 3. START MLFLOW RUN
    # ========================================================

    print()
    print("[3/8] Starting MLflow run...")

    with mlflow.start_run(
        run_name=RUN_NAME
    ):

        run_id = (
            mlflow.active_run()
            .info
            .run_id
        )

        print(
            "MLflow run started."
        )

        print(
            f"Run ID: {run_id}"
        )

        # ====================================================
        # 4. LOG DATASET INFORMATION
        # ====================================================

        print()
        print(
            "[4/8] Logging dataset information..."
        )

        log_dataset_information(
            wine_dataset
        )

        print(
            "Dataset information logged."
        )

        # ====================================================
        # 5. PREPROCESS DATA
        # ====================================================

        print()
        print(
            "[5/8] Preprocessing dataset..."
        )

        x_train, x_test, y_train, y_test = (
            preprocess_data(
                wine_dataset
            )
        )

        print(
            "Preprocessing completed."
        )

        print(
            f"X_train shape: {x_train.shape}"
        )

        print(
            f"X_test shape: {x_test.shape}"
        )

        print(
            f"y_train shape: {y_train.shape}"
        )

        print(
            f"y_test shape: {y_test.shape}"
        )

        # ====================================================
        # 6. TRAIN MODEL
        # ====================================================

        print()
        print(
            "[6/8] Training model..."
        )

        print(
            "Starting Random Forest training..."
        )

        model = train_model(
            x_train,
            y_train
        )

        print(
            "Model training completed."
        )

        # ====================================================
        # 7. EVALUATE MODEL
        # ====================================================

        print()
        print(
            "[7/8] Evaluating model..."
        )

        results = evaluate_model(
            model,
            x_test,
            y_test,
            wine_dataset
        )

        accuracy = results[
            "accuracy"
        ]

        print()
        print(
            f"Model accuracy: "
            f"{accuracy:.4f}"
        )

        # ====================================================
        # LOG TRAINING PARAMETERS
        # ====================================================

        print()
        print(
            "Logging training parameters..."
        )

        log_training_parameters(
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            n_estimators=N_ESTIMATORS
        )

        print(
            "Training parameters logged."
        )

        # ====================================================
        # LOG METRICS
        # ====================================================

        print()
        print(
            "Logging model metrics..."
        )

        log_metrics(
            accuracy=accuracy
        )

        print(
            "Model metrics logged."
        )

        # ====================================================
        # 8. LOG ARTIFACTS
        # ====================================================

        print()
        print(
            "[8/8] Logging artifacts..."
        )

        print()
        print(
            "MLflow artifact URI:"
        )

        try:

            artifact_uri = (
                mlflow.get_artifact_uri()
            )

            print(
                artifact_uri
            )

        except Exception as error:

            print(
                f"Could not determine artifact URI: "
                f"{error}"
            )

        # ----------------------------------------------------
        # Classification Report
        # ----------------------------------------------------

        classification_report_file = (
            results.get(
                "classification_report_file"
            )
        )

        if classification_report_file:

            log_artifact_safely(
                classification_report_file,
                "reports"
            )

        # ----------------------------------------------------
        # Confusion Matrix
        # ----------------------------------------------------

        confusion_matrix_file = (
            results.get(
                "confusion_matrix_file"
            )
        )

        if confusion_matrix_file:

            log_artifact_safely(
                confusion_matrix_file,
                "plots"
            )

        # ----------------------------------------------------
        # Other Plots
        # ----------------------------------------------------

        plots = results.get(
            "plots",
            []
        )

        print()
        print(
            "Processing additional plots..."
        )

        for plot in plots:

            log_artifact_safely(
                plot,
                "plots"
            )

        print()
        print(
            "Artifact processing completed."
        )

        # ====================================================
        # REGISTER MODEL
        # ====================================================

        print()
        print(
            "Registering model with MLflow..."
        )

        try:

            model_info = register_model(
                model
            )

            print()
            print(
                "Model registered successfully."
            )

            print(
                f"Model URI: "
                f"{model_info.model_uri}"
            )

        except Exception as error:

            print()
            print(
                "ERROR: Model registration failed."
            )

            print(
                f"Exception type: "
                f"{type(error).__name__}"
            )

            print(
                f"Exception message: "
                f"{error}"
            )

            raise

        # ====================================================
        # DISPLAY RUN INFORMATION
        # ====================================================

        print()
        print("=" * 60)
        print(
            "MLFLOW RUN COMPLETED"
        )
        print("=" * 60)

        print(
            f"Experiment: "
            f"{EXPERIMENT_NAME}"
        )

        print(
            f"Run ID: "
            f"{run_id}"
        )

        print(
            f"Run Name: "
            f"{RUN_NAME}"
        )

        print(
            f"Model: "
            f"{MODEL_NAME}"
        )

        print(
            f"Accuracy: "
            f"{accuracy:.4f}"
        )

        print(
            f"Tracking URI: "
            f"{mlflow.get_tracking_uri()}"
        )

        try:

            print(
                f"Artifact URI: "
                f"{mlflow.get_artifact_uri()}"
            )

        except Exception:
            pass

        print("=" * 60)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()