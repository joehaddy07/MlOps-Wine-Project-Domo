
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
    log_artifacts,
    register_model,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "data/raw/wine.csv"

TEST_SIZE = 0.2

RANDOM_STATE = 42

N_ESTIMATORS = 100


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

    print("\n[1/8] Configuring MLflow...")

    configure_mlflow()

    print("MLflow Tracking URI:")
    print(mlflow.get_tracking_uri())

    print("MLflow Experiment:")
    print("Wine_Quality_Predictions")


    # ========================================================
    # 2. LOAD DATASET
    # ========================================================

    print("\n[2/8] Loading dataset...")

    print(f"Loading dataset from: {os.path.abspath(DATASET_PATH)}")

    wine_dataset = load_dataset()

    print(
        f"Dataset loaded successfully: "
        f"{wine_dataset.shape}"
    )

    print(f"Dataset shape: {wine_dataset.shape}")


    # ========================================================
    # 3. START MLFLOW RUN
    # ========================================================

    print("\n[3/8] Starting MLflow run...")

    with mlflow.start_run(
        run_name="Wine_Quality_RandomForest"
    ):

        print("MLflow run started.")

        print(
            f"Run ID: "
            f"{mlflow.active_run().info.run_id}"
        )


        # ====================================================
        # 4. LOG DATASET INFORMATION
        # ====================================================

        print("\n[4/8] Logging dataset information...")

        log_dataset_information(
            wine_dataset
        )

        print("Dataset information logged.")


        # ====================================================
        # 5. PREPROCESS DATA
        # ====================================================

        print("\n[5/8] Preprocessing dataset...")

        x_train, x_test, y_train, y_test = preprocess_data(
            wine_dataset,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        print("Preprocessing completed.")


        # ====================================================
        # 6. TRAIN MODEL
        # ====================================================

        print("\n[6/8] Training model...")

        model = train_model(
            X_train,
            Y_train
        )

        print("Model training completed.")


        # ====================================================
        # 7. EVALUATE MODEL
        # ====================================================

        print("\n[7/8] Evaluating model...")

        results = evaluate_model(
            model,
            X_test,
            Y_test,
            wine_dataset
        )

        accuracy = results["accuracy"]

        print(
            f"Model accuracy: {accuracy:.4f}"
        )


        # ====================================================
        # LOG TRAINING PARAMETERS
        # ====================================================

        print("\nLogging training parameters...")

        log_training_parameters(
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            n_estimators=N_ESTIMATORS
        )


        # ====================================================
        # LOG METRICS
        # ====================================================

        print("Logging model metrics...")

        log_metrics(
            accuracy=accuracy
        )


        # ====================================================
        # LOG ARTIFACTS
        # ====================================================

        print("Logging artifacts...")

        if "classification_report_file" in results:
            if os.path.exists(
                results["classification_report_file"]
            ):
                mlflow.log_artifact(
                    results["classification_report_file"],
                    artifact_path="reports"
                )

        if "confusion_matrix_file" in results:
            if os.path.exists(
                results["confusion_matrix_file"]
            ):
                mlflow.log_artifact(
                    results["confusion_matrix_file"],
                    artifact_path="plots"
                )

        if "plots" in results:

            for plot in results["plots"]:

                if os.path.exists(plot):

                    mlflow.log_artifact(
                        plot,
                        artifact_path="plots"
                    )


        # ====================================================
        # REGISTER MODEL
        # ====================================================

        print("\nRegistering model with MLflow...")

        model_info = register_model(
            model
        )

        print("\nModel registered successfully.")

        print(
            f"Model URI: "
            f"{model_info.model_uri}"
        )


        # ====================================================
        # DISPLAY RUN INFORMATION
        # ====================================================

        run_id = mlflow.active_run().info.run_id

        print("\n" + "=" * 60)
        print("MLFLOW RUN COMPLETED")
        print("=" * 60)

        print(f"Experiment: Wine_Quality_Predictions")
        print(f"Run ID: {run_id}")
        print(f"Model: WineQualityModel")
        print(f"Accuracy: {accuracy:.4f}")

        print("=" * 60)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
