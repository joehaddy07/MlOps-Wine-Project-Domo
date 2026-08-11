
# ============================================================
# WINE QUALITY MLOPS PIPELINE
# ============================================================
#
# This file is the main entry point for the machine learning
# pipeline.
#
# Pipeline:
#
# 1. Load dataset
# 2. Preprocess dataset
# 3. Train Random Forest model
# 4. Evaluate model
# 5. Configure MLflow
# 6. Log parameters
# 7. Log metrics
# 8. Log artifacts
# 9. Register trained model
#
# This file is intentionally kept as the orchestrator.
# Individual responsibilities are handled by separate modules.
# ============================================================


# ============================================================
# IMPORT LIBRARIES
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
    register_model
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "data/raw/wine.csv"

TEST_SIZE = 0.20

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
    # STEP 1
    # LOAD DATASET
    # ========================================================

    print("\n[1/8] Loading dataset...")

    wine_dataset = load_dataset(DATASET_PATH)

    print("Dataset loaded successfully.")

    print(f"Rows: {wine_dataset.shape[0]}")
    print(f"Columns: {wine_dataset.shape[1]}")


    # ========================================================
    # STEP 2
    # PREPROCESS DATA
    # ========================================================

    print("\n[2/8] Preprocessing dataset...")

    X_train, X_test, Y_train, Y_test = preprocess_data(
        wine_dataset,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print("Data preprocessing completed.")


    # ========================================================
    # STEP 3
    # CONFIGURE MLFLOW
    # ========================================================

    print("\n[3/8] Configuring MLflow...")

    configure_mlflow()

    print("MLflow tracking URI:")
    print(mlflow.get_tracking_uri())

    print("MLflow experiment:")
    print("Wine_Quality_Predictions")


    # ========================================================
    # STEP 4
    # START MLFLOW RUN
    # ========================================================

    print("\n[4/8] Starting MLflow run...")

    with mlflow.start_run(
        run_name="Wine_Quality_RandomForest"
    ):

        print("MLflow run started.")

        # ----------------------------------------------------
        # Log dataset information
        # ----------------------------------------------------

        log_dataset_information(
            wine_dataset
        )

        # ----------------------------------------------------
        # Log training parameters
        # ----------------------------------------------------

        log_training_parameters(
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            n_estimators=N_ESTIMATORS
        )


        # ====================================================
        # STEP 5
        # TRAIN MODEL
        # ====================================================

        print("\n[5/8] Training Random Forest model...")

        model = train_model(
            X_train,
            Y_train
        )

        print("Model training completed.")


        # ====================================================
        # STEP 6
        # EVALUATE MODEL
        # ====================================================

        print("\n[6/8] Evaluating model...")

        results = evaluate_model(
            model,
            X_test,
            Y_test
        )

        print("Model evaluation completed.")

        print(
            f"Accuracy: {results['accuracy']}"
        )


        # ----------------------------------------------------
        # Log metrics
        # ----------------------------------------------------

        log_metrics(
            results["accuracy"]
        )


        # ====================================================
        # STEP 7
        # LOG ARTIFACTS
        # ====================================================

        print("\n[7/8] Logging artifacts...")

        log_artifacts(
            results
        )

        print("Artifacts logged successfully.")


        # ====================================================
        # STEP 8
        # REGISTER MODEL
        # ====================================================

        print("\n[8/8] Registering model...")

        model_info = register_model(
            model
        )

        print("\nModel successfully registered.")

        print(
            f"Model URI: {model_info.model_uri}"
        )


        # ====================================================
        # DISPLAY RUN INFORMATION
        # ====================================================

        run = mlflow.active_run()

        if run:

            print("\nMLflow Run Information")
            print("-----------------------------")

            print(
                f"Run ID: {run.info.run_id}"
            )

            print(
                f"Experiment ID: {run.info.experiment_id}"
            )

            print(
                f"Tracking URI: {mlflow.get_tracking_uri()}"
            )


    # ========================================================
    # PIPELINE COMPLETED
    # ========================================================

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ============================================================
# PYTHON ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

