print("\n[8/8] Logging artifacts...")

classification_report_file = results.get(
    "classification_report_file"
)

if classification_report_file:

    print(
        f"Classification report: "
        f"{classification_report_file}"
    )

    print(
        f"Exists: "
        f"{os.path.exists(classification_report_file)}"
    )

    if os.path.exists(classification_report_file):

        print(
            "Uploading classification report "
            "to MLflow..."
        )

        mlflow.log_artifact(
            classification_report_file,
            artifact_path="reports"
        )

        print(
            "Classification report uploaded successfully."
        )


confusion_matrix_file = results.get(
    "confusion_matrix_file"
)

if confusion_matrix_file:

    print(
        f"Confusion matrix: "
        f"{confusion_matrix_file}"
    )

    if os.path.exists(confusion_matrix_file):

        print(
            "Uploading confusion matrix..."
        )

        mlflow.log_artifact(
            confusion_matrix_file,
            artifact_path="plots"
        )

        print(
            "Confusion matrix uploaded successfully."
        )


plots = results.get(
    "plots",
    []
)

for plot in plots:

    print(
        f"Checking plot: {plot}"
    )

    if os.path.exists(plot):

        print(
            f"Uploading {plot}..."
        )

        mlflow.log_artifact(
            plot,
            artifact_path="plots"
        )

        print(
            f"Successfully uploaded {plot}"
        )

    else:

        print(
            f"WARNING: File does not exist: {plot}"
        )

print(
    "All artifacts processed successfully."
)