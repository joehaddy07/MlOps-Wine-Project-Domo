# ============================================================
# IMPORT LIBRARIES
# ============================================================

# Visualization
import matplotlib
matplotlib.use("Agg")  # Prevent GUI errors when running on servers

import matplotlib.pyplot as plt
import seaborn as sns

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, X_test, Y_test, wine_dataset):
    """
    Evaluate the trained machine learning model.

    Parameters
    ----------
    model : Trained Machine Learning Model

    X_test : Testing Features

    Y_test : Testing Labels

    wine_dataset : Original Wine Dataset

    Returns
    -------
    accuracy : float

    report : str

    confusion_matrix_file : str

    classification_report_file : str

    plot_files : list
    """

    # ========================================================
    # MAKE PREDICTIONS
    # ========================================================

    predictions = model.predict(X_test)

    # ========================================================
    # CALCULATE MODEL ACCURACY
    # ========================================================

    accuracy = accuracy_score(
        Y_test,
        predictions
    )

    print("\nModel Accuracy")
    print("---------------------------")
    print(accuracy)

    # ========================================================
    # GENERATE CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(
        Y_test,
        predictions
    )

    print("\nClassification Report")
    print("---------------------------")
    print(report)

    report_file = "classification_report.txt"

    with open(report_file, "w") as file:
        file.write(report)

    # ========================================================
    # CREATE CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        Y_test,
        predictions
    )

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title("Confusion Matrix")

    confusion_matrix_file = "confusion_matrix.png"

    plt.savefig(
        confusion_matrix_file,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # WINE QUALITY DISTRIBUTION
    # ========================================================

    plt.figure(figsize=(8,5))

    sns.countplot(
        x="quality",
        data=wine_dataset
    )

    plt.title("Wine Quality Distribution")

    count_plot_file = "wine_quality_countplot.png"

    plt.savefig(
        count_plot_file,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # VOLATILE ACIDITY VS QUALITY
    # ========================================================

    plt.figure(figsize=(8,5))

    sns.barplot(
        x="quality",
        y="volatile acidity",
        data=wine_dataset
    )

    plt.title("Volatile Acidity vs Quality")

    volatile_plot_file = "volatile_vs_quality.png"

    plt.savefig(
        volatile_plot_file,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # CORRELATION HEATMAP
    # ========================================================

    correlation = wine_dataset.corr()

    plt.figure(figsize=(12,10))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm"
    )

    plt.title("Wine Correlation Heatmap")

    heatmap_file = "wine_correlation_heatmap.png"

    plt.savefig(
        heatmap_file,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "classification_report_file": report_file,
        "confusion_matrix_file": confusion_matrix_file,
        "plots": [
            count_plot_file,
            volatile_plot_file,
            heatmap_file
        ]
    }