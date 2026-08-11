# ============================================================
# IMPORT LIBRARIES
# ============================================================

from pathlib import Path

from src.data.load_data import load_dataset
from src.features.preprocessing import preprocess_data
from src.models.train import train_model


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "wine.csv"


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("WINE QUALITY MLOPS PIPELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1 — LOAD DATASET
    # --------------------------------------------------------

    print("\n[1/3] Loading dataset...")

    wine_dataset = load_dataset(DATASET_PATH)

    print(f"Dataset shape: {wine_dataset.shape}")

    # --------------------------------------------------------
    # STEP 2 — PREPROCESS DATA
    # --------------------------------------------------------

    print("\n[2/3] Preprocessing dataset...")

    X_train, X_test, Y_train, Y_test = preprocess_data(
        wine_dataset
    )

    print("Preprocessing completed.")

    # --------------------------------------------------------
    # STEP 3 — TRAIN MODEL
    # --------------------------------------------------------

    print("\n[3/3] Training model...")

    model = train_model(
        X_train,
        Y_train
    )

    print("Model training completed.")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()