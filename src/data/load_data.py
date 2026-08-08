# ============================================================
# IMPORT LIBRARIES
# ============================================================

# Path handling
from pathlib import Path

# Data manipulation
import pandas as pd

# ============================================================
# DATASET LOCATION
# ============================================================

# Get the root directory of the project.
# __file__ refers to this file (load_data.py).
# parents[2] moves up to the project root folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Build the full path to the dataset.
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "wine.csv"

# ============================================================
# LOAD DATASET FUNCTION
# ============================================================

def load_dataset():
    """
    Load the Wine Quality dataset.

    Returns
    -------
    pandas.DataFrame
        The loaded wine dataset.
    """

    print("\nLoading dataset...")
    print(f"Dataset location: {DATASET_PATH}")

    # Read the CSV file
    wine_dataset = pd.read_csv(DATASET_PATH)

    # Display basic information
    print("\nDataset Shape")
    print("---------------------------")
    print(wine_dataset.shape)

    print("\nFirst Five Rows")
    print("---------------------------")
    print(wine_dataset.head())

    print("\nMissing Values")
    print("---------------------------")
    print(wine_dataset.isnull().sum())

    print("\nDataset Statistics")
    print("---------------------------")
    print(wine_dataset.describe())

    # Return the dataset so other modules can use it
    return wine_dataset