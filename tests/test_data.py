
# This file verifies that the dataset is valid before training begins.
# ============================================================
# IMPORT LIBRARIES
# ============================================================

from src.data.load_data import load_dataset

# ============================================================
# TEST DATASET LOADS SUCCESSFULLY
# ============================================================

def test_dataset_loads():
    """
    Verify that the dataset loads successfully.
    """

    dataset = load_dataset()

    assert dataset is not None

# ============================================================
# TEST DATASET IS NOT EMPTY
# ============================================================

def test_dataset_not_empty():
    """
    Verify the dataset contains rows.
    """

    dataset = load_dataset()

    assert len(dataset) > 0

# ============================================================
# TEST TARGET COLUMN EXISTS
# ============================================================

def test_quality_column_exists():
    """
    Verify the target column exists.
    """

    dataset = load_dataset()

    assert "quality" in dataset.columns

# ============================================================
# TEST MISSING VALUES
# ============================================================

def test_missing_values():
    """
    Verify there are no missing values.
    """

    dataset = load_dataset()

    assert dataset.isnull().sum().sum() == 0