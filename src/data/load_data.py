import pandas as pd


DEFAULT_DATASET_PATH = "data/raw/wine.csv"


def load_dataset(file_path=DEFAULT_DATASET_PATH):
    """
    Load the wine quality dataset.

    Parameters
    ----------
    file_path : str
        Path to the CSV dataset.

    Returns
    -------
    pandas.DataFrame
        Loaded wine dataset.
    """

    print(f"Loading dataset from: {file_path}")

    dataset = pd.read_csv(file_path)

    if dataset.empty:
        raise ValueError("Dataset is empty.")

    print(f"Dataset loaded successfully: {dataset.shape}")

    return dataset