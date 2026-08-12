
# ============================================================
# IMPORT LIBRARIES
# ============================================================

from sklearn.model_selection import train_test_split


# ============================================================
# PREPROCESS DATA
# ============================================================

def preprocess_data(
    wine_dataset,
    test_size=0.2,
    random_state=42
):
    """
    Prepare the wine dataset for machine learning.

    Steps:
        1. Separate features and target
        2. Convert quality scores into binary labels
        3. Split the dataset into training and testing sets

    Parameters
    ----------
    wine_dataset : pandas.DataFrame
        The loaded wine dataset.

    test_size : float
        Percentage of data used for testing.

    random_state : int
        Random seed used for reproducibility.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """

    # ========================================================
    # FEATURE SELECTION
    # ========================================================

    print("Separating features and target...")

    X = wine_dataset.drop(
        "quality",
        axis=1
    )

    # ========================================================
    # TARGET VARIABLE
    # ========================================================

    print("Converting quality scores into binary labels...")

    # Quality >= 7 -> Good wine = 1
    # Quality < 7  -> Bad wine  = 0

    y = wine_dataset["quality"].apply(
        lambda quality: 1 if quality >= 7 else 0
    )

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    print("Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    # ========================================================
    # DISPLAY INFORMATION
    # ========================================================

    print(f"Training features: {X_train.shape}")
    print(f"Testing features: {X_test.shape}")

    print(f"Training labels: {y_train.shape}")
    print(f"Testing labels: {y_test.shape}")

    # ========================================================
    # RETURN PREPROCESSED DATA
    # ========================================================

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )
