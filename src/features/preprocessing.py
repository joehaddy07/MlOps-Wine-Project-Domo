# ============================================================
# IMPORT LIBRARIES
# ============================================================

from sklearn.model_selection import train_test_split

# ============================================================
# PREPROCESS DATA
# ============================================================

def preprocess_data(wine_dataset):
    """
    Prepare the dataset for machine learning.

    Steps:
        1. Separate features and target
        2. Convert quality scores into binary labels
        3. Split the dataset into training and testing sets

    Parameters
    ----------
    wine_dataset : pandas.DataFrame
        The loaded wine dataset.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """

    # ========================================================
    # FEATURE SELECTION
    # ========================================================

    # Remove the target column from the input features.
    X = wine_dataset.drop("quality", axis=1)

    # ========================================================
    # TARGET VARIABLE
    # ========================================================

    # Convert wine quality into a binary classification task.
    # Quality >= 7  -> Good wine (1)
    # Quality < 7   -> Bad wine (0)

    y = wine_dataset["quality"].apply(
        lambda quality: 1 if quality >= 7 else 0
    )

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    test_size = 0.20

    random_state = 2

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=test_size,

        random_state=random_state
    )

    # ========================================================
    # RETURN PREPROCESSED DATA
    # ========================================================

    return (

        X_train,

        X_test,

        y_train,

        y_test
    )