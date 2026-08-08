# ============================================================
# IMPORT LIBRARIES
# ============================================================

from sklearn.ensemble import RandomForestClassifier

# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(X_train, Y_train):
    """
    Train a Random Forest classifier.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training features.

    Y_train : pandas.Series
        Training labels.

    Returns
    -------
    RandomForestClassifier
        The trained machine learning model.
    """

    # ========================================================
    # MODEL CREATION
    # ========================================================

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # ========================================================
    # TRAIN MODEL
    # ========================================================

    model.fit(X_train, Y_train)

    # ========================================================
    # RETURN TRAINED MODEL
    # ========================================================

    return model