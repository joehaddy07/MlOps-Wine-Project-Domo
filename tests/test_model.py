# This file checks that your machine learning model is working properly.

# ============================================================
# IMPORT LIBRARIES
# ============================================================

from sklearn.ensemble import RandomForestClassifier

# ============================================================
# TEST MODEL CREATION
# ============================================================

def test_model_creation():
    """
    Verify the Random Forest model is created.
    """

    model = RandomForestClassifier()

    assert model is not None

# ============================================================
# TEST NUMBER OF TREES
# ============================================================

def test_n_estimators():
    """
    Verify the model uses 100 trees.
    """

    model = RandomForestClassifier(
        n_estimators=100
    )

    assert model.n_estimators == 100