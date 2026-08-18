# ============================================================
# IMPORT LIBRARIES
# ============================================================

from src.data.load_data import load_dataset

from src.features.preprocessing import preprocess_data

from src.models.train import train_model

# ============================================================
# TEST COMPLETE PIPELINE
# ============================================================

def test_training_pipeline():
    """
    Verify the pipeline executes successfully.
    """

    dataset = load_dataset()

    X_train, X_test, y_train, y_test = preprocess_data(
        dataset
    )

    model = train_model(
        X_train,
        y_train
    )

    assert model is not None