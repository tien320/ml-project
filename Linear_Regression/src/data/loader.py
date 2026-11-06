
import pandas as pd

from src.core.config import DATA_PATH, TARGET_COLUMN


def load_dataset():
    """
    Load house price dataset from CSV.
    """

    # Check whether dataset exists
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # Load CSV
    df = pd.read_csv(DATA_PATH)

    # Validate dataset
    if df.empty:
        raise ValueError("Dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    # Separate features and target
    X = df.drop(
        columns=[TARGET_COLUMN],
        errors="ignore"
    )

    y = df[TARGET_COLUMN]

    return X, y
