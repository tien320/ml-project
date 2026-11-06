
import numpy as np
from sklearn.model_selection import KFold

from src.config import LEARNING_RATE, EPOCHS, RANDOM_STATE
from src.models.linear_regression import LinearRegressionGD
from src.preprocessing.preprocessor import DataPreprocessor
from src.evaluation.metrics import evaluate_regression


def cross_validate_regression(
    X,
    y,
    n_splits=5,
    learning_rate=LEARNING_RATE,
    epochs=EPOCHS,
    random_state=RANDOM_STATE,
):
    """
    Evaluate Linear Regression using K-Fold Cross Validation.

    Each fold gets a new preprocessor and a new model.
    """

    # Convert input to NumPy arrays
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)

    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array.")

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples.")

    if n_splits < 2 or n_splits > len(X):
        raise ValueError("Invalid number of splits.")

    # Create K-Fold splitter
    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_results = []

    print("=" * 50)
    print(f"{n_splits}-FOLD CROSS VALIDATION")
    print("=" * 50)

    # Iterate through folds
    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(X), start=1
    ):

        print(f"\nFold {fold}/{n_splits}")

        # Split current fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create a new preprocessor for this fold
        preprocessor = DataPreprocessor()

        X_train_scaled, y_train_scaled = preprocessor.fit_transform(
            X_train,
            y_train,
        )

        X_val_scaled = preprocessor.transform(X_val)

        # Create a new model for this fold
        model = LinearRegressionGD(
            learning_rate=learning_rate,
            epochs=epochs,
        )

        # Train model
        model.fit(X_train_scaled, y_train_scaled)

        # Predict validation data
        y_pred_scaled = model.predict(X_val_scaled)

        # Convert predictions back to original target scale
        y_pred = preprocessor.inverse_transform_y(y_pred_scaled)

        # Evaluate
        metrics = evaluate_regression(y_val, y_pred)

        fold_results.append(metrics)

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

    # Aggregate results
    metric_names = fold_results[0].keys()

    summary = {}

    for name in metric_names:
        values = np.array([
            result[name] for result in fold_results
        ])

        summary[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    print("\n" + "=" * 50)
    print("CROSS VALIDATION SUMMARY")
    print("=" * 50)

    for name, result in summary.items():
        print(
            f"{name}: "
            f"{result['mean']:.4f} "
            f"+/- {result['std']:.4f}"
        )

    return {
        "fold_results": fold_results,
        "summary": summary,
    }
