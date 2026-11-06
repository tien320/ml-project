
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from src.core.config import (
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    LEARNING_RATE,
    EPOCHS,
    FEATURE_COLUMNS_TO_DROP,
    MODEL_PATH,
    ensure_directories,
)
from src.data.loader import load_dataset
from src.preprocessing.preprocessor import DataPreprocessor
from src.models.linear_regression import LinearRegressionGD
from src.evaluations.metrics import evaluate_regression


def train():
    print("=" * 50)
    print("LINEAR REGRESSION TRAINING")
    print("=" * 50)

    # 1. Load dataset
    X, y = load_dataset()
    print("Dataset loaded:", X.shape)

    # 2. Select numeric columns only
    X = X.select_dtypes(include=np.number)

    # 3. Drop unnecessary columns
    X = X.drop(columns=FEATURE_COLUMNS_TO_DROP, errors="ignore")
    print("Selected features:", X.shape[1])

    # 4. Split train/test set before preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    print("Train samples:", len(X_train))
    print("Test samples:", len(X_test))

    # 5. Preprocessing
    preprocessor = DataPreprocessor()
    X_train_scaled, y_train_scaled = preprocessor.fit_transform(X_train, y_train)
    X_test_scaled = preprocessor.transform(X_test)

    # 6. Train model
    model = LinearRegressionGD(learning_rate=LEARNING_RATE, epochs=EPOCHS)
    model.fit(X_train_scaled, y_train_scaled)
    print("Training completed.")

    # 7. Predict on train and test
    train_pred_scaled = model.predict(X_train_scaled)
    test_pred_scaled = model.predict(X_test_scaled)

    # 8. Convert predictions back to original target scale
    train_pred = preprocessor.inverse_transform_y(train_pred_scaled)
    test_pred = preprocessor.inverse_transform_y(test_pred_scaled)

    # 9. Evaluate
    train_metrics = evaluate_regression(y_train, train_pred)
    test_metrics = evaluate_regression(y_test, test_pred)

    print("\nTRAIN METRICS")
    for name, value in train_metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nTEST METRICS")
    for name, value in test_metrics.items():
        print(f"{name}: {value:.4f}")

    # 10. Save model and preprocessor together
    ensure_directories()
    artifact = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": X.columns.tolist(),
        "target_column": TARGET_COLUMN,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    joblib.dump(artifact, MODEL_PATH)

    print("\nModel saved to:", MODEL_PATH)
    print("=" * 50)
    return artifact


if __name__ == "__main__":
    train()