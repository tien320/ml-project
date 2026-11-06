
import joblib
import numpy as np

from sklearn.model_selection import train_test_split

from src.config import (
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
from src.evaluation.metrics import evaluate_regression


def train():

    print("=" * 50)
    print("LINEAR REGRESSION TRAINING")
    print("=" * 50)

    # 1. Load dataset
    X, y = load_dataset()

    print("Dataset loaded:", X.shape)

    # 2. Select numerical features
    X = X.select_dtypes(include=np.number)

    # 3. Drop unnecessary columns
    X = X.drop(
        columns=FEATURE_COLUMNS_TO_DROP,
        errors="ignore"
    )

    print("Selected features:", X.shape[1])
# Dataset có 80 cột X, nhưng không phải cột nào cũng là số.
# Ta chỉ chọn các cột numeric vì Linear Regression hiện tại của chúng ta chưa hỗ trợ categorical encoding.
# Sau đó loại Id, vì đây chỉ là mã định danh căn nhà, không phải feature có ý nghĩa dự đoán trong baseline hiện tại.
    # 4. Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
# Chia dữ liệu:
# - 80% để học.
# - 20% để kiểm tra khả năng dự đoán trên dữ liệu chưa được dùng để train.
# Điểm quan trọng: ta chia dữ liệu trước khi fit preprocessing.
    print("Train samples:", len(X_train))
    print("Test samples:", len(X_test))

    # 5. Preprocessing
    preprocessor = DataPreprocessor()

    X_train_scaled, y_train_scaled = preprocessor.fit_transform(
        X_train,
        y_train
    )

    X_test_scaled = preprocessor.transform(X_test)
    """
Đây là quy tắc chống Data Leakage.
- Train: học median, mean, std rồi biến đổi.
- Test: chỉ sử dụng các thông số đã học từ train.
Không được tính lại thông số trên test.
"""
    # 6. Train model
    model = LinearRegressionGD(
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS
    )

    model.fit(X_train_scaled, y_train_scaled)
    """
Lúc này Gradient Descent mới thực sự học weights và bias.
Lưu ý rằng model học trên dữ liệu đã chuẩn hóa cả X và y.
"""

    print("Training completed.")

    # 7. Predict on train and test
    train_pred_scaled = model.predict(X_train_scaled)
    test_pred_scaled = model.predict(X_test_scaled)

    # 8. Convert predictions to original target scale
    train_pred = preprocessor.inverse_transform_y(
        train_pred_scaled
    )

    test_pred = preprocessor.inverse_transform_y(
        test_pred_scaled
    )
#Model dự đoán trên scale đã chuẩn hóa. Ta chuyển kết quả về đơn vị giá nhà ban đầu trước khi đánh giá.
    # 9. Evaluate
    train_metrics = evaluate_regression(
        y_train,
        train_pred
    )

    test_metrics = evaluate_regression(
        y_test,
        test_pred
    )

    print("\nTRAIN METRICS")
    for name, value in train_metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nTEST METRICS")
    for name, value in test_metrics.items():
        print(f"{name}: {value:.4f}")
"""
Train metrics: model khớp với dữ liệu đã học đến đâu.
Test metrics: model tổng quát hóa sang dữ liệu chưa thấy tốt đến đâu.
"""
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
"""
Tại sao không chỉ lưu model?
Vì model cần cả preprocessor để biến đổi dữ liệu mới giống lúc training. Ta lưu cả hai cùng metadata về thứ tự feature.
joblib giúp serialize đối tượng Python thành file để nạp lại sau này.
"""
    print("\nModel saved to:", MODEL_PATH)
    print("=" * 50)

    return artifact


if __name__ == "__main__":
    train()