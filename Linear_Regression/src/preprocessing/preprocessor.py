
import numpy as np
# 1. Nhiệm vụ
# File này chịu trách nhiệm:
# 1. Xử lý giá trị bị thiếu (Missing Values).
# 2. Chuẩn hóa features bằng Standardization.
# 3. Chuẩn hóa target SalePrice.
# 4. Lưu các tham số preprocessing đã học từ tập train.
# 5. Dùng lại chính xác các tham số đó khi dự đoán dữ liệu mới.
# Ta tự code bằng NumPy, không dùng StandardScaler hay SimpleImputer của sklearn.

class DataPreprocessor:

    def __init__(self):
        # Parameters learned from training data
        self.medians = None
        self.x_mean = None
        self.x_std = None

        self.y_mean = None
        self.y_std = None

    def fit_transform(self, X, y):
        """
Đây là hàm dành cho giai đoạn training.
Nó vừa học các thông số preprocessing từ tập train, vừa biến đổi dữ liệu.
"""

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        # 1. Calculate median from training data
        self.medians = np.nanmedian(X, axis=0)

        # 2. Impute missing values
        X = self._impute(X)

        # 3. Calculate mean and standard deviation
        self.x_mean = np.mean(X, axis=0)
        self.x_std = np.std(X, axis=0)

        # Avoid division by zero
        self.x_std[self.x_std == 0] = 1.0

        # 4. Standardize features
        X_scaled = (X - self.x_mean) / self.x_std

        # 5. Standardize target
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)

        if self.y_std == 0:
            self.y_std = 1.0

        y_scaled = (y - self.y_mean) / self.y_std

        return X_scaled, y_scaled

    def transform(self, X):
        """
        Transform new data using learned training parameters.
        """
# Đây là hàm dành cho dữ liệu validation, test hoặc dữ liệu khách hàng mới.
# Nó không tính lại median, mean hay std.
# Nó chỉ sử dụng các thông số đã được lưu từ fit_transform().
# Đây chính là cách ngăn data leakage trong preprocessing.

        if self.medians is None:
            raise RuntimeError("Preprocessor has not been fitted.")

        X = np.asarray(X, dtype=float)

        X = self._impute(X)

        X_scaled = (X - self.x_mean) / self.x_std

        return X_scaled

    def inverse_transform_y(self, y_scaled):
        """
        Convert standardized predictions back to original target scale.
        """

        if self.y_mean is None:
            raise RuntimeError("Preprocessor has not been fitted.")

        y_scaled = np.asarray(y_scaled, dtype=float)

        return y_scaled * self.y_std + self.y_mean

    def _impute(self, X):
        """
        Replace missing values using training medians.
        """

        X = X.copy()

        rows, cols = np.where(np.isnan(X))

        X[rows, cols] = self.medians[cols]

        return X
