
import numpy as np


class LinearRegressionGD:

    def __init__(self, learning_rate=0.01, epochs=5000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = None
        self.loss_history = []
        # class LinearRegressionGD:
# Tạo một class đại diện cho thuật toán Linear Regression sử dụng Gradient Descent.
# __init__:
# Hàm khởi tạo, được gọi khi tạo object:
# Ý nghĩa:
# Thuộc tính	Công dụng
# learning_rate	Tốc độ cập nhật tham số
# epochs	Số vòng huấn luyện
# theta	Trọng số và bias
# loss_history	Lưu lịch sử sai số
# Tại sao theta = None?
# Vì khi mới tạo model, nó chưa được train nên chưa có trọng số.
# loss_history = [] là danh sách rỗng để lưu lại sai số qua từng epoch.

    def fit(self, X, y):
        m = X.shape[0] #Lấy số lượng mẫu train.

        # Thêm cột bias
        X_bias = np.c_[np.ones((m, 1)), X] #Thêm một cột toàn số 1 vào X.

        # Khởi tạo weights và bias
        self.theta = np.zeros((X_bias.shape[1], 1)) #Khởi tạo trọng số bằng 0.

        # Gradient Descent
        for epoch in range(self.epochs):

            y_pred = X_bias @ self.theta #Đây là quá trình tính giá nhà dự đoán từ các feature và tham số hiện tại. @ là toán tử nhân ma trận trong Python.

            error = y_pred - y # tính sai số

            gradient = (2 / m) * (X_bias.T @ error) #Đây là đạo hàm của MSE Loss theo các tham số.
#Mục đích: xác định mỗi trọng số cần thay đổi theo hướng nào để giảm sai số.

            self.theta -= self.learning_rate * gradient
#             Đây là Gradient Descent.
# - Gradient cho biết hướng thay đổi.
# - Learning rate quyết định độ lớn bước thay đổi.
# - Theta được cập nhật sau mỗi vòng lặp.

            loss = np.mean(error ** 2) #Tính trung bình bình phương sai số và lưu lại.
            self.loss_history.append(loss)
# Lưu ý nhỏ: loss ở đây được tính trước khi cập nhật theta trong epoch đó. Điều này không sai; nó ghi nhận loss của lần dự đoán trước cập nhật.

        return self

    def predict(self, X):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        return X_bias @ self.theta
#hàm predict dự đoán:
# Khác với fit():
# - fit() học trọng số.
# - predict() chỉ sử dụng trọng số đã học.
# Predict không cập nhật theta.

# Đây chính là cách model được sử dụng sau khi đã train xong.
