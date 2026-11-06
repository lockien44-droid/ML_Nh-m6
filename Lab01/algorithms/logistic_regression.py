import numpy as np
from scipy.sparse import issparse


class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0.0

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, *, on_iteration=None):
        #Chuẩn bị dữ liệu
        X = X.tocsr().astype(float, copy=False) if issparse(X) else np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        #Kiểm tra dữ liệu hợp lệ
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X phải là ma trận có ít nhất một dòng và một cột.")

        n_samples, n_features = X.shape

        if y.shape != (n_samples,) or not np.all((y == 0) | (y == 1)):
            raise ValueError("y phải có một nhãn 0 hoặc 1 cho mỗi dòng của X.")

        # Khởi tạo trọng số
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for iteration in range(self.n_iterations):
            if on_iteration is not None:
                on_iteration(iteration, self)
            # Tính xác suất dự đoán
            z = X @ self.weights + self.bias
            probabilities = self.sigmoid(z)

           # Tính sai số và gradient
            error = probabilities - y
            dw = (X.T @ error) / n_samples
            db = np.mean(error)

            # 3. Cập nhật trọng số và bias.
    
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        if on_iteration is not None:
            on_iteration(self.n_iterations, self)
        return self

    # Tính xác suất Spam
    def predict_proba(self, X):
        if self.weights is None:
            raise ValueError("Hãy gọi fit() trước khi dự đoán.")

        X = X.tocsr().astype(float, copy=False) if issparse(X) else np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.weights.size:
            raise ValueError("X phải có cùng số cột đặc trưng như khi huấn luyện.")

        z = X @ self.weights + self.bias
        return self.sigmoid(z)
    # Đưa ra nhãn cuối cùng
    def predict(self, X):
        probabilities = self.predict_proba(X)
        # True thành 1 (spam), False thành 0 (ham). Đúng 0,5 được gán spam.
        return (probabilities >= 0.5).astype(int)
