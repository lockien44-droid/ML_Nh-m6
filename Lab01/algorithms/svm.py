import numpy as np


class LinearSVM:
    def __init__(self, learning_rate=0.01, n_iterations=1000, C=1.0, *,
                 class_weight=None, threshold=0.0):
        if learning_rate <= 0 or n_iterations < 1 or C <= 0:
            raise ValueError("learning_rate, n_iterations và C phải dương.")

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.C = C
        self.class_weight = class_weight
        self.threshold = threshold
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y, *, on_iteration=None):
        # Giữ nguyên ma trận thưa nếu X được tạo bằng TF-IDF.
        X = X.astype(float) if hasattr(X, "tocsr") else np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X phải có ít nhất một dòng và một cột.")

        n_samples, n_features = X.shape

        if y.shape != (n_samples,) or not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("y phải có một nhãn cho mỗi dòng X và chứa cả 0 lẫn 1.")

        # Đổi nhãn 0/1 sang -1/+1 để tính margin.
        y_signed = np.where(y == 1, 1.0, -1.0)
        class_weight = getattr(self, "class_weight", None)
        if class_weight == "balanced":
            weights = n_samples / (2 * np.bincount(y.astype(int)))
        elif class_weight is None:
            weights = np.ones(2)
        else:
            weights = np.array([class_weight[0], class_weight[1]], dtype=float)
            if not np.all(np.isfinite(weights) & (weights > 0)):
                raise ValueError("Trọng số lớp phải hữu hạn và dương.")
        sample_weight = weights[y.astype(int)]

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for iteration in range(self.n_iterations):
            if on_iteration is not None:
                on_iteration(iteration, self)
            # 1. Tính điểm và margin của từng mẫu.
            scores = X @ self.weights + self.bias
            margins = y_signed * scores

            # 2. Chỉ các mẫu có margin < 1 đóng góp vào gradient hinge loss.
            active_labels = sample_weight * y_signed * (margins < 1)

            # 3. Tính gradient của hàm mục tiêu theo w và b.
            dw = self.weights - self.C * (X.T @ active_labels) / n_samples
            db = -self.C * np.mean(active_labels)

            # 4. Cập nhật trọng số và bias.
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        if on_iteration is not None:
            on_iteration(self.n_iterations, self)
        return self

    def decision_function(self, X):
        if self.weights is None:
            raise ValueError("Hãy gọi fit() trước khi dự đoán.")

        X = X if hasattr(X, "tocsr") else np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.weights.size:
            raise ValueError("X phải có cùng số cột như dữ liệu huấn luyện.")

        return X @ self.weights + self.bias

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores >= getattr(self, "threshold", 0.0)).astype(int)
