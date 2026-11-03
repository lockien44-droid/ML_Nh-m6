import numpy as np


class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha phải là số dương hữu hạn.")

        self.alpha = alpha
        self.class_log_prior = None
        self.feature_log_prob = None

    def _prepare_X(self, X):
        if hasattr(X, "tocsr"):
            # Giữ ma trận TF-IDF ở dạng thưa.
            X = X.tocsr().astype(float, copy=False)
            values = X.data
        else:
            X = np.asarray(X, dtype=float)
            values = X

        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X phải là ma trận có ít nhất một cột.")

        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Các giá trị trong X phải hữu hạn và không âm.")

        return X

    def fit(self, X, y):
        X = self._prepare_X(X)
        y = np.asarray(y)

        if y.shape != (X.shape[0],) or not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("y phải có một nhãn cho mỗi dòng X và chứa cả 0 lẫn 1.")

        # 1. Đếm số email thuộc từng lớp: hàng 0 là ham, hàng 1 là spam.
        class_counts = np.bincount(y.astype(int), minlength=2)
        class_log_prior = np.log(class_counts / len(y))

        # 2. Trong mỗi lớp, cộng trọng số TF-IDF theo từng cột từ.
        feature_totals = np.vstack([
            np.asarray(X[y == label].sum(axis=0)).ravel()
            for label in (0, 1)
        ])

        # 3. Làm trơn rồi chuẩn hóa các trọng số từ trong từng lớp.
        smoothed = feature_totals + self.alpha
        feature_probs = smoothed / smoothed.sum(axis=1, keepdims=True)

        self.class_log_prior = class_log_prior
        self.feature_log_prob = np.log(feature_probs)
        return self

    def predict(self, X):
        if self.feature_log_prob is None:
            raise ValueError("Hãy gọi fit() trước khi dự đoán.")

        X = self._prepare_X(X)

        if X.shape[1] != self.feature_log_prob.shape[1]:
            raise ValueError("X phải có cùng số cột như khi huấn luyện.")

        # Mỗi email có hai điểm: một cho ham và một cho spam.
        scores = X @ self.feature_log_prob.T + self.class_log_prior
        return np.argmax(scores, axis=1)