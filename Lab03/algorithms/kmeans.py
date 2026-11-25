"""Lloyd K-Means with NumPy, k-means++ initialization and repeated restarts."""
import numpy as np


class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=42, n_init=20):
        if not isinstance(n_clusters, int) or n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer.")
        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
        if not isinstance(n_init, int) or n_init < 1:
            raise ValueError("n_init must be a positive integer.")
        if not np.isfinite(tol) or tol < 0:
            raise ValueError("tol must be finite and nonnegative.")
        self.n_clusters, self.max_iter, self.tol = n_clusters, max_iter, tol
        self.random_state, self.n_init = random_state, n_init

    def _distances(self, X, centers):
        return np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)

    def _initialize(self, X, rng):
        # Chọn ngẫu nhiên tâm đầu tiên
        centers = [X[rng.integers(len(X))].copy()]

        while len(centers) < self.n_clusters:
            # Khoảng cách bình phương đến tâm gần nhất
            nearest = self._distances(X, np.asarray(centers)).min(axis=1)

            # Điểm xa có xác suất được chọn cao hơn
            index = rng.choice(len(X), p=nearest / nearest.sum())
            centers.append(X[index].copy())
        return np.asarray(centers)

    def fit(self, X):
        # Chuẩn bị: X là ma trận (số mẫu, số đặc trưng), đã chuẩn hóa ở preprocessing.
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not X.size or not np.isfinite(X).all():
            raise ValueError("X must be a finite nonempty matrix.")
        if len(np.unique(X, axis=0)) < self.n_clusters:
            raise ValueError("K exceeds the number of distinct samples.")
        # Bước 1 — K đã được truyền qua n_clusters; fit không tự chọn K.
        rng = np.random.default_rng(self.random_state)
        best_inertia = np.inf
        # Lặp n_init lần khởi tạo độc lập, giữ nghiệm có inertia nhỏ nhất cho cùng K.
        for _ in range(self.n_init):
            # Bước 2 — Khởi tạo K tâm bằng k-means++.
            centers = self._initialize(X, rng)
            for iteration in range(1, self.max_iter + 1):
                # Bước 3 — Gán mỗi mẫu vào tâm gần nhất.
                distances = self._distances(X, centers)
                labels = distances.argmin(axis=1)
                # Bước 4 — Tâm mới là trung bình vector của các thành viên trong cụm.
                updated = centers.copy()
                nearest = distances.min(axis=1)
                for cluster in range(self.n_clusters):
                    members = X[labels == cluster]
                    if len(members):
                        updated[cluster] = members.mean(axis=0)
                    else:
                        # Cụm rỗng: đưa tâm đến điểm xa tâm gần nhất nhất hiện tại.
                        index = nearest.argmax()
                        updated[cluster] = X[index]
                        # Tránh chọn lại điểm này cho cụm rỗng tiếp theo trong cùng vòng.
                        nearest[index] = -1
                # Bước 5 — Kiểm tra hội tụ bằng độ dịch chuyển tổng thể của tất cả tâm.
                shift = np.sqrt(np.sum((updated - centers) ** 2))
                centers = updated
                # Nếu chưa đạt tol, lặp lại bước 3–4; tối đa max_iter vòng.
                if shift <= self.tol:
                    break
            distances = self._distances(X, centers)
            labels = distances.argmin(axis=1)
            inertia = float(distances[np.arange(len(X)), labels].sum())
            # Lưu các thuộc tính của nghiệm tốt nhất trong n_init lần.
            if inertia < best_inertia:
                best_inertia = inertia
                self.cluster_centers_ = centers.copy()
                self.labels_ = labels.copy()
                self.inertia_, self.n_iter_ = inertia, iteration
                # Hết max_iter mà shift > tol thì vẫn dừng, nhưng converged_ = False.
                self.converged_ = shift <= self.tol
        # Lưu số đặc trưng để kiểm tra đầu vào predict; trả self để gọi nối tiếp.
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Call fit before predict.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError("Prediction features must be finite and match training columns.")
        return self._distances(X, self.cluster_centers_).argmin(axis=1)

    def fit_predict(self, X):
        return self.fit(X).labels_
