# src/train.py

import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import MLP
from .evaluate import predict

def fit_mlp(X, y, V, actual, params, seed):
    """
    Huấn luyện MLP bằng PyTorch.

    X, y: đặc trưng và giá thật của tập train.
    V, actual: đặc trưng và giá thật của tập validation.
    params: cấu hình mô hình và huấn luyện.
    seed: giá trị đặt để kết quả có thể lặp lại.
    """

    # 1. Đặt seed cho các thao tác ngẫu nhiên
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 2. Khởi tạo MLP
    # X.shape[1] là số đặc trưng đầu vào.
    # params["hidden"] chứa cấu trúc các lớp ẩn.
    model = MLP(X.shape[1], params["hidden"])

    # 3. Chọn cách cập nhật trọng số và cách tính lỗi
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["learning_rate"],
    )
    criterion = nn.MSELoss()

    # 4. Đóng gói dữ liệu train thành các batch
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )

    loader = DataLoader(
        dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    # 5. Khởi tạo thông tin theo dõi epoch tốt nhất
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history = []

    # 6. Huấn luyện qua nhiều epoch
    for epoch in range(1, params["epochs"] + 1):
        model.train()

        # 7. Học lần lượt trên từng batch
        for batch_X, batch_y in loader:
            optimizer.zero_grad()               # Xóa gradient cũ

            predictions = model(batch_X)        # Tính giá dự đoán
            loss = criterion(predictions, batch_y)  # Tính MSE

            loss.backward()                     # Tính gradient
            optimizer.step()                    # Adam cập nhật trọng số

        # 8. Tính dự đoán trên train và validation sau epoch này
        predicted_train = predict(model, X)
        predicted_val = predict(model, V)

        # 9. Tính lỗi trên validation
        val_mse = float(np.mean((predicted_val - actual) ** 2))

        if not np.isfinite(val_mse):
            raise ValueError("Nonfinite validation loss.")

        # 10. Lưu MSE và MAE của epoch vào lịch sử
        history.append({
            "epoch": epoch,
            "train_mse": float(np.mean((predicted_train - y) ** 2)),
            "validation_mse": val_mse,
            "train_mae": float(np.mean(np.abs(predicted_train - y))),
            "validation_mae": float(np.mean(np.abs(predicted_val - actual))),
        })

        # 11. Nếu validation MSE tốt hơn, lưu lại trọng số
        if val_mse < best_loss:
            best_loss = val_mse
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        # 12. Dừng sớm nếu validation không cải thiện đủ lâu
        if stale >= params["patience"]:
            break

    # Không có checkpoint nếu không chạy epoch nào
    if best_state is None:
        raise ValueError("Training requires at least one epoch.")

    # 13. Khôi phục trọng số ở epoch validation tốt nhất
    model.load_state_dict(best_state)
    model.eval()

    # 14. Trả về mô hình tốt nhất, lịch sử và epoch tốt nhất
    return model, pd.DataFrame(history), best_epoch