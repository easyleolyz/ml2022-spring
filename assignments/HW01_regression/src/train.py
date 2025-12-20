import os
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import MLP


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    set_seed(int(cfg["seed"]))

    data_dir = cfg["data_dir"]
    train_path = os.path.join(data_dir, cfg["train_csv"])
    test_path = os.path.join(data_dir, cfg["test_csv"])

    id_col = cfg["id_col"]
    target_col = cfg["target_col"]

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train csv not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test csv not found: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 用 test 的列（去掉 id）作为特征列，保证 train/test 对齐
    feature_cols = [c for c in test_df.columns if c != id_col]
    missing = set(feature_cols) - set(train_df.columns)
    if missing:
        raise ValueError(f"train 缺少 test 的特征列：{sorted(list(missing))[:20]} ...")

    X = train_df[feature_cols].values.astype(np.float32)
    y = train_df[target_col].values.astype(np.float32)

    # normalize by train stats
    mean = X.mean(axis=0, keepdims=True).astype(np.float32)
    std = (X.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    Xn = (X - mean) / std

    # split train/valid
    n = len(Xn)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * 0.8)
    tr_idx, va_idx = idx[:split], idx[split:]

    X_tr, y_tr = Xn[tr_idx], y[tr_idx]
    X_va, y_va = Xn[va_idx], y[va_idx]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).unsqueeze(1)),
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    valid_x = torch.from_numpy(X_va).to(device)
    valid_y = torch.from_numpy(y_va).unsqueeze(1).to(device)

    hidden_dims = tuple(cfg["hidden_dims"])
    model = MLP(
        in_dim=Xn.shape[1],
        hidden_dims=hidden_dims,
        dropout=float(cfg["dropout"]),
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    loss_fn = nn.MSELoss()

    best_mse = float("inf")
    best_state = None
    patience = int(cfg["patience"])
    wait = 0

    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            va_pred = model(valid_x)
            va_mse = loss_fn(va_pred, valid_y).item()

        if epoch % 10 == 0:
            print(f"epoch {epoch:03d} | valid MSE: {va_mse:.6f}")

        if va_mse < best_mse - 1e-6:
            best_mse = va_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {epoch}, best valid MSE = {best_mse:.6f}")
                break

    if best_state is None:
        raise RuntimeError("best_state is None: training did not run correctly")

    # save ckpt to outputs/
    hw_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(hw_root, cfg["output_dir"])
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, cfg["ckpt_name"])

    torch.save(
        {
            "model_state": best_state,
            "mean": mean,
            "std": std,
            "feature_cols": feature_cols,
            "id_col": cfg["id_col"],
            "target_col": cfg["target_col"],
            "submit_col": cfg["submit_col"],
            "best_valid_mse": float(best_mse),
            "hidden_dims": hidden_dims,
            "dropout": float(cfg["dropout"]),
        },
        ckpt_path,
    )
    print("saved ckpt:", ckpt_path)


if __name__ == "__main__":
    main()
