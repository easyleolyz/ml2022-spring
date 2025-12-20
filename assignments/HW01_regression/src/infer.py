import os
import argparse
import yaml
import numpy as np
import pandas as pd
import torch

from models import MLP


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    data_dir = cfg["data_dir"]
    test_path = os.path.join(data_dir, cfg["test_csv"])

    hw_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(hw_root, cfg["output_dir"])
    ckpt_path = os.path.join(out_dir, cfg["ckpt_name"])

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test csv not found: {test_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}. Run train.py first.")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    test_df = pd.read_csv(test_path)

    id_col = ckpt["id_col"]
    submit_col = ckpt["submit_col"]  # 必须是 tested_positive
    feature_cols = ckpt["feature_cols"]

    X_test = test_df[feature_cols].values.astype(np.float32)
    mean = ckpt["mean"].astype(np.float32)
    std = ckpt["std"].astype(np.float32)
    Xn_test = (X_test - mean) / std

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(
        in_dim=Xn_test.shape[1],
        hidden_dims=ckpt["hidden_dims"],
        dropout=float(ckpt["dropout"]),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        preds = model(torch.from_numpy(Xn_test).to(device)).cpu().numpy().reshape(-1)

    sub = pd.DataFrame(
        {
            id_col: test_df[id_col].to_numpy(),
            submit_col: preds.astype(np.float32),
        }
    ).sort_values(id_col).reset_index(drop=True)

    # sanity checks
    assert sub.shape[0] == test_df.shape[0]
    assert np.isfinite(sub[submit_col].to_numpy()).all()

    os.makedirs(out_dir, exist_ok=True)
    sub_path = os.path.join(out_dir, cfg["submission_name"])
    sub.to_csv(sub_path, index=False)

    print("saved submission:", sub_path)
    print("columns:", list(sub.columns))
    print(sub.head())


if __name__ == "__main__":
    main()
