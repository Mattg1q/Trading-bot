import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import SpatialLOBModel


def build_features(frame):
    features = []
    mid_prices = frame["mid_price"].to_numpy(dtype=np.float32)

    for _, row in frame.iterrows():
        mid = float(row["mid_price"])
        levels = []
        for level in range(1, 21):
            levels.append([
                (float(row[f"bid_price_{level}"]) / mid) - 1.0,
                float(row[f"bid_quantity_{level}"]),
                (float(row[f"ask_price_{level}"]) / mid) - 1.0,
                float(row[f"ask_quantity_{level}"]),
            ])
        features.append(levels)

    return np.asarray(features, dtype=np.float32), mid_prices


def build_labels(mid_prices, horizon, threshold_bps):
    future = np.roll(mid_prices, -horizon)
    returns = (future - mid_prices) / mid_prices
    threshold = threshold_bps / 10000.0
    labels = np.clip(returns / threshold, -1.0, 1.0)
    return labels.astype(np.float32)


def train(args):
    frame = pd.read_csv(args.input).dropna()
    min_rows = args.horizon + 100
    if len(frame) < min_rows:
        raise RuntimeError(f"Need at least {min_rows} rows, found {len(frame)}.")

    features, mid_prices = build_features(frame)
    labels = build_labels(mid_prices, args.horizon, args.threshold_bps)
    features = features[:-args.horizon]
    labels = labels[:-args.horizon]

    split_index = int(len(features) * 0.8)
    train_x = torch.tensor(features[:split_index], dtype=torch.float32)
    train_y = torch.tensor(labels[:split_index], dtype=torch.float32).unsqueeze(1)
    val_x = torch.tensor(features[split_index:], dtype=torch.float32)
    val_y = torch.tensor(labels[split_index:], dtype=torch.float32).unsqueeze(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatialLOBModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x.to(device))
            val_loss = loss_fn(val_pred, val_y.to(device)).item()
            directional_accuracy = (
                torch.sign(val_pred.cpu()).eq(torch.sign(val_y)).float().mean().item()
            )

        train_loss /= len(train_x)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_dir_acc={directional_accuracy:.3f}"
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "horizon": args.horizon,
            "threshold_bps": args.threshold_bps,
            "rows": len(features),
        },
        args.output,
    )
    print(f"saved weights to {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train SpatialLOBModel from collected LOB snapshots.")
    parser.add_argument("--input", default="data/lob_snapshots.csv")
    parser.add_argument("--output", default="models/snn_weights.pth")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--threshold-bps", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
