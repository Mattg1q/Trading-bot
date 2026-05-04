import os
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from train_snn import train


def make_synthetic_dataset(path, rows=160):
    records = []
    start = datetime.now(timezone.utc)
    for index in range(rows):
        mid = 80000.0 + np.sin(index / 8.0) * 120.0 + index * 0.4
        record = {
            "timestamp_utc": (start + timedelta(seconds=index)).isoformat(),
            "mid_price": mid,
        }
        for level in range(1, 21):
            spread = 1.0 + level * 0.8
            record[f"bid_price_{level}"] = mid - spread
            record[f"bid_quantity_{level}"] = 1.0 + level * 0.1 + (index % 5) * 0.01
            record[f"ask_price_{level}"] = mid + spread
            record[f"ask_quantity_{level}"] = 1.0 + level * 0.1 + ((index + 2) % 5) * 0.01
        records.append(record)

    pd.DataFrame(records).to_csv(path, index=False)


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "lob.csv")
        weights_path = os.path.join(tmpdir, "snn_weights.pth")
        make_synthetic_dataset(data_path)

        class Args:
            input = data_path
            output = weights_path
            horizon = 10
            threshold_bps = 5.0
            epochs = 1
            batch_size = 32
            lr = 0.001
            weight_decay = 0.0001

        train(Args())
        if not os.path.exists(weights_path):
            raise RuntimeError("training did not create weights")
        print("smoke test passed")


if __name__ == "__main__":
    main()
