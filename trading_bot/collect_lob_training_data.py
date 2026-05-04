import argparse
import asyncio
import csv
import os
from datetime import datetime, timezone

from data_handler import DataHandler


FIELDNAMES = [
    "timestamp_utc",
    "mid_price",
    *[f"bid_price_{i}" for i in range(1, 21)],
    *[f"bid_quantity_{i}" for i in range(1, 21)],
    *[f"ask_price_{i}" for i in range(1, 21)],
    *[f"ask_quantity_{i}" for i in range(1, 21)],
]


async def collect_snapshots(output_path, samples, interval_seconds):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    file_exists = os.path.exists(output_path)
    handler = DataHandler()

    try:
        await handler.start_lob_stream()
        with open(output_path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
            if not file_exists:
                writer.writeheader()

            for sample_index in range(samples):
                rows, mid_price = await handler.get_lob_snapshot_rows()
                if rows:
                    record = {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "mid_price": mid_price,
                    }
                    for row in rows:
                        level = row["level"]
                        record[f"bid_price_{level}"] = row["bid_price"]
                        record[f"bid_quantity_{level}"] = row["bid_quantity"]
                        record[f"ask_price_{level}"] = row["ask_price"]
                        record[f"ask_quantity_{level}"] = row["ask_quantity"]

                    writer.writerow(record)
                    csv_file.flush()
                    print(f"saved {sample_index + 1}/{samples} mid={mid_price:.2f}")
                else:
                    print(f"skipped {sample_index + 1}/{samples}: empty LOB")

                await asyncio.sleep(interval_seconds)
    finally:
        await handler.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Collect Binance BTC/USDT LOB snapshots for SNN training.")
    parser.add_argument("--output", default="data/lob_snapshots.csv")
    parser.add_argument("--samples", type=int, default=1800)
    parser.add_argument("--interval", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(collect_snapshots(args.output, args.samples, args.interval))
