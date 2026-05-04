# SNN Training

This bot needs trained SNN weights before it should trade in `Demo Futures`.

## Quick Start

Run:

```bat
train_snn.bat
```

This collects one hour of BTC/USDT Futures LOB snapshots at one sample per second, trains
`SpatialLOBModel`, and writes:

```text
models/snn_weights.pth
```

## Manual Commands

Collect data:

```bat
venv\Scripts\python.exe collect_lob_training_data.py --samples 3600 --interval 1 --output data\lob_snapshots.csv
```

Train:

```bat
venv\Scripts\python.exe train_snn.py --input data\lob_snapshots.csv --output models\snn_weights.pth --horizon 30 --threshold-bps 5 --epochs 20
```

## When Weights Exist

Update `.env`:

```text
TRADING_MODE=Demo Futures
ALLOW_UNTRAINED_SNN=0
SNN_WEIGHTS_PATH=models/snn_weights.pth
```

Then restart the app.

## Notes

The short smoke test only proves that the training pipeline works. It does not create a
profitable model. For a real model, collect more data across different market regimes and
validate out-of-sample before enabling order execution.
