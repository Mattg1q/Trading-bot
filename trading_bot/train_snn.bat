@echo off
title Train SNN Weights

call venv\Scripts\activate.bat

echo Collecting LOB snapshots...
python collect_lob_training_data.py --samples 3600 --interval 1 --output data\lob_snapshots.csv
if errorlevel 1 (
    echo Failed to collect training data.
    pause
    exit /b 1
)

echo Training SNN...
python train_snn.py --input data\lob_snapshots.csv --output models\snn_weights.pth --horizon 30 --threshold-bps 5 --epochs 20
if errorlevel 1 (
    echo Failed to train SNN.
    pause
    exit /b 1
)

echo Done. Weights saved to models\snn_weights.pth
pause
