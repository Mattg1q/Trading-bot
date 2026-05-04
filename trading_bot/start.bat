@echo off
title Trading Bot

if not exist venv\ (
    echo Creating virtual environment...
    py -3.11 -m venv venv
    if errorlevel 1 (
        echo Python 3.11 is required. Install it, then run this script again.
        pause
        exit /b 1
    )
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)"
if errorlevel 1 (
    echo This venv is not Python 3.11. Delete the venv folder and run start.bat again.
    pause
    exit /b 1
)

echo Installing dependencies...
pip install -r requirements.txt

echo Starting Trading Bot Server...
start "" "http://localhost:8000"
python app.py
pause
