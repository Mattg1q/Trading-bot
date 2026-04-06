@echo off
title Trading Bot

if not exist venv\ (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo Starting Trading Bot Server...
python app.py
pause
