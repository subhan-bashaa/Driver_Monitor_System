@echo off
REM Windows batch script to run the Driver Monitoring System

echo ================================================
echo  Driver Monitoring System - Launcher
echo ================================================
echo.

REM Check if virtual environment exists
if not exist "venv312\" (
    echo Virtual environment not found!
    echo Please run: python -m venv venv312
    echo Then run: venv312\Scripts\activate
    echo Then run: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
call venv312\Scripts\activate

REM Check if alarm.wav exists
if not exist "assets\alarm.wav" (
    echo Alarm sound not found. Generating...
    python generate_alarm.py
    echo.
)

REM Run the main program
echo Starting Driver Monitoring System...
echo Press 'q' to quit
echo.
python src\main.py

REM Deactivate virtual environment
deactivate

pause
