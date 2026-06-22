@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

:MENU
cls
echo   ================================================
echo       FlappyAI - NEAT training dashboard
echo   ================================================
echo.
echo   1. Install Python dependencies
echo   2. Train NEAT (100 generations, ~5-15 min)
echo   3. Replay best bird (Pygame window)
echo   4. Open web dashboard (http://127.0.0.1:8765/)
echo   5. Train + auto-open dashboard
echo   0. Exit
echo.
set /p choice=Select:

if "%choice%"=="1" goto INSTALL
if "%choice%"=="2" goto TRAIN
if "%choice%"=="3" goto REPLAY
if "%choice%"=="4" goto WEB
if "%choice%"=="5" goto TRAINWEB
if "%choice%"=="0" exit /b

echo Invalid choice.
pause
goto MENU

:INSTALL
echo Installing neat-python + pygame...
python -m pip install -r requirements.txt
echo Done.
pause
goto MENU

:TRAIN
if not exist logs mkdir logs
python train.py %2
if errorlevel 1 (
  echo Training failed.
) else (
  echo Winner saved to logs\winner.pkl
)
pause
goto MENU

:REPLAY
if not exist logs\winner.pkl (
  echo No winner found. Run training first (option 2).
  pause
  goto MENU
)
python replay.py
goto MENU

:WEB
python server.py
goto MENU

:TRAINWEB
start "flappyai-train" cmd /c "python train.py"
echo Training started in a separate window.
echo When winner is saved, run option 4 to open the dashboard.
pause
goto MENU