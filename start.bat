@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

:MENU
cls
echo   ================================================
echo       FlappyAI - NEAT training dashboard
echo   ================================================
echo.
echo   1. Install Python dependencies (chay 1 lan)
echo   2. Train NEAT (100 generations)
echo   3. Xem AI choi lai (Pygame window)
echo   4. Mo web dashboard (http://127.0.0.1:8765)
echo   0. Exit
echo.
set /p "choice=Chon so: "

if "%choice%"=="1" goto INSTALL
if "%choice%"=="2" goto TRAIN
if "%choice%"=="3" goto REPLAY
if "%choice%"=="4" goto WEB
if "%choice%"=="0" exit /b

echo.
echo Nhap sai roi. Nhap 1, 2, 3, 4 hoac 0.
pause
goto MENU

:INSTALL
echo.
echo Dang cai dat neat-python + pygame...
echo.
python -m pip install -r requirements.txt
echo.
echo Xong! Nhan phim bat ky de quay ve menu.
pause
goto MENU

:TRAIN
echo.
if not exist logs\ mkdir logs
echo Dang train NEAT trong %1 generations...
echo Mac dinh: 100 generations, ~5-15 phut.
echo.
python train.py
echo.
if %errorlevel% neq 0 (
    echo Loi! Kiem tra lai Python da cai dat chua.
) else (
    echo Xong! Winner saved to logs\winner.pkl
)
pause
goto MENU

:REPLAY
echo.
if not exist logs\winner.pkl (
    echo Chua co winner. Train truoc da (option 2).
    pause
    goto MENU
)
python replay.py
pause
goto MENU

:WEB
echo.
echo Mo trinh duyet: http://127.0.0.1:8765
echo Nhan Ctrl+C trong cua so nay de tat.
echo.
python server.py
pause
goto MENU
