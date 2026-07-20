@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
:loop
python tournament_bot/main.py %*
if errorlevel 1 (
    echo [!] Bot crashed. Restarting in 5s...
    timeout /t 5 /nobreak >nul
    goto loop
)
echo Bot finished normally.
