@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat

:loop
echo [+] Starting bot...
python tournament_bot/main.py %*

echo [!] Bot stopped. Restarting in 5s...
timeout /t 5 /nobreak >nul
goto loop