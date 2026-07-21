@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat

:loop
echo [+] Starting bot at %DATE% %TIME%
call python tournament_bot/main.py %*
set EXIT_CODE=%ERRORLEVEL%

echo [!] Bot stopped with exit code %EXIT_CODE%. Restarting in 5s...
timeout /t 5 /nobreak >nul
goto loop
