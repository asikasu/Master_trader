@echo off
cd /d "%~dp0"
call "%~dp0.venv\Scripts\activate.bat"

:loop
echo [+] %DATE% %TIME% Starting bot...
call python tournament_bot/main.py --mode LIVE
set EXIT_CODE=%ERRORLEVEL%

echo [!] %DATE% %TIME% Bot stopped (exit=%EXIT_CODE%). Restarting in 5s...
timeout /t 5 /nobreak >nul
goto loop
