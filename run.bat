@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python tournament_bot/main.py %*
