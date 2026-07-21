@echo off
cd /d "%~dp0"

if "%1"=="install" goto install
if "%1"=="start" goto start
if "%1"=="stop" goto stop
if "%1"=="remove" goto remove
if "%1"=="" (
    echo Usage: %0 {install^|start^|stop^|remove}
    exit /b 1
)

:install
echo [+] Installing GoldBot Windows Service...
nssm install GoldBot "%~dp0run_nssm.bat"
echo [+] Set working directory...
nssm set GoldBot AppDirectory "%~dp0"
echo [+] Set automatic start...
nssm set GoldBot Start SERVICE_AUTO_START
echo [+] Set stdout log...
nssm set GoldBot AppStdout "%~dp0nssm_stdout.log"
echo [+] Set stderr log...
nssm set GoldBot AppStderr "%~dp0nssm_stderr.log"
echo [+] Set restart on crash...
nssm set GoldBot AppThrottle 0
nssm set GoldBot AppExit Default Exit
echo [+] GoldBot service installed.
nssm start GoldBot
goto end

:start
nssm start GoldBot
goto end

:stop
nssm stop GoldBot
goto end

:remove
nssm stop GoldBot
nssm remove GoldBot confirm
goto end

:end
