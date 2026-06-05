@echo off
cd /d %~dp0..\..
echo Watchdog started at %date% %time%
echo Monitoring orchestrator every 5 minutes...

:loop
tasklist /FI "WINDOWTITLE eq lstm_overnight" 2>nul | find "python" >nul
if errorlevel 1 (
    echo [%date% %time%] Orchestrator dead, restarting...
    start "lstm_overnight" /MIN py -3.13 cli\overnight\orchestrator.py
)
timeout /t 300 /nobreak >nul
goto loop
