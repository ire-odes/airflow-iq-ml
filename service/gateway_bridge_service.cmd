@echo off
REM Keeps service/gateway_bridge.py running for as long as the workstation is up.
REM
REM The retry loop is the point. The bridge exits whenever COM18 is not there --
REM the gateway unplugged, or USB not yet enumerated at boot -- and a device
REM behind the gateway gets 504s for every second the bridge is down. Restarting
REM every 15 s means an unplug-replug heals itself instead of needing someone to
REM notice. `ping` is the sleep: `timeout` needs a console and Task Scheduler
REM does not give it one.
REM
REM PORT: COM18 is the gateway's native-USB port. Windows keeps that number tied
REM to the board, so it survives replugging into the same machine. If the
REM gateway is ever swapped for a different board, change it here.

set "ML=C:\Users\otc.student\Desktop\afiq development\ML"
set "LOG=%ML%\logs\gateway_bridge.log"
set "PORT=COM18"

cd /d "%ML%"
if not exist "logs" mkdir "logs"

:loop
echo. >> "%LOG%"
echo ===== bridge starting %DATE% %TIME% on %PORT% ===== >> "%LOG%"
"%ML%\venv\Scripts\python.exe" -u service\gateway_bridge.py --port %PORT% >> "%LOG%" 2>&1
echo ===== bridge exited (code %ERRORLEVEL%) at %DATE% %TIME%, retrying in 15 s ===== >> "%LOG%"
ping -n 16 127.0.0.1 >nul
goto loop
