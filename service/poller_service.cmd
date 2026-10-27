@echo off
REM Runs the inference poller with its output captured.
REM
REM WHY THIS EXISTS. The poller prints the blower on/off verdict for every clip:
REM   {mac}: blower off (-1.4 dB absolute, no baseline yet (cut +2.5)) -- marking processed
REM   {mac}: blower on (+6.0 dB vs its own baseline of 43 clips)
REM A gated clip writes NO row to filter_ml_readings, so without this redirect
REM that verdict existed only in a console nobody was attached to, and the only
REM way to answer "was the blower on at 3am" was to infer it from the absence of
REM a row. Task Scheduler cannot redirect on its own, hence the wrapper.
REM
REM The retry loop covers an outright crash; --loop 10 already keeps the poller
REM alive across ordinary errors.

set "ML=C:\Users\otc.student\Desktop\afiq development\ML"
set "LOG=%ML%\logs\poller.log"

cd /d "%ML%"
if not exist "logs" mkdir "logs"

:loop
echo. >> "%LOG%"
echo ===== poller starting %DATE% %TIME% ===== >> "%LOG%"
"%ML%\venv\Scripts\python.exe" -u service\poll_and_infer.py --loop 10 >> "%LOG%" 2>&1
echo ===== poller exited (code %ERRORLEVEL%) at %DATE% %TIME%, retrying in 15 s ===== >> "%LOG%"
ping -n 16 127.0.0.1 >nul
goto loop
