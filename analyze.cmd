@echo off
REM Analyse one saved clip: spectrum chart, gain chart, blower on/off, filter verdict.
REM
REM   analyze "unknown 504 - 20261006_163505"
REM   analyze "BLOWER ON - 20261004_203111.wav" --csv out.csv
REM
REM The filename alone is enough -- analyze_clip.py searches the data folders
REM for it and infers the device from where it finds it. Any extra flags are
REM passed straight through.
setlocal
set "ML=%~dp0"
"%ML%venv\Scripts\python.exe" "%ML%analyze_clip.py" %*
