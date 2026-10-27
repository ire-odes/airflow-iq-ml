@echo off
REM One-shot wrapper for the 2026-09-24 20:30 P3 recalibration.
REM Exists so the scheduled run leaves evidence: Task Scheduler reports an exit
REM code but not what the script actually said, and a recalibration that half
REM ran is worth being able to see after the fact.
cd /d "C:\Users\otc.student\Desktop\ML"
echo. >> "logs\recalibrate_p3.log"
echo ===== run started %DATE% %TIME% ===== >> "logs\recalibrate_p3.log"
"C:\Users\otc.student\Desktop\ML\venv\Scripts\python.exe" service\recalibrate_all.py --yes --mac 58E6C5204B28 >> "logs\recalibrate_p3.log" 2>&1
echo ===== exit code %ERRORLEVEL% at %DATE% %TIME% ===== >> "logs\recalibrate_p3.log"
