@echo off
rem --- change to the folder where this batch file lives
cd /d %~dp0

rem --- activate the venv
call venv\Scripts\activate.bat

rem --- run the scraper; log output for debugging
python EV_Comp_Checker.py >> ev_scraper.log 2>&1
