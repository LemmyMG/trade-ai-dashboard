@echo off

echo Starting Backend Server...
start cmd /k "cd /d %~dp0 && uvicorn app:app --reload"

timeout /t 3

echo Starting Frontend Dashboard...
start cmd /k "cd /d %~dp0 && streamlit run frontend.py"