@echo off
echo Starting Predictive Maintenance FastAPI Backend Server...
python -m uvicorn backend.main:app --reload --port 8000
pause
