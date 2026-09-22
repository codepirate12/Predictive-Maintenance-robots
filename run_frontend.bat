@echo off
echo Starting Predictive Maintenance Frontend Web Server...
python -m http.server 5500 --directory frontend
pause
