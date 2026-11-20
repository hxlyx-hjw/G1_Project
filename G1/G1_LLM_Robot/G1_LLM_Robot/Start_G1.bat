@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>nul || (
  echo Python was not found.
  pause
  exit /b 1
)
python Start.py %*
set "CODE=%ERRORLEVEL%"
if not "%CODE%"=="0" echo Program exited with code %CODE%.
pause
exit /b %CODE%
