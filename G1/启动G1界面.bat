@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -m G1_Desktop.app %*
if errorlevel 1 pause
