@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "G1_DEEPSEEK_G1_ROOT=%~dp0G1_LLM_Robot\G1_LLM_Robot"
set "G1_DEEPSEEK_SUPERMIMIC_ROOT=%~dp0..\..\SuperMimic"
set "G1_DEEPSEEK_RAX_ROOT=%~dp0..\..\RAX"
set "PYTHONUTF8=1"

where python >nul 2>nul || (
  echo Python was not found. Install Python or add it to PATH.
  pause
  exit /b 1
)

echo Starting G1 DeepSeek with MuJoCo and research tools...
python "%~dp0Start_G1_Unified.py" %*
set "CODE=%ERRORLEVEL%"
if not "%CODE%"=="0" (
  echo.
  echo G1 exited with code %CODE%.
  pause
)
exit /b %CODE%
