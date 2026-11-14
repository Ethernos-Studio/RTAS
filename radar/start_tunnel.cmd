@echo off
setlocal
REM Start frpc: local 8080 -> 121.196.206.100:8000  (nginx maps bare IP :80 to 8000)
set "FRP_DIR=%~dp0tools\frp_0.62.1_windows_amd64"
if not exist "%FRP_DIR%\frpc.exe" (
  echo [error] frpc.exe not found: "%FRP_DIR%"
  pause
  exit /b 1
)
echo Starting frpc from %FRP_DIR%
frpc.exe -c "%FRP_DIR%\frpc-radar.toml"
pause