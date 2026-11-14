@echo off
setlocal
REM Start Label Studio (local files + public URL in CSRF trusted origins)
cd /d "%~dp0"
uv run radar ls-serve --background --public-url "http://121.196.206.100,http://121.196.206.100:8000"
pause