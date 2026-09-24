@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHON=%ROOT%SalsaNext-Fork\.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo ERROR: Project Python environment was not found:
    echo        %PYTHON%
    pause
    exit /b 1
)

if not exist "%ROOT%frontend\node_modules\vite" (
    echo Frontend dependencies are missing. Installing them now...
    pushd "%ROOT%frontend"
    call npm.cmd install
    if errorlevel 1 (
        echo ERROR: npm install failed.
        popd
        pause
        exit /b 1
    )
    popd
)

echo Starting LiDAR backend on http://127.0.0.1:8000 ...
start "LiDAR Backend" "%ComSpec%" /k "cd /d "%ROOT%" && "%PYTHON%" "%ROOT%backend\server.py""

timeout /t 3 /nobreak >nul

echo Starting React frontend on http://127.0.0.1:5173 ...
start "LiDAR React Frontend" "%ComSpec%" /k "cd /d "%ROOT%frontend" && npm.cmd run dev -- --host 127.0.0.1"

timeout /t 4 /nobreak >nul
start "" "http://127.0.0.1:5173/"

echo.
echo Backend:  http://127.0.0.1:8000
echo Frontend: http://127.0.0.1:5173
echo Close the two opened command windows to stop the services.

endlocal
