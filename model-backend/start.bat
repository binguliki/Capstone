@echo off
REM Quick start script for Leaf Disease Detection API on Windows

echo 🌿 Leaf Disease Detection API - Quick Start
echo ===========================================
echo.

REM Check Python version
echo [1/5] Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.9 or higher
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✓ Python %PYTHON_VERSION%
echo.

REM Create virtual environment
echo [2/5] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo ✓ Virtual environment created
) else (
    echo ✓ Virtual environment already exists
)
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo [3/5] Installing dependencies...
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo ✓ Dependencies installed
echo.

REM Check model file
echo [4/5] Checking model file...
if not exist "modelv1.keras" (
    echo ⚠️  modelv1.keras not found in current directory
    echo Please ensure modelv1.keras is in the model-backend directory
) else (
    for %%A in (modelv1.keras) do set FILE_SIZE=%%~zA
    echo ✓ Model found (size: %FILE_SIZE% bytes)
)
echo.

REM Setup environment
echo [5/5] Setting up environment...
if not exist ".env" (
    copy .env.example .env
    echo ✓ .env file created
) else (
    echo ✓ .env file already exists
)
echo.

REM Start server
echo.
echo ========================================
echo ✓ Setup complete!
echo ========================================
echo.
echo Starting Leaf Disease Detection API...
echo.
echo 📊 API Documentation:
echo    • Swagger UI: http://localhost:8000/docs
echo    • ReDoc: http://localhost:8000/redoc
echo.
echo 🔗 Endpoints:
echo    • POST   /predict  - Disease detection
echo    • GET    /health   - Health check
echo    • GET    /metrics  - Performance metrics
echo.
echo Press Ctrl+C to stop the server
echo.

python main.py
pause
