@echo off
REM Commodity Market Price Prediction & Analysis Platform
REM Conda Environment Setup Script for Windows

REM Set environment name
set ENV_NAME=commodity

REM Check if conda is installed
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Conda is not installed. Please install Miniconda or Anaconda first.
    exit /b 1
)

REM Create conda environment if it doesn't exist
conda info --envs | findstr /C:"%ENV_NAME%" >nul
if %ERRORLEVEL% neq 0 (
    echo Creating new conda environment: %ENV_NAME%
    conda create -y -n %ENV_NAME% python=3.9
) else (
    echo Conda environment %ENV_NAME% already exists
)

REM Activate environment
echo Activating conda environment: %ENV_NAME%
call conda activate %ENV_NAME%

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Setup project
echo Setting up project directories...
python main.py setup

echo.
echo Setup complete! You can now run the project with the following commands:
echo.
echo 1. Activate the environment:
echo    conda activate %ENV_NAME%
echo.
echo 2. Generate sample data:
echo    python main.py collect
echo.
echo 3. Train models:
echo    python main.py train
echo.
echo 4. Run the dashboard:
echo    python main.py dashboard
echo.
echo The dashboard will be available at http://localhost:8501

pause 