@echo off
REM Navigate to the dependencies directory and run setup.py
cd /d %~dp0dependencies
python setup.py

REM Check if the first script was successful
if %ERRORLEVEL% NEQ 0 (
    echo Failed to run setup.py
    exit /b %ERRORLEVEL%
)

REM Navigate to the gui directory and run main.py
cd /d %~dp0gui
python main.py

REM Check if the second script was successful
if %ERRORLEVEL% NEQ 0 (
    echo Failed to run main.py
    exit /b %ERRORLEVEL%
)

REM End of script
echo Scripts executed successfully.
