@echo off
REM Start Gray's dashboard.  Just type:  run
REM
REM Exists because plain "python" on Windows is usually the Microsoft Store
REM placeholder, which prints "Python was not found" instead of running anything.
REM This finds a real interpreter and prefers the project's own .venv.

setlocal

if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0run.py" %*
    goto :done
)

REM No project environment yet - fall back to the py launcher, then to python,
REM so the user at least gets run.py's own explanation of how to set one up.
where py >nul 2>nul
if %errorlevel%==0 (
    py "%~dp0run.py" %*
    goto :done
)

python "%~dp0run.py" %*
if errorlevel 1 (
    echo.
    echo Could not find Python.
    echo.
    echo   1. Install it:  winget install Python.Python.3.13
    echo   2. Then set the project up, from this folder:
    echo        uv venv --python 3.13
    echo        uv pip install -e ".[sim,tools,dev]"
    echo.
)

:done
endlocal
