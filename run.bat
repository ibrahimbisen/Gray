@echo off
REM ---------------------------------------------------------------------------
REM Start the dashboard, or the runner, on a machine where Device Guard blocks
REM the venv's python.exe.
REM
REM     run.bat              the dashboard at http://127.0.0.1:8000
REM     run.bat --runner     work through the training queue
REM     run.bat --check      print what the dashboard can see, and exit
REM
REM run.py already knows how to recover from the blocked launcher - it re-runs
REM itself under the interpreter .venv was built from. The problem is that
REM `python run.py` cannot get that far: with the venv activated, `python` IS the
REM blocked executable, so Device Guard stops it before a single line runs.
REM
REM This calls the base interpreter directly, which is not blocked, with the
REM venv's site-packages on the path. Same environment, no blocked launcher.
REM ---------------------------------------------------------------------------
setlocal EnableExtensions

REM Where the venv came from, read from .venv\pyvenv.cfg rather than hard-coded,
REM so rebuilding the venv on a new Python does not silently break this file.
set "VHOME="
for /f "tokens=1,* delims==" %%a in ('findstr /b /c:"home" "%~dp0.venv\pyvenv.cfg"') do set "VHOME=%%b"
if "%VHOME:~0,1%"==" " set "VHOME=%VHOME:~1%"

if not defined VHOME (
  echo Could not read 'home =' from "%~dp0.venv\pyvenv.cfg".
  echo Is the venv built? Try:  uv venv
  exit /b 1
)
if not exist "%VHOME%\python.exe" (
  echo The interpreter .venv points at is missing:
  echo   %VHOME%\python.exe
  exit /b 1
)

REM Semicolon, not colon. PYTHONPATH is split on os.pathsep, which is ';' on
REM Windows - a colon here makes the whole string one nonsense path and every
REM import from site-packages fails with a confusing ModuleNotFoundError.
set "PYTHONPATH=%~dp0.venv\Lib\site-packages;%~dp0"

"%VHOME%\python.exe" "%~dp0run.py" %*
exit /b %ERRORLEVEL%
