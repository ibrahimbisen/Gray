@echo off
REM ---------------------------------------------------------------------------
REM Drive a trained policy with the keyboard.
REM
REM     Playground\pilot.bat                the newest walk run
REM     Playground\pilot.bat --run 25       a run by number, name, or folder
REM
REM Same launcher trick as run.bat, and for the same reason: Device Guard on
REM this machine refuses to run .venv\Scripts\python.exe, so the interpreter the
REM venv was built from is called directly with the venv's packages on the path.
REM ---------------------------------------------------------------------------
setlocal EnableExtensions

set "VHOME="
for /f "tokens=1,* delims==" %%a in ('findstr /b /c:"home" "%~dp0..\.venv\pyvenv.cfg"') do set "VHOME=%%b"
if "%VHOME:~0,1%"==" " set "VHOME=%VHOME:~1%"

if not defined VHOME (
  echo Could not read 'home =' from "%~dp0..\.venv\pyvenv.cfg".
  echo Is the venv built? Try:  uv venv
  exit /b 1
)
if not exist "%VHOME%\python.exe" (
  echo The interpreter .venv points at is missing:
  echo   %VHOME%\python.exe
  exit /b 1
)

REM Semicolon, not colon - PYTHONPATH is split on ';' on Windows.
set "PYTHONPATH=%~dp0..\.venv\Lib\site-packages;%~dp0.."

"%VHOME%\python.exe" "%~dp0pilot.py" %*
exit /b %ERRORLEVEL%
