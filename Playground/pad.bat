@echo off
REM ---------------------------------------------------------------------------
REM Watch what the gamepad is doing, live. Ctrl-C to stop.
REM
REM Use this when a stick does nothing, or does the wrong thing: it prints every
REM axis and every button as you move them, so the map in pad.py can be checked
REM against the pad actually plugged in.
REM ---------------------------------------------------------------------------
setlocal EnableExtensions

set "VHOME="
for /f "tokens=1,* delims==" %%a in ('findstr /b /c:"home" "%~dp0..\.venv\pyvenv.cfg"') do set "VHOME=%%b"
if "%VHOME:~0,1%"==" " set "VHOME=%VHOME:~1%"

if not defined VHOME (
  echo Could not read 'home =' from "%~dp0..\.venv\pyvenv.cfg".
  exit /b 1
)

set "PYTHONPATH=%~dp0..\.venv\Lib\site-packages;%~dp0.."

"%VHOME%\python.exe" "%~dp0pad.py" %*
exit /b %ERRORLEVEL%
