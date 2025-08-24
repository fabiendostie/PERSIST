@echo off
REM Prsist Memory System CLI - Windows Batch Wrapper
REM Usage: prsist -tsc

cd /d "%~dp0"
python prsist.py %*