@echo off
REM Activate Prsist Memory System for Claude Code
REM This script can be called automatically when Claude Code starts

echo Activating Prsist Memory System for Claude Code...

REM Run the integration script
python "%~dp0bin\claude-integration.py"

REM Set environment variable to indicate prsist is active
set PRSIST_ACTIVE=true
set PRSIST_CONTEXT_FILE=%~dp0context\claude-context.md

REM Optional: Display context file location for reference
if exist "%PRSIST_CONTEXT_FILE%" (
    echo 📄 Project context available at: %PRSIST_CONTEXT_FILE%
) else (
    echo 🔄 Context file will be created when needed
)

echo.
echo 🧠 Prsist Memory System is now active for your Claude Code session
echo.