@echo off
cd /d "%~dp0"
python -m tools.analytics.cli.main %*
pause 