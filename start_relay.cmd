@echo off
cd /d "%~dp0"
set HF_HUB_DISABLE_XET=1
".relay_venv\Scripts\python.exe" "scripts\relay_client.py" --repo HYHero/kroc-act-emg-checkpoints
pause
