@echo off
title NrityaVision AI Launcher
echo -----------------------------------------
echo 🪷 Starting NrityaVision AI...
echo -----------------------------------------
cd /d %~dp0
.\.venv\Scripts\python.exe app.py
pause
