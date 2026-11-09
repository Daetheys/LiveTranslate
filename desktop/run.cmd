@echo off
rem Double-click or run from any terminal. Windows blocks .ps1 scripts by default; this bypasses it for run.ps1 only.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
