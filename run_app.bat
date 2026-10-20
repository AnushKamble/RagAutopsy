@echo off
cd /d "%~dp0"
echo Starting RAG-Autopsy Interactive Studio...
"%~dp0.venv\Scripts\streamlit.exe" run "%~dp0app.py"
