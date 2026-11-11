@echo off
cd /d "%~dp0"
echo Starting AI Resume & Job Description Matcher Dashboard...
.venv\Scripts\streamlit.exe run app.py
pause

