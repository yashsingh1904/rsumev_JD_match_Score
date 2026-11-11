# 🎯 AI Resume & Skill Gap Analyzer

An enterprise-grade, modern recruitment intelligence web application powered by Groq LLM and Pydantic. It allows users to upload any resume and paste any target job description to compute an **objective match score**, perform a **deep skill gap analysis**, and provide **actionable hiring recommendations**.

---

## ✨ Features & Capabilities

- **Secure & Production-Ready**:
  - API keys and LLM model configurations are managed strictly server-side through environment variables (never exposed in the UI or client browser).
- **Direct Input & Universal Ingestion**:
  - **Job Description**: Clean, dedicated text area for pasting any job description with real-time word/character count.
  - **Candidate Resume**: Drag-and-drop file uploader supporting **PDF**, **DOCX**, and **TXT** files, or paste raw text.
- **Executive Visual Dashboard**:
  - **Hero Match Banner**: Shows candidate name, applied role, compatibility status badge, and an animated colored score bar.
  - **4 KPI Metric Tiles**:
    1. Overall Match Score (%)
    2. Skills Coverage (Matched vs Total Required)
    3. Experience Verification (Candidate Experience vs Minimum Required)
    4. Education Fit Status
  - **🧩 Deep Skill Gap & Compatibility Matrix**:
    - 🟢 **Matched Skills**: Highlighted emerald tags with checkmarks (`✓ Python`, `✓ AWS`, `✓ Git`).
    - 🔴 **Missing Skills (The Critical Gap)**: Highlighted crimson/rose warning tags (`✗ Kubernetes`, `✗ CI/CD`) pinpointing competencies the candidate lacks.
    - 💡 **Bonus Qualifications**: Purple tags showcasing additional candidate strengths not required by the JD.
  - **🧠 AI Recruiter Intelligence**:
    - **Executive Hiring Verdict**: High-level recruiter summary.
    - **Core Strengths**: Standout candidate highlights.
    - **Actionable Recommendations**: Specific advice on how to bridge the skill gap.
  - **📂 Structured Profile Accordion**: Expandable view of extracted candidate work history, internships, education, and projects.
  - **📥 Instant Export & Reset**: One-click download of the complete assessment report as JSON, plus a reset button to evaluate another candidate.

---

## 🚀 Running the App

Start the application with Streamlit:

```bash
cd c:\dev\AI_PROJECT\JD_RESUME__matchScore
.venv\Scripts\streamlit.exe run app.py
```

Or double-click:
👉 **[`run_app.bat`](run_app.bat)**

Access the dashboard at:
👉 **`http://localhost:8501`**

---

## 🔒 Server Configuration

Configure your Groq API key in `.env` or in your production deployment environment:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```
*(No API keys or model controls are shown to users in the browser).*
