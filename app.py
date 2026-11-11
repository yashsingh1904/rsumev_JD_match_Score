import io
import json
import os
import time
from pathlib import Path

import docx
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field
import pypdf
import streamlit as st

# ---------------------------------------------------------
# Environment & Server-side Configuration
# ---------------------------------------------------------
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH)

# Groq API credentials are handled strictly server-side (never exposed in the UI)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


# ---------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------
class JobDescriptionData(BaseModel):
    role: str
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    minimum_experience: float | None = None
    education_requirements: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)


class ExperienceItem(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = Field(default_factory=list)


class ResumeData(BaseModel):
    name: str | None = "Candidate"
    email: str | None = None
    phone: str | None = None
    total_experience_years: float | None = None
    skills: list[str] = Field(default_factory=list)
    experiences: list[ExperienceItem] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


class MatchAnalysis(BaseModel):
    candidate_name: str | None = None
    score: float = Field(..., description="Overall match percentage from 0 to 100")
    matching_skills: list[str] = Field(default_factory=list, description="Skills present in both JD and Resume")
    missing_skills: list[str] = Field(default_factory=list, description="Crucial JD skills missing in Resume (Skill Gap)")
    additional_skills: list[str] = Field(default_factory=list, description="Candidate bonus skills not in JD")
    experience_met: bool = Field(False, description="True if candidate meets or exceeds required experience")
    experience_notes: str | None = Field("", description="Comparison of candidate exp vs required exp")
    education_met: bool = Field(True, description="True if education requirements appear satisfied")
    education_notes: str | None = Field("", description="Education match details")
    verdict: str = Field(..., description="Crisp executive evaluation summary")
    strengths: list[str] = Field(default_factory=list, description="Key candidate strengths for this role")
    recommendations: list[str] = Field(default_factory=list, description="Specific recommendations to close skill gaps")


# ---------------------------------------------------------
# File Ingestion Helpers
# ---------------------------------------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF byte content."""
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts).strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX byte content."""
    doc = docx.Document(io.BytesIO(file_bytes))
    text_parts = []
    for p in doc.paragraphs:
        if p.text.strip():
            text_parts.append(p.text)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    text_parts.append(cell.text)
    return "\n".join(text_parts).strip()


def extract_text(file_obj, filename: str) -> str:
    """Extract text from an uploaded file or file-like object."""
    file_bytes = file_obj.read() if hasattr(file_obj, "read") else file_obj
    name_lower = filename.lower()
    if name_lower.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif name_lower.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    elif name_lower.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported format: {filename}. Please upload a PDF, DOCX, or TXT file.")


# ---------------------------------------------------------
# LLM Core Logic
# ---------------------------------------------------------
def call_groq_json(client: Groq, model: str, system_prompt: str, user_prompt: str) -> dict:
    """Helper to call Groq with JSON object response format."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    return json.loads(content)


def parse_job_description(client: Groq, model: str, jd_text: str) -> JobDescriptionData:
    schema = JobDescriptionData.model_json_schema()
    system_prompt = f"""You are an expert HR recruitment specialist.
Analyze the job description and extract structured requirements into a valid JSON object matching this schema:
{json.dumps(schema, indent=2)}

Rules:
1. Return ONLY valid JSON adhering strictly to the schema.
2. If minimum experience is not explicitly mentioned, return null.
3. If a list is empty or not mentioned, return [].
4. Do not invent facts."""

    user_prompt = f"Extract structured requirements from this Job Description:\n\n{jd_text}"
    data = call_groq_json(client, model, system_prompt, user_prompt)
    return JobDescriptionData(**data)


def parse_resume(client: Groq, model: str, resume_text: str) -> ResumeData:
    schema = ResumeData.model_json_schema()
    system_prompt = f"""You are an expert resume parser.
Extract comprehensive candidate details from the provided resume text into a valid JSON object matching this schema:
{json.dumps(schema, indent=2)}

Rules:
1. Return ONLY valid JSON adhering to the schema.
2. If total experience years is not stated, calculate or estimate it from work history, or null.
3. Extract all programming languages, tools, frameworks, and domains into 'skills'.
4. Include internships within experiences.
5. Do not invent information."""

    user_prompt = f"Parse this resume:\n\n{resume_text}"
    data = call_groq_json(client, model, system_prompt, user_prompt)
    return ResumeData(**data)


def evaluate_match(client: Groq, model: str, job: JobDescriptionData, resume: ResumeData) -> MatchAnalysis:
    schema = MatchAnalysis.model_json_schema()
    system_prompt = f"""You are a senior technical hiring manager and talent evaluator.
Compare the Candidate Resume with the Job Description to perform a rigorous compatibility and gap analysis.
Return ONLY valid JSON matching this schema:
{json.dumps(schema, indent=2)}

Evaluation Criteria:
1. 'score': Objective match score from 0 to 100 based on technical skills overlap, relevant experience, and role alignment.
2. 'matching_skills': List of required/preferred skills from the JD that the candidate demonstrates.
3. 'missing_skills': Critical skills from the JD that are MISSING in the candidate's resume (Skill Gap).
4. 'additional_skills': Candidate's other valuable technical skills not requested in the JD.
5. 'experience_met': True if candidate meets or exceeds the minimum experience required.
6. 'experience_notes': Comparison note of candidate experience vs required experience.
7. 'education_met': True if candidate qualifications satisfy JD education requirements.
8. 'education_notes': Brief explanation of education compatibility.
9. 'verdict': Crisp 2-3 sentence executive evaluation.
10. 'strengths': 3-5 bullet points of where candidate shines.
11. 'recommendations': 2-4 actionable suggestions on what candidate should learn or improve to achieve a 100% fit."""

    user_prompt = f"""JOB REQUIREMENTS:
{job.model_dump_json(indent=2)}

CANDIDATE RESUME:
{resume.model_dump_json(indent=2)}

Analyze the match and provide the JSON assessment:"""

    data = call_groq_json(client, model, system_prompt, user_prompt)
    return MatchAnalysis(**data)


# ---------------------------------------------------------
# Streamlit Application
# ---------------------------------------------------------
def main():
    st.set_page_config(
        page_title="AI Resume & Skill Gap Matcher",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Enterprise Dark Theme & Visual Styling
    st.markdown("""
    <style>
        /* Base typography & backgrounds */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        
        * {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Glassmorphism Containers */
        .glass-panel {
            background: linear-gradient(145deg, rgba(20, 27, 45, 0.85), rgba(13, 18, 30, 0.95));
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
            margin-bottom: 24px;
            backdrop-filter: blur(12px);
        }

        .hero-banner {
            background: radial-gradient(circle at top right, rgba(99, 102, 241, 0.15), transparent 45%),
                        radial-gradient(circle at bottom left, rgba(16, 185, 129, 0.12), transparent 40%),
                        linear-gradient(135deg, #131b2e 0%, #0d121f 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 32px 36px;
            margin-bottom: 28px;
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6);
        }

        /* KPI Stat Cards */
        .kpi-card {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 20px;
            text-align: center;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            border-color: rgba(99, 102, 241, 0.4);
        }

        .kpi-label {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94a3b8;
            margin-bottom: 8px;
        }

        .kpi-val {
            font-size: 2rem;
            font-weight: 800;
            color: #ffffff;
            margin: 4px 0;
        }

        .kpi-sub {
            font-size: 0.85rem;
            color: #cbd5e1;
        }

        /* Badges & Tags */
        .skill-badge {
            display: inline-flex;
            align-items: center;
            padding: 7px 15px;
            border-radius: 10px;
            font-size: 0.88rem;
            font-weight: 600;
            margin: 5px 4px;
            letter-spacing: 0.01em;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15);
        }

        .skill-matched {
            background: rgba(16, 185, 129, 0.15);
            color: #6ee7b7;
            border: 1px solid rgba(16, 185, 129, 0.5);
        }

        .skill-gap {
            background: rgba(244, 63, 94, 0.15);
            color: #fda4af;
            border: 1px solid rgba(244, 63, 94, 0.5);
        }

        .skill-bonus {
            background: rgba(139, 92, 246, 0.15);
            color: #c4b5fd;
            border: 1px solid rgba(139, 92, 246, 0.5);
        }

        /* Status Pills */
        .status-pill {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .status-pill-high {
            background: rgba(16, 185, 129, 0.2);
            color: #34d399;
            border: 1px solid #10b981;
        }
        .status-pill-med {
            background: rgba(245, 158, 11, 0.2);
            color: #fbbf24;
            border: 1px solid #f59e0b;
        }
        .status-pill-low {
            background: rgba(244, 63, 94, 0.2);
            color: #fb7185;
            border: 1px solid #f43f5e;
        }

        /* Progress Bar */
        .progress-container {
            background: #1e293b;
            border-radius: 9999px;
            height: 14px;
            width: 100%;
            overflow: hidden;
            margin-top: 14px;
        }
        .progress-fill-high {
            background: linear-gradient(90deg, #10b981, #34d399);
            height: 100%;
            border-radius: 9999px;
        }
        .progress-fill-med {
            background: linear-gradient(90deg, #f59e0b, #fbbf24);
            height: 100%;
            border-radius: 9999px;
        }
        .progress-fill-low {
            background: linear-gradient(90deg, #f43f5e, #fb7185);
            height: 100%;
            border-radius: 9999px;
        }

        /* Callout Box */
        .verdict-box {
            background: rgba(15, 23, 42, 0.6);
            border-left: 4px solid #6366f1;
            padding: 18px 22px;
            border-radius: 0 12px 12px 0;
            font-size: 1.02rem;
            line-height: 1.6;
            color: #e2e8f0;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Server-side API key validation
    if not GROQ_API_KEY:
        st.error(
            "⚠️ **Server Configuration Error**: `GROQ_API_KEY` is not set in the server environment. "
            "Please configure `GROQ_API_KEY` in your `.env` file or deployment settings."
        )
        st.stop()

    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize Groq Client: {e}")
        st.stop()

    # ---------------------------------------------------------
    # Header Section
    # ---------------------------------------------------------
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 25px 0;">
        <div style="display: inline-block; padding: 6px 16px; border-radius: 9999px; background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.4); color: #a5b4fc; font-size: 0.85rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 14px;">
            AI Recruitment & Talent Intelligence
        </div>
        <h1 style="font-size: 2.8rem; font-weight: 800; background: linear-gradient(90deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 12px 0;">
            Resume & Skill Gap Analyzer
        </h1>
        <p style="font-size: 1.12rem; color: #94a3b8; max-width: 760px; margin: 0 auto; line-height: 1.6;">
            Upload any resume and paste the target job description to get an <b>objective match score</b>, identify <b>critical skill gaps</b>, and receive <b>instant hiring recommendations</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Input Cards (Job Description & Resume Upload)
    # ---------------------------------------------------------
    col_jd, col_res = st.columns([1, 1], gap="large")

    with col_jd:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
            <span style="font-size: 1.4rem;">📋</span>
            <span style="font-size: 1.2rem; font-weight: 700; color: #f8fafc;">Job Description</span>
        </div>
        """, unsafe_allow_html=True)

        jd_text = st.text_area(
            label="Job Description Input",
            placeholder="Paste the target job description here...\n\nInclude:\n• Role title & team overview\n• Core responsibilities\n• Required technical skills & qualifications\n• Preferred qualifications & experience requirements",
            height=320,
            label_visibility="collapsed"
        )
        if jd_text:
            word_count = len(jd_text.split())
            st.caption(f"📝 {word_count} words | {len(jd_text)} characters")

    with col_res:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
            <span style="font-size: 1.4rem;">📄</span>
            <span style="font-size: 1.2rem; font-weight: 700; color: #f8fafc;">Candidate Resume</span>
        </div>
        """, unsafe_allow_html=True)

        input_mode = st.radio(
            "Resume Input Method",
            ["Upload File (PDF / DOCX / TXT)", "Paste Resume Text"],
            horizontal=True,
            label_visibility="collapsed"
        )

        resume_text = ""
        resume_filename = "Candidate Resume"

        if input_mode == "Upload File (PDF / DOCX / TXT)":
            uploaded_file = st.file_uploader(
                label="Upload Resume",
                type=["pdf", "docx", "txt"],
                help="Select a candidate resume in PDF, Word (DOCX), or plain text format.",
                label_visibility="collapsed"
            )
            if uploaded_file is not None:
                resume_filename = uploaded_file.name
                try:
                    resume_text = extract_text(uploaded_file, resume_filename)
                    st.success(f"✅ **Loaded `{resume_filename}`** ({len(resume_text.split())} words extracted)")
                except Exception as e:
                    st.error(f"Error parsing file: {e}")
        else:
            resume_text = st.text_area(
                label="Paste Resume Text",
                placeholder="Paste the full candidate resume text here (Work history, education, skills, projects)...",
                height=260,
                label_visibility="collapsed"
            )
            if resume_text:
                st.caption(f"📝 {len(resume_text.split())} words detected")

        if resume_text:
            with st.expander("👁️ Preview Extracted Resume Text"):
                st.text(resume_text[:1000] + ("..." if len(resume_text) > 1000 else ""))

    # ---------------------------------------------------------
    # Action Trigger Button
    # ---------------------------------------------------------
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    analyze_btn = st.button("⚡ Run Compatibility & Gap Analysis", type="primary", use_container_width=True)

    # ---------------------------------------------------------
    # Evaluation Execution & Results Dashboard
    # ---------------------------------------------------------
    if analyze_btn:
        if not jd_text.strip():
            st.warning("⚠️ Please provide a Job Description to compare against.")
        elif not resume_text.strip():
            st.warning("⚠️ Please upload or paste a Candidate Resume before proceeding.")
        else:
            progress_bar = st.progress(0, text="Initializing AI Talent Engine...")

            try:
                # Step 1: Parse JD
                progress_bar.progress(30, text="🤖 Step 1/3: Extracting structured requirements from Job Description...")
                job_parsed = parse_job_description(groq_client, MODEL_NAME, jd_text)

                # Step 2: Parse Resume
                progress_bar.progress(65, text="📄 Step 2/3: Parsing candidate background, skills, and work history...")
                resume_parsed = parse_resume(groq_client, MODEL_NAME, resume_text)

                # Step 3: Match & Gap Analysis
                progress_bar.progress(90, text="🧠 Step 3/3: Running deep skill gap analysis & candidate scoring...")
                match_result = evaluate_match(groq_client, MODEL_NAME, job_parsed, resume_parsed)

                progress_bar.progress(100, text="✨ Analysis complete!")
                time.sleep(0.4)
                progress_bar.empty()

                # Store result in session state to persist display
                st.session_state["analysis_done"] = True
                st.session_state["job_parsed"] = job_parsed
                st.session_state["resume_parsed"] = resume_parsed
                st.session_state["match_result"] = match_result

            except Exception as ex:
                st.error(f"❌ Evaluation error occurred: {ex}")

    # Render results if available in session_state
    if st.session_state.get("analysis_done"):
        job_parsed = st.session_state["job_parsed"]
        resume_parsed = st.session_state["resume_parsed"]
        match_result = st.session_state["match_result"]

        score = match_result.score
        candidate_display_name = resume_parsed.name or match_result.candidate_name or "Candidate"
        role_display_name = job_parsed.role or "Target Role"

        # Dynamic color & status styling based on score
        if score >= 75:
            score_color = "#10b981"
            fill_class = "progress-fill-high"
            status_html = '<span class="status-pill status-pill-high">⭐ Strong Match</span>'
        elif score >= 50:
            score_color = "#f59e0b"
            fill_class = "progress-fill-med"
            status_html = '<span class="status-pill status-pill-med">⚡ Moderate Match</span>'
        else:
            score_color = "#f43f5e"
            fill_class = "progress-fill-low"
            status_html = '<span class="status-pill status-pill-low">⚠️ Significant Skill Gap</span>'

        st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 36px 0;'>", unsafe_allow_html=True)

        # -----------------------------------------------------
        # 1. HERO RESULTS BANNER
        # -----------------------------------------------------
        st.markdown(f"""
        <div class="hero-banner">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 20px;">
                <div>
                    <div style="font-size: 0.9rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #a5b4fc; margin-bottom: 6px;">
                        Candidate Fit Assessment
                    </div>
                    <h2 style="font-size: 2.2rem; font-weight: 800; color: #ffffff; margin: 0 0 8px 0;">
                        {candidate_display_name}
                    </h2>
                    <div style="font-size: 1.1rem; color: #94a3b8;">
                        Applied for: <span style="color: #f1f5f9; font-weight: 600;">{role_display_name}</span>
                    </div>
                </div>
                <div style="text-align: right; min-width: 180px;">
                    {status_html}
                    <div style="font-size: 3.2rem; font-weight: 800; color: {score_color}; margin-top: 6px; line-height: 1;">
                        {score:.0f}%
                    </div>
                    <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 600;">OVERALL COMPATIBILITY</div>
                </div>
            </div>
            
            <div class="progress-container">
                <div class="{fill_class}" style="width: {min(max(score, 5), 100)}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # -----------------------------------------------------
        # 2. KEY METRICS TILES (4 KPI Cards)
        # -----------------------------------------------------
        k1, k2, k3, k4 = st.columns(4)

        # Metric 1: Match Score
        with k1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Match Score</div>
                <div class="kpi-val" style="color: {score_color};">{score:.0f}%</div>
                <div class="kpi-sub">Calculated via AI Matrix</div>
            </div>
            """, unsafe_allow_html=True)

        # Metric 2: Skills Coverage
        total_skills_count = len(match_result.matching_skills) + len(match_result.missing_skills)
        matched_count = len(match_result.matching_skills)
        skill_coverage_pct = (matched_count / total_skills_count * 100) if total_skills_count > 0 else 0

        with k2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Skills Coverage</div>
                <div class="kpi-val" style="color: #38bdf8;">{matched_count}/{total_skills_count}</div>
                <div class="kpi-sub">{skill_coverage_pct:.0f}% of required skills found</div>
            </div>
            """, unsafe_allow_html=True)

        # Metric 3: Experience Alignment
        cand_exp = f"{resume_parsed.total_experience_years:.1f} yrs" if resume_parsed.total_experience_years is not None else "Entry / N/A"
        req_exp = f"{job_parsed.minimum_experience:.1f} yrs" if job_parsed.minimum_experience is not None else "None specified"
        exp_badge = "✅ Met" if match_result.experience_met else "⚠️ Below Req"
        exp_color = "#34d399" if match_result.experience_met else "#fbbf24"

        with k3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Experience Check</div>
                <div class="kpi-val" style="color: {exp_color}; font-size: 1.6rem; margin: 8px 0;">{cand_exp}</div>
                <div class="kpi-sub">Req: {req_exp} ({exp_badge})</div>
            </div>
            """, unsafe_allow_html=True)

        # Metric 4: Education Qualification
        edu_status = "✅ Verified" if match_result.education_met else "⚠️ Review Needed"
        edu_color = "#34d399" if match_result.education_met else "#fbbf24"

        with k4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Education Fit</div>
                <div class="kpi-val" style="color: {edu_color}; font-size: 1.6rem; margin: 8px 0;">{edu_status}</div>
                <div class="kpi-sub">Qualifications evaluated</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

        # -----------------------------------------------------
        # 3. CORE HIGHLIGHT: SKILL GAP & FIT ANALYSIS
        # -----------------------------------------------------
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
            <span style="font-size: 1.5rem;">🧩</span>
            <span style="font-size: 1.4rem; font-weight: 700; color: #ffffff;">Deep Skill Gap & Compatibility Analysis</span>
        </div>
        """, unsafe_allow_html=True)

        gap_col1, gap_col2 = st.columns(2, gap="large")

        with gap_col1:
            st.markdown(f"""
            <div class="glass-panel" style="border-top: 4px solid #10b981; height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                    <div style="font-size: 1.15rem; font-weight: 700; color: #6ee7b7; display: flex; align-items: center; gap: 8px;">
                        <span>✓</span> Matched Skills
                    </div>
                    <span style="background: rgba(16, 185, 129, 0.2); color: #34d399; padding: 2px 10px; border-radius: 999px; font-size: 0.82rem; font-weight: 700;">
                        {len(match_result.matching_skills)} Skills
                    </span>
                </div>
                <p style="font-size: 0.88rem; color: #94a3b8; margin-bottom: 16px;">
                    Skills from the Job Description that the candidate actively demonstrates in their resume:
                </p>
            """, unsafe_allow_html=True)

            if match_result.matching_skills:
                matched_pills = "".join([f'<span class="skill-badge skill-matched">✓ {skill}</span>' for skill in match_result.matching_skills])
                st.markdown(f"<div style='line-height: 2.2;'>{matched_pills}</div></div>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: #94a3b8;'>No direct matching required skills found.</p></div>", unsafe_allow_html=True)

        with gap_col2:
            st.markdown(f"""
            <div class="glass-panel" style="border-top: 4px solid #f43f5e; height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                    <div style="font-size: 1.15rem; font-weight: 700; color: #fda4af; display: flex; align-items: center; gap: 8px;">
                        <span>✗</span> Missing Skills (Skill Gap)
                    </div>
                    <span style="background: rgba(244, 63, 94, 0.2); color: #fb7185; padding: 2px 10px; border-radius: 999px; font-size: 0.82rem; font-weight: 700;">
                        {len(match_result.missing_skills)} Gaps Detected
                    </span>
                </div>
                <p style="font-size: 0.88rem; color: #94a3b8; margin-bottom: 16px;">
                    Important competencies required by the job that are <b>missing</b> or not explicitly stated in the candidate's resume:
                </p>
            """, unsafe_allow_html=True)

            if match_result.missing_skills:
                missing_pills = "".join([f'<span class="skill-badge skill-gap">✗ {skill}</span>' for skill in match_result.missing_skills])
                st.markdown(f"<div style='line-height: 2.2;'>{missing_pills}</div></div>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: #34d399; font-weight: 600;'>🎉 Zero critical skill gaps! Exceptional skill coverage.</p></div>", unsafe_allow_html=True)

        # Additional candidate strengths (Bonus skills)
        if match_result.additional_skills:
            st.markdown("""
            <div class="glass-panel" style="margin-top: 20px; border-top: 4px solid #8b5cf6;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #c4b5fd; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
                    <span>💡</span> Candidate's Additional Skills (Bonus Qualifications)
                </div>
                <p style="font-size: 0.88rem; color: #94a3b8; margin-bottom: 14px;">
                    Valuable competencies the candidate possesses that were not explicitly demanded by the job description:
                </p>
            """, unsafe_allow_html=True)
            bonus_pills = "".join([f'<span class="skill-badge skill-bonus">+ {skill}</span>' for skill in match_result.additional_skills])
            st.markdown(f"<div style='line-height: 2.2;'>{bonus_pills}</div></div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # -----------------------------------------------------
        # 4. AI RECRUITER VERDICT & ACTIONABLE RECOMMENDATIONS
        # -----------------------------------------------------
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
            <span style="font-size: 1.5rem;">🧠</span>
            <span style="font-size: 1.4rem; font-weight: 700; color: #ffffff;">AI Recruiter Intelligence & Guidance</span>
        </div>
        """, unsafe_allow_html=True)

        # Executive Summary Callout
        st.markdown(f"""
        <div class="verdict-box">
            <div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: #a5b4fc; letter-spacing: 0.05em; margin-bottom: 6px;">
                Executive Hiring Verdict
            </div>
            {match_result.verdict}
        </div>
        """, unsafe_allow_html=True)

        # Strengths & Actionable Recommendations
        v_col1, v_col2 = st.columns(2, gap="large")

        with v_col1:
            st.markdown("""
            <div class="glass-panel" style="height: 100%;">
                <div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; margin-bottom: 14px; display: flex; align-items: center; gap: 8px;">
                    <span>🌟</span> Core Strengths Identified
                </div>
            """, unsafe_allow_html=True)
            if match_result.strengths:
                for s in match_result.strengths:
                    st.markdown(f"• **{s}**")
            else:
                st.write("No major standout strengths recorded.")
            st.markdown("</div>", unsafe_allow_html=True)

        with v_col2:
            st.markdown("""
            <div class="glass-panel" style="height: 100%;">
                <div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; margin-bottom: 14px; display: flex; align-items: center; gap: 8px;">
                    <span>🎯</span> Recommendations to Close the Gap
                </div>
            """, unsafe_allow_html=True)
            if match_result.recommendations:
                for r in match_result.recommendations:
                    st.markdown(f"📌 {r}")
            else:
                st.write("Candidate already fulfills all essential criteria.")
            
            if match_result.experience_notes:
                st.markdown(f"<div style='margin-top: 12px; font-size: 0.85rem; color: #94a3b8;'><b>Experience Notes:</b> {match_result.experience_notes}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # -----------------------------------------------------
        # 5. STRUCTURED CANDIDATE PROFILE & JOB SPECIFICATION
        # -----------------------------------------------------
        with st.expander("📂 View Extracted Candidate Profile & Job Specification Details"):
            prof1, prof2 = st.columns(2, gap="large")

            with prof1:
                st.markdown("### 👤 Candidate Profile")
                st.write(f"**Name:** {resume_parsed.name or 'N/A'}")
                st.write(f"**Email:** {resume_parsed.email or 'N/A'}")
                st.write(f"**Phone:** {resume_parsed.phone or 'N/A'}")
                st.write(f"**Total Experience:** {resume_parsed.total_experience_years or 'Not stated'} years")

                st.markdown("#### 🎓 Education:")
                if resume_parsed.education:
                    for edu in resume_parsed.education:
                        st.markdown(f"- {edu}")
                else:
                    st.caption("No explicit education records detected.")

                st.markdown("#### 💼 Experience & Internships:")
                if resume_parsed.experiences:
                    for exp in resume_parsed.experiences:
                        st.markdown(f"**{exp.role or 'Position'}** at *{exp.company or 'Organization'}* ({exp.duration or 'Dates'})")
                        if exp.description:
                            st.caption(exp.description)
                        if exp.skills_used:
                            st.caption("Tools/Skills: " + ", ".join(exp.skills_used))
                else:
                    st.caption("No structured work history records detected.")

                if resume_parsed.projects:
                    st.markdown("#### 🚀 Projects:")
                    for proj in resume_parsed.projects:
                        st.markdown(f"- {proj}")

            with prof2:
                st.markdown("### 📋 Job Requirements Extracted")
                st.write(f"**Role:** {job_parsed.role}")
                st.write(f"**Required Experience:** {job_parsed.minimum_experience or 'Not specified'} years")

                st.markdown("#### Required Skills:")
                if job_parsed.required_skills:
                    st.write(", ".join(job_parsed.required_skills))
                else:
                    st.caption("None specified explicitly.")

                st.markdown("#### Preferred Skills:")
                if job_parsed.preferred_skills:
                    st.write(", ".join(job_parsed.preferred_skills))
                else:
                    st.caption("None specified.")

                st.markdown("#### Key Responsibilities:")
                if job_parsed.responsibilities:
                    for resp in job_parsed.responsibilities:
                        st.markdown(f"- {resp}")
                else:
                    st.caption("None specified.")

        # -----------------------------------------------------
        # 6. ACTION & EXPORT BAR
        # -----------------------------------------------------
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        act1, act2 = st.columns([1, 1])

        report_payload = {
            "evaluation": match_result.model_dump(),
            "candidate": resume_parsed.model_dump(),
            "job_specification": job_parsed.model_dump()
        }

        with act1:
            st.download_button(
                label="📥 Download Full Evaluation Report (JSON)",
                data=json.dumps(report_payload, indent=2),
                file_name=f"match_report_{candidate_display_name.replace(' ', '_')}.json",
                mime="application/json",
                use_container_width=True
            )

        with act2:
            if st.button("🔄 Analyze Another Candidate", use_container_width=True):
                st.session_state["analysis_done"] = False
                st.rerun()


if __name__ == "__main__":
    main()
