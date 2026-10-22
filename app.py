"""
Streamlit Web Application for Learning Agent System

A professional, modern web interface for the autonomous learning agent.
Provides interactive learning experiences with checkpoint progression,
dynamic materials, and adaptive teaching.
"""

import streamlit as st
import asyncio
import sys
import time
import re
import os
import random
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Bootstrap environment from Streamlit secrets before importing app modules
def _load_env_from_streamlit_secrets():
    try:
        secrets = st.secrets
    except Exception:
        return

    keys = [
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "EMBEDDING_MODEL",
        "THRESHOLD",
        "HF_TOKEN",
        "LANGSMITH_TRACING_ENABLED",
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_PROJECT",
        "LANGCHAIN_ENDPOINT",
    ]

    # Support both top-level keys and [env] table in secrets.toml
    env_table = {}
    try:
        env_table = secrets.get("env", {})
    except Exception:
        env_table = {}

    for key in keys:
        if key in os.environ and os.environ.get(key):
            continue
        if key in secrets:
            os.environ[key] = str(secrets[key])
        elif isinstance(env_table, dict) and key in env_table:
            os.environ[key] = str(env_table[key])

_load_env_from_streamlit_secrets()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.sample_data import create_learning_paths
from src.custom_topics import load_custom_topics, create_topic_wizard, add_custom_topic
from src.multi_checkpoint import run_multi_checkpoint_session
from src.models import LearningPath, Checkpoint
from src.workflow import create_unified_workflow, create_question_generation_workflow
from src.models import LearningAgentState
from src.user_interaction import collect_user_answers, display_score_feedback
from src.file_upload import get_upload_handler
from src.langsmith_config import langsmith_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Learning Agent System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS theme: modern "Command Center" look and feel
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg: #f6f4ee;
        --surface: #ffffff;
        --surface-soft: #fbfaf6;
        --ink: #191817;
        --muted: #6f6a61;
        --line: #e6e1d6;
        --line-strong: #d8d2c4;
        --accent: #1e5c47;
        --accent-soft: #edf5f1;
        --success: #1b7a4b;
        --warn: #a3620b;
        --danger: #b04238;
        --amber: #b7791f;
        --shadow-xs: 0 1px 2px rgba(25, 24, 23, 0.05);
        --shadow-sm: 0 1px 2px rgba(25, 24, 23, 0.04), 0 10px 30px -12px rgba(25, 24, 23, 0.12);
        --radius: 14px;
    }

    .stApp {
        background: linear-gradient(180deg, #faf8f2 0%, var(--bg) 140px);
        color: var(--ink);
        font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
    }

    .main {
        padding-top: 0.5rem;
    }

    h1, h2, h3, h4 {
        font-family: 'Fraunces', Georgia, serif;
        font-weight: 600;
        letter-spacing: -0.01em;
        color: var(--ink);
    }

    @keyframes popIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: none; }
    }

    @keyframes growBar {
        from { width: 0; }
    }

    .shell {
        background: transparent;
        border: none;
        border-bottom: 1px solid var(--line-strong);
        box-shadow: none;
        padding: 0.35rem 0.2rem 1.05rem 0.2rem;
        margin-bottom: 1.15rem;
        animation: popIn 0.3s ease both;
    }

    .hero-row {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
    }

    .hero-title {
        font-family: 'Fraunces', Georgia, serif;
        font-size: 2.05rem;
        font-weight: 600;
        line-height: 1.1;
        margin: 0;
        color: var(--ink);
    }

    .hero-title em {
        font-style: italic;
        font-weight: 500;
        color: var(--accent);
    }

    .hero-subtitle {
        font-size: 0.92rem;
        color: var(--muted);
        margin: 0.3rem 0 0 0;
    }

    .stage-chip {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        padding: 0.3rem 0.75rem;
        border: 1px solid var(--line-strong);
        border-radius: 999px;
        background: var(--surface);
        white-space: nowrap;
    }

    .pills-row {
        display: flex;
        gap: 0.55rem;
        flex-wrap: wrap;
        margin: 0.7rem 0 0.1rem 0;
    }

    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--muted);
    }

    .pill .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--line-strong);
        flex-shrink: 0;
    }

    .pill.ok .dot   { background: var(--success); box-shadow: 0 0 0 3px rgba(27, 122, 75, 0.12); }
    .pill.warn .dot { background: var(--amber);   box-shadow: 0 0 0 3px rgba(183, 121, 31, 0.12); }
    .pill.err .dot  { background: var(--danger);  box-shadow: 0 0 0 3px rgba(176, 66, 56, 0.12); }
    .pill.ok   { color: var(--ink); }

    .plate-pill {
        padding: 0.32rem 0.75rem;
        border: 1px solid var(--line-strong);
        border-radius: 999px;
        background: var(--surface);
        box-shadow: var(--shadow-xs);
    }

    .stepper {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        align-items: center;
        padding: 0 0.2rem 0.9rem 0.2rem;
        margin-bottom: 0.9rem;
        border-bottom: 1px solid var(--line);
    }

    .step {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--muted);
    }

    .step .sd {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        border: 1.5px solid var(--line-strong);
        background: var(--surface);
    }

    .step.done .sd { background: var(--accent); border-color: var(--accent); }

    .step.active { color: var(--ink); }

    .step.active .sd {
        background: var(--ink);
        border-color: var(--ink);
        box-shadow: 0 0 0 3px rgba(25, 24, 23, 0.08);
    }

    .step + .step::before {
        content: "";
        width: 16px;
        height: 1px;
        background: var(--line-strong);
        margin-right: 0.55rem;
    }

    .card {
        padding: 1rem 1.05rem;
        border-radius: var(--radius);
        background: var(--surface);
        border: 1px solid var(--line);
        box-shadow: var(--shadow-sm);
        margin-bottom: 0.85rem;
        animation: popIn 0.3s ease both;
    }

    .kicker {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: var(--muted);
        margin-bottom: 0.3rem;
    }

    .mono {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 2.5rem;
        height: 2.5rem;
        font-family: 'Fraunces', Georgia, serif;
        font-size: 1.15rem;
        font-weight: 600;
        color: var(--accent);
        background: var(--accent-soft);
        border: 1px solid rgba(30, 92, 71, 0.18);
        border-radius: 10px;
        flex-shrink: 0;
    }

    .path-badge {
        display: inline-block;
        margin-top: 0.55rem;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
    }

    .mix-note {
        display: flex;
        justify-content: space-between;
        gap: 0.8rem;
        flex-wrap: wrap;
        align-items: baseline;
        padding: 0.55rem 0.8rem;
        border: 1px solid var(--line);
        border-radius: 10px;
        background: var(--surface-soft);
        font-size: 0.8rem;
        color: var(--muted);
        margin-bottom: 0.8rem;
    }

    .mix-note strong { color: var(--ink); font-weight: 600; }

    .chip-row {
        display: flex;
        gap: 0.45rem;
        flex-wrap: wrap;
        margin: 0.45rem 0 0.8rem 0;
    }

    .chip {
        padding: 0.28rem 0.66rem;
        border: 1px solid var(--line-strong);
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        background: var(--surface);
        color: var(--ink);
        white-space: nowrap;
    }

    .q-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.8rem;
        margin: 0 0 0.4rem 0;
    }

    .q-index {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
    }

    .question-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        padding: 0.95rem 1.05rem;
        margin-bottom: 0.75rem;
        line-height: 1.55;
        box-shadow: var(--shadow-xs);
        font-size: 0.98rem;
    }

    .q-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: var(--muted);
        border: 1px solid var(--line-strong);
        border-radius: 999px;
        padding: 0.24rem 0.6rem;
        background: var(--surface);
        margin: 0 0 0.45rem 0;
        margin-right: 0.3rem;
    }

    .q-badge::before {
        content: "";
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--accent);
    }

    .q-badge.short::before { background: var(--amber); }
    .q-badge.long::before  { background: var(--danger); }

    .tile {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        box-shadow: var(--shadow-xs);
        padding: 0.85rem 0.9rem;
        text-align: center;
    }

    .tile .v {
        font-family: 'Fraunces', Georgia, serif;
        font-size: 1.5rem;
        font-weight: 600;
        line-height: 1.15;
        color: var(--ink);
    }

    .tile .v.ok  { color: var(--success); }
    .tile .v.bad { color: var(--danger); }

    .tile .k {
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--muted);
        margin-top: 0.15rem;
    }

    .score-ring {
        width: 150px;
        height: 150px;
        border-radius: 50%;
        background: conic-gradient(var(--ring-c, var(--success)) calc(var(--p, 0) * 1%), #e9e4d9 0);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: var(--shadow-sm);
    }

    .score-ring .inner {
        width: 114px;
        height: 114px;
        background: var(--surface);
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    .score-ring .val {
        font-family: 'Fraunces', Georgia, serif;
        font-size: 1.8rem;
        font-weight: 600;
        line-height: 1.05;
        color: var(--ink);
    }

    .score-ring .lbl {
        font-size: 0.62rem;
        font-weight: 600;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--muted);
    }

    .qbar-wrap {
        margin: 0.5rem 0 0.25rem 0;
        padding: 0.7rem 0.85rem;
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        box-shadow: var(--shadow-xs);
    }

    .qbar-label {
        display: flex;
        justify-content: space-between;
        gap: 0.8rem;
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--ink);
        margin-bottom: 0.35rem;
    }

    .qbar-label .score { color: var(--muted); font-weight: 600; }

    .qbar {
        height: 8px;
        border-radius: 999px;
        background: #ece7dc;
        overflow: hidden;
    }

    .qbar > i {
        display: block;
        height: 100%;
        background: var(--bar-c, var(--success));
        border-radius: 999px;
        animation: growBar 0.7s ease both;
    }

    .resource-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        box-shadow: var(--shadow-xs);
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.55rem;
    }

    .resource-card .rt {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--ink);
        margin-bottom: 0.15rem;
    }

    .resource-card a {
        color: var(--accent);
        font-weight: 500;
        text-decoration: none;
        font-size: 0.82rem;
        word-break: break-all;
    }

    .resource-card a:hover { text-decoration: underline; }

    .resource-card .src {
        font-size: 0.74rem;
        color: var(--muted);
        margin-top: 0.25rem;
    }

    .success-box, .warning-box, .error-box {
        padding: 0.95rem 1.1rem;
        border-radius: var(--radius);
        border: 1px solid var(--line);
        border-left: 3px solid var(--success);
        background: var(--surface);
        box-shadow: var(--shadow-xs);
        margin-bottom: 0.7rem;
        font-size: 0.93rem;
        color: var(--ink);
        animation: popIn 0.3s ease both;
    }

    .success-box { border-left-color: var(--success); background: #f6faf7; }
    .warning-box { border-left-color: var(--amber);  background: #fbf8f2; }
    .error-box   { border-left-color: var(--danger); background: #fbf5f4; }

    .stButton > button,
    [data-testid="stFormSubmitButton"] button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid var(--ink);
        background: var(--surface);
        color: var(--ink);
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0.42rem 0.9rem;
        box-shadow: var(--shadow-xs);
        transition: background 0.15s ease, color 0.15s ease, transform 0.1s ease, box-shadow 0.15s ease;
    }

    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        background: var(--ink);
        color: #fbfaf6;
        box-shadow: var(--shadow-sm);
        transform: translateY(-1px);
    }

    .stButton > button:active,
    [data-testid="stFormSubmitButton"] button:active {
        transform: translateY(0);
        box-shadow: var(--shadow-xs);
    }

    button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {
        background: var(--ink) !important;
        color: #fbfaf6 !important;
        border-color: var(--ink) !important;
    }

    button[kind="primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        background: #2e2c29 !important;
        color: #fbfaf6 !important;
    }

    .stProgress > div > div > div > div {
        background: var(--ink);
    }

    .stTextArea textarea, .stTextInput input {
        background: var(--surface);
        color: var(--ink);
        border: 1px solid var(--line-strong);
        border-radius: 10px;
        box-shadow: var(--shadow-xs);
    }

    .stTextArea textarea:focus, .stTextInput input:focus {
        outline: none;
        border-color: var(--ink);
        box-shadow: 0 0 0 3px rgba(25, 24, 23, 0.06);
    }

    .stRadio > div { gap: 0.15rem; }

    .stRadio label {
        border: 1px solid transparent;
        border-radius: 9px;
        padding: 0.32rem 0.55rem;
        margin-bottom: 0.15rem;
        background: transparent;
        transition: background 0.12s ease;
    }

    .stRadio label:hover {
        background: var(--surface-soft);
    }

    [data-testid="stFileUploaderDropzone"] {
        border: 1px solid var(--line-strong) !important;
        border-radius: var(--radius) !important;
        background: var(--surface) !important;
        box-shadow: var(--shadow-xs);
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--ink) !important;
    }

    .stAlert {
        border-radius: 10px;
        border: 1px solid var(--line-strong);
    }

    details[data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: var(--radius);
        background: var(--surface);
        box-shadow: var(--shadow-xs);
        margin-bottom: 0.5rem;
    }

    details[data-testid="stExpander"] summary {
        font-weight: 600;
    }

    [data-testid="stCaptionContainer"] {
        color: var(--muted);
        font-size: 0.8rem;
        letter-spacing: 0.02em;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if 'stage' not in st.session_state:
        st.session_state.stage = 'select_path'  # select_path, upload_files, learning, results
    
    if 'selected_path' not in st.session_state:
        st.session_state.selected_path = None
    
    if 'current_checkpoint_index' not in st.session_state:
        st.session_state.current_checkpoint_index = 0
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    
    if 'learning_state' not in st.session_state:
        st.session_state.learning_state = None
    
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    
    if 'answers' not in st.session_state:
        st.session_state.answers = {}
    
    if 'show_create_topic' not in st.session_state:
        st.session_state.show_create_topic = False
    
    if 'completed_checkpoints' not in st.session_state:
        st.session_state.completed_checkpoints = []

def render_header():
    """Render application header."""
    stage = st.session_state.get("stage", "select_path")
    stage_labels = {
        "select_path": "Select Path",
        "upload_files": "Upload Materials",
        "study": "Study",
        "learning": "Assessment",
        "results": "Results"
    }

    st.markdown('<div class="shell">', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-row">'
        '<p class="hero-title">Learning <em>Agent</em> Studio</p>'
        f'<span class="stage-chip">{stage_labels.get(stage, "Unknown")}</span>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown('<p class="hero-subtitle">Adaptive checkpoint learning — clear goals, strong feedback, Feynman-style re-teaching.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Check LLM provider status
    llm_provider_pref = os.getenv("LLM_PROVIDER", "auto").lower().strip()
    hf_token_set = bool(os.getenv("HF_TOKEN", ""))
    hf_model = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")
    ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_up = False

    def _pill(kind: str, text: str) -> str:
        """Build a quiet status pill with a colored signal dot."""
        return f'<span class="pill plate-pill {kind}"><i class="dot"></i>{text}</span>'

    try:
        import httpx
        response = httpx.get(f"{ollama_base}/api/tags", timeout=2.0)
        if response.status_code == 200:
            ollama_up = True
    except Exception:
        ollama_up = False

    if llm_provider_pref == "huggingface":
        provider_pill = _pill("warn", f"Provider · Hugging Face ({hf_model})")
    elif llm_provider_pref == "ollama":
        provider_pill = _pill("ok", "Provider · Ollama")
    else:
        if ollama_up:
            provider_pill = _pill("ok", "Provider · Ollama (auto)")
        elif hf_token_set:
            provider_pill = _pill("warn", "Provider · Hugging Face (fallback)")
        else:
            provider_pill = _pill("err", "Provider · none available")

    ollama_pill = _pill("ok", "Ollama · Online") if ollama_up else _pill("err", "Ollama · Offline")

    st.markdown(f'<div class="pills-row">{provider_pill}{ollama_pill}</div>', unsafe_allow_html=True)

    if not ollama_up:
        with st.expander("How to start Ollama"):
            st.code("ollama serve", language="bash")
            st.markdown("Or ensure Ollama is running in the background.")

    if not hf_token_set and not ollama_up:
        st.warning("No usable LLM backend detected — start Ollama or set `HF_TOKEN` in `.env`.")

    with st.expander("LangSmith Status", expanded=False):
        tracing_env = os.getenv("LANGSMITH_TRACING_ENABLED", "false").lower() == "true"
        api_key_set = bool(os.getenv("LANGCHAIN_API_KEY"))
        project = os.getenv("LANGCHAIN_PROJECT", "Learning-Agent-System")
        endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        runtime_enabled = bool(getattr(langsmith_config, "enabled", False))

        st.write(f"- Env tracing toggle: {'ON' if tracing_env else 'OFF'}")
        st.write(f"- API key present: {'YES' if api_key_set else 'NO'}")
        st.write(f"- Project: `{project}`")
        st.write(f"- Endpoint: `{endpoint}`")
        st.write(f"- Runtime tracing active: {'YES' if runtime_enabled else 'NO'}")

        if not tracing_env:
            st.warning("Set `LANGSMITH_TRACING_ENABLED=true` in `.env`.")
        elif not api_key_set:
            st.warning("Set `LANGCHAIN_API_KEY` in `.env`.")
        elif not runtime_enabled:
            st.warning("Tracing is configured but inactive at runtime; restart app and recheck credentials/project.")
        else:
            st.success("LangSmith tracing is active for this app session.")

def _question_type_label(question: Dict) -> str:
    q_type = str(question.get('type', '')).lower()
    if q_type in ('mcq', 'multiple_choice'):
        return "Multiple Choice"
    if q_type in ('short_answer', 'short'):
        return "Short Answer"
    if q_type in ('long_answer', 'long'):
        return "Long Answer"
    return "Open Answer"

def ensure_question_distribution(questions: List[Dict], checkpoint_requirements: Optional[List[str]] = None) -> List[Dict]:
    """UI-level hard guard: always return 5 MCQ, 3 short, 2 long questions."""
    checkpoint_requirements = checkpoint_requirements or []
    by_type = {"multiple_choice": [], "short_answer": [], "long_answer": []}

    for q in questions or []:
        q_type = str(q.get("type", "")).lower()
        if q_type in ("mcq", "multiple_choice"):
            q["type"] = "multiple_choice"
            by_type["multiple_choice"].append(q)
        elif q_type in ("long", "long_answer"):
            q["type"] = "long_answer"
            by_type["long_answer"].append(q)
        else:
            q["type"] = "short_answer"
            by_type["short_answer"].append(q)

    def mcq_stub(i: int) -> Dict:
        req = checkpoint_requirements[min(i, len(checkpoint_requirements)-1)] if checkpoint_requirements else "the checkpoint topic"
        return {
            "question": f"Which choice best aligns with the objective: '{req}'?",
            "type": "multiple_choice",
            "options": [
                "A) It captures the core concept with practical relevance.",
                "B) It avoids key concepts and remains vague.",
                "C) It contradicts the objective directly.",
                "D) It is unrelated to the checkpoint topic."
            ],
            "correct_answer": "A",
            "expected_concepts": checkpoint_requirements[:2]
        }

    def short_stub(i: int) -> Dict:
        req = checkpoint_requirements[min(i, len(checkpoint_requirements)-1)] if checkpoint_requirements else "the checkpoint topic"
        return {
            "question": f"In 2-4 sentences, explain the key idea behind '{req}'.",
            "type": "short_answer",
            "expected_concepts": checkpoint_requirements[:3]
        }

    def long_stub(i: int) -> Dict:
        req = checkpoint_requirements[min(i, len(checkpoint_requirements)-1)] if checkpoint_requirements else "the checkpoint topic"
        return {
            "question": f"Provide a detailed analysis of '{req}', including reasoning, examples, and limitations.",
            "type": "long_answer",
            "expected_concepts": checkpoint_requirements
        }

    while len(by_type["multiple_choice"]) < 5:
        by_type["multiple_choice"].append(mcq_stub(len(by_type["multiple_choice"])))
    while len(by_type["short_answer"]) < 3:
        by_type["short_answer"].append(short_stub(len(by_type["short_answer"])))
    while len(by_type["long_answer"]) < 2:
        by_type["long_answer"].append(long_stub(len(by_type["long_answer"])))

    mcq_selected = random.sample(by_type["multiple_choice"], 5) if len(by_type["multiple_choice"]) > 5 else by_type["multiple_choice"][:5]
    short_selected = random.sample(by_type["short_answer"], 3) if len(by_type["short_answer"]) > 3 else by_type["short_answer"][:3]
    long_selected = random.sample(by_type["long_answer"], 2) if len(by_type["long_answer"]) > 2 else by_type["long_answer"][:2]

    selected = mcq_selected + short_selected + long_selected
    random.shuffle(selected)
    for idx, q in enumerate(selected, 1):
        q["question_id"] = f"q_{idx}"
    return selected

def render_stage_rail():
    """Render the minimal workflow stepper."""
    stage = st.session_state.get("stage", "select_path")
    order = ["select_path", "upload_files", "study", "learning", "results"]
    labels = ["Path", "Materials", "Study", "Assessment", "Results"]
    current_idx = order.index(stage) if stage in order else 0

    steps = []
    for idx, label in enumerate(labels):
        cls = "step"
        if idx < current_idx:
            cls += " done"
        elif idx == current_idx:
            cls += " active"
        steps.append(f'<span class="{cls}"><i class="sd"></i>{label}</span>')

    st.markdown('<div class="stepper">' + "".join(steps) + '</div>', unsafe_allow_html=True)

def show_loading_sequence(status, messages: List[str], delay: float = 0.18):
    """Show rotating status messages to keep users engaged during long tasks."""
    for msg in messages:
        status.write(msg)
        time.sleep(delay)

def format_study_markdown(text: str) -> str:
    """Normalize generated study text into readable Markdown blocks."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    cleaned = cleaned.replace("\r\n", "\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"(?m)^(\d+)\.\s*([A-Z][^\n:]{2,40})$", r"## \1. \2", cleaned)
    cleaned = re.sub(r"(?m)^([A-Z][A-Z\s]{4,40}):\s*$", lambda m: f"## {m.group(1).title()}", cleaned)
    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    return "\n\n".join(paragraphs)

def is_study_content_sufficient(summary: str, materials: List[Dict]) -> bool:
    """Check whether we have enough high-quality content for pre-assessment study."""
    summary_words = len((summary or "").split())
    material_chars = sum(len((m.get("content") or "")) for m in materials)
    return summary_words >= 220 or material_chars >= 1800

def build_study_pack_markdown(checkpoint: Dict, summary: str, materials: List[Dict]) -> str:
    """Build robust study material markdown from summary + collected sources."""
    title = checkpoint.get("title", "This checkpoint")
    reqs = checkpoint.get("requirements", [])
    req_section = "\n".join([f"- {r}" for r in reqs]) if reqs else "- Understand the main concepts\n- Apply them correctly"

    curated_points = []
    for material in materials[:6]:
        content = (material.get("content") or "").strip()
        if not content:
            continue
        snippet = re.sub(r"\s+", " ", content)[:320].strip()
        if len(snippet) > 60:
            curated_points.append(f"- **{material.get('title','Source')}**: {snippet}...")

    if not curated_points:
        curated_points.append("- Review the generated explanation and checkpoint objectives carefully before proceeding.")

    summary_md = format_study_markdown(summary) if summary else ""
    if not summary_md or len(summary_md.split()) < 120:
        summary_md = (
            f"## Core Concepts for {title}\n\n"
            "Study this section before attempting the assessment:\n\n"
            f"{req_section}\n\n"
            "### Practical Understanding Notes\n\n"
            + "\n".join(curated_points)
        )

    return (
        f"## Checkpoint Focus: {title}\n\n"
        "### Learning Objectives\n"
        f"{req_section}\n\n"
        "### Comprehensive Explanation\n\n"
        f"{summary_md}\n\n"
        "### Source-Based Key Notes\n\n"
        + "\n".join(curated_points)
    )

def render_path_selection():
    """Render learning path selection interface."""
    st.markdown('<div class="card"><div class="kicker">Setup</div><h3 style="margin:0;">Choose your learning path</h3></div>', unsafe_allow_html=True)

    default_paths = create_learning_paths()
    custom_paths = load_custom_topics()
    all_paths = default_paths + custom_paths

    if st.button("Create custom topic", use_container_width=True):
        st.session_state.show_create_topic = True
        st.rerun()

    cols = st.columns(2, gap="medium")
    for i, path in enumerate(all_paths):
        with cols[i % 2]:
            monogram = (path["title"] or "?").strip()[:1].upper()
            n_checkpoints = len(path["checkpoints"])
            st.markdown(f"""
            <div class="card" style="margin-bottom:0.9rem;">
                <div style="display:flex;align-items:center;gap:0.75rem;">
                    <span class="mono">{monogram}</span>
                    <h4 style="margin:0;">{path['title']}</h4>
                </div>
                <p style="margin:0.55rem 0 0 0;color:var(--muted);">{path['description']}</p>
                <span class="path-badge">{n_checkpoints} checkpoint{'s' if n_checkpoints != 1 else ''}</span>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Select this path", key=f"select_{i}"):
                st.session_state.selected_path = path
                st.session_state.stage = 'upload_files'
                st.rerun()

    # Custom topic creation form
    if st.session_state.show_create_topic:
        render_custom_topic_form()

def render_custom_topic_form():
    """Render custom topic creation form."""
    st.markdown("---")
    st.subheader("Create Custom Learning Topic")
    
    st.info("Enter a topic name (e.g., 'Python Basics', 'Machine Learning') and the system will handle everything else.")
    
    with st.form("custom_topic_form"):
        topic_name = st.text_input(
            "Topic Name",
            placeholder="e.g., Python Basics, Machine Learning, Web Development",
            help="Enter any topic you want to learn about"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("Start Learning", use_container_width=True)
        with col2:
            cancelled = st.form_submit_button("Cancel", use_container_width=True)
        
        if submitted and topic_name and topic_name.strip():
            # Auto-generate topic ID from name
            topic_id = topic_name.lower().replace(" ", "_").replace("-", "_")
            # Remove special characters
            topic_id = ''.join(c for c in topic_id if c.isalnum() or c == '_')
            
            # Auto-generate description
            description = f"Learn the fundamentals and key concepts of {topic_name}"
            
            # Create a single checkpoint for the custom topic
            # The system will dynamically generate materials
            new_topic = {
                "id": topic_id,
                "title": topic_name.strip(),
                "description": description,
                "checkpoints": [
                    {
                        "id": f"{topic_id}_main",
                        "title": topic_name.strip(),
                        "description": description,
                        "requirements": [
                            f"Understand the fundamentals of {topic_name}",
                            f"Apply key concepts from {topic_name}",
                            f"Demonstrate knowledge of {topic_name}"
                        ]
                    }
                ]
            }
            
            if add_custom_topic(new_topic):
                st.success(f"Topic '{topic_name}' created! Starting learning session...")
                st.session_state.show_create_topic = False
                st.session_state.selected_path = new_topic
                st.session_state.stage = 'upload_files'
                st.rerun()
            else:
                st.error("This topic already exists. Please choose a different name.")
        
        if cancelled:
            st.session_state.show_create_topic = False
            st.rerun()

def render_file_upload():
    """Render file upload interface."""
    st.markdown('<div class="card"><div class="kicker">Setup</div><h3 style="margin:0;">Upload learning materials <span style="font-weight:500;color:var(--muted);">(optional)</span></h3></div>', unsafe_allow_html=True)

    path = st.session_state.selected_path
    st.info(f"Selected path — **{path['title']}**")

    st.markdown(
        "Add your own notes to anchor the session — or skip ahead and let the AI gather "
        "materials from the web for you.",
        unsafe_allow_html=False
    )

    st.markdown(
        '<div class="chip-row">'
        '<span class="chip">PDF</span>'
        '<span class="chip">DOCX</span>'
        '<span class="chip">Markdown</span>'
        '<span class="chip">TXT</span>'
        '<span class="chip">Max 10 MB per file</span>'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        "Upload files",
        type=['pdf', 'docx', 'md', 'txt'],
        accept_multiple_files=True,
        help="Maximum 10MB per file"
    )

    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.success(f"Uploaded {len(uploaded_files)} file(s)")
        chips = "".join(
            f'<span class="chip">{f.name} · {f.size / 1024:.1f} KB</span>'
            for f in uploaded_files
        )
        st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start with my files", use_container_width=True, disabled=not uploaded_files):
            st.session_state.stage = 'learning'
            st.rerun()

    with col2:
        if st.button("Generate materials instead", use_container_width=True):
            st.session_state.uploaded_files = []
            st.session_state.stage = 'learning'
            st.rerun()

def render_study_materials():
    """Display learning materials for study before assessment."""
    st.markdown('<div class="card"><div class="kicker">Study Zone</div><h3 style="margin:0;">Study Materials</h3></div>', unsafe_allow_html=True)
    st.info("Review the learning materials below before starting the assessment.")
    
    state = st.session_state.learning_state
    
    if not state:
        st.error("No learning materials available.")
        return
    
    # Display learning content as one continuous explanation
    summary = state.get('summary', '')
    materials = state.get('collected_materials', [])
    checkpoint = state.get('current_checkpoint', {})
    has_enough_content = is_study_content_sufficient(summary, materials)
    study_pack = build_study_pack_markdown(checkpoint, summary, materials)
    
    if study_pack:
        word_count = len(study_pack.split())
        read_mins = max(1, round(word_count / 200))
        st.markdown("### Study material")
        st.caption(f"About {read_mins} min read · {word_count} words")

        formatted_md = format_study_markdown(study_pack)
        st.markdown(formatted_md)

        if not has_enough_content:
            st.warning("The generated summary was brief, so this section was expanded using collected source materials.")
    else:
        st.warning("No learning content available.")
    
    # Extract and display resource links
    resource_links = []
    for material in materials:
        url = material.get('url') or material.get('link') or material.get('source_url')
        title = material.get('title', 'Resource')
        source = material.get('source', 'Unknown')
        
        if url and url.startswith('http'):
            resource_links.append({
                'title': title,
                'url': url,
                'source': source
            })
    
    # Display resource links section
    if resource_links:
        st.markdown("### Additional Resources")
        st.caption(f"Explore {len(resource_links)} external resources for deeper learning")

        cards = "".join(
            '<div class="resource-card">'
            f'<div class="rt">{i}. {resource["title"]}</div>'
            f'<a href="{resource["url"]}" target="_blank">{resource["url"]}</a>'
            f'<div class="src">Source · {resource["source"]}</div>'
            '</div>'
            for i, resource in enumerate(resource_links, 1)
        )
        st.markdown(cards, unsafe_allow_html=True)
    
    # Button to proceed to assessment
    st.markdown("")
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start Assessment", use_container_width=True, type="primary"):
            st.session_state.stage = 'learning'
            st.rerun()

def render_learning_session():
    """Render the learning session with materials, questions, and assessment."""
    path = st.session_state.selected_path
    checkpoint_idx = st.session_state.current_checkpoint_index
    
    if checkpoint_idx >= len(path['checkpoints']):
        render_completion()
        return
    
    checkpoint = path['checkpoints'][checkpoint_idx]
    
    # Progress bar
    
    if checkpoint_idx >= len(path['checkpoints']):
        render_completion()
        return
    
    checkpoint = path['checkpoints'][checkpoint_idx]
    
    # Progress bar
    progress = checkpoint_idx / len(path['checkpoints'])
    st.progress(progress, text=f"Progress: {checkpoint_idx}/{len(path['checkpoints'])} checkpoints completed")
    
    st.markdown(
        f'<div class="card"><div class="kicker">Checkpoint</div><h3 style="margin:0;">{checkpoint["title"]}</h3><p style="margin:8px 0 0 0;color:#3f3f46;">{checkpoint["description"]}</p></div>',
        unsafe_allow_html=True
    )
    
    # Learning objectives
    with st.expander("Learning Objectives", expanded=False):
        for req in checkpoint['requirements']:
            st.markdown(f"- {req}")
    
    # Run workflow in background
    if st.session_state.learning_state is None:
        with st.status("Preparing learning materials...", expanded=True) as prep_status:
            show_loading_sequence(prep_status, [
                "Warming up retrieval + reasoning pipelines...",
                "Aligning checkpoint objectives with source context...",
                "Drafting an assessment-focused study brief..."
            ])
            # Process uploaded files
            uploaded_paths = None
            if st.session_state.uploaded_files:
                upload_handler = get_upload_handler()
                temp_paths = []
                prep_status.write("Processing your uploaded files...")
                
                for uploaded_file in st.session_state.uploaded_files:
                    file_content = uploaded_file.read()
                    result = upload_handler.handle_upload(
                        file_content=file_content,
                        file_name=uploaded_file.name
                    )
                    if result['success']:
                        temp_paths.append(result['file_path'])
                
                uploaded_paths = temp_paths if temp_paths else None
            
            # Initialize state
            initial_state: LearningAgentState = {
                "learning_path": path,
                "current_checkpoint_index": checkpoint_idx,
                "completed_checkpoints": st.session_state.completed_checkpoints,
                "total_checkpoints": len(path['checkpoints']),
                "has_next_checkpoint": checkpoint_idx < len(path['checkpoints']) - 1,
                "learning_path_completed": False,
                "current_checkpoint": checkpoint,
                "collected_materials": [],
                "summary": "",
                "milestone1_score": 0.0,
                "processed_context": [],
                "generated_questions": [],
                "verification_results": [],
                "score_percentage": 0.0,
                "meets_threshold": False,
                "feynman_retry_count": 0,
                "feynman_retry_requested": False,
                "feynman_explanations": [],
                "user_uploaded_notes_path": uploaded_paths,
                "materials_validation": None,
                "workflow_step": "initialized",
                "workflow_history": [],
                "errors": [],
                "user_mode": "streamlit"  # Flag to skip CLI input collection
            }
            
            # Run workflow up to question generation only (Streamlit mode)
            # This stops before verify_understanding so UI can collect answers
            workflow = create_question_generation_workflow()
            
            try:
                show_loading_sequence(prep_status, [
                    "Collecting materials from trusted sources...",
                    "Distilling context into concept chunks...",
                    "Generating checkpoint question set..."
                ])
                result = asyncio.run(workflow.ainvoke(initial_state))
                
                st.session_state.learning_state = result
                st.session_state.questions = ensure_question_distribution(
                    result.get('generated_questions', []),
                    checkpoint.get("requirements", [])
                )
                
                if st.session_state.questions:
                    prep_status.write(f"Generated {len(st.session_state.questions)} questions.")
                    st.session_state.study_materials_ready = True
                    prep_status.update(label="Study materials ready!", state="complete")
                    # Redirect to study stage to show materials before assessment
                    st.session_state.stage = 'study'
                else:
                    prep_status.write("No questions generated.")
                    prep_status.update(label="Error", state="error")
                        
            except Exception as e:
                st.error(f"Error running workflow: {e}")
                logger.exception("Workflow error")
                return
            
            # After processing, redirect to study stage
            if st.session_state.study_materials_ready:
                st.rerun()
    
    # Display questions only if user has reviewed materials (coming from study stage)
    if st.session_state.questions and not st.session_state.answers:
        st.markdown("---")
        render_questions()
    elif st.session_state.answers:
        st.info("Answers submitted. Evaluating...")
    else:
        st.warning("No questions generated. Please try again or contact support.")

def render_questions():
    """Render interactive questions."""
    st.markdown('<div class="card"><div class="kicker">Assessment</div><h3 style="margin:0;">Checkpoint Questions</h3></div>', unsafe_allow_html=True)
    question_count = len(st.session_state.questions)
    st.markdown(
        '<div class="mix-note"><strong>Question mix</strong><span>5 multiple choice · 3 short answer · 2 long answer</span></div>',
        unsafe_allow_html=True
    )
    st.info(f"Answer all {question_count} questions. Passing threshold remains 70%.")
    
    questions = st.session_state.questions
    
    with st.form("questions_form"):
        answers = {}
        
        for i, question in enumerate(questions):
            # Display question with clear formatting
            st.markdown(f'<div class="q-head"><span class="q-index">Question {i + 1} of {len(questions)}</span></div>', unsafe_allow_html=True)
            q_label = _question_type_label(question)
            badge_cls = "q-badge"
            if q_label == "Long Answer":
                badge_cls += " long"
            elif q_label == "Short Answer":
                badge_cls += " short"
            st.markdown(f'<span class="{badge_cls}">{q_label}</span>', unsafe_allow_html=True)
            
            # Show question text in a box
            st.markdown(f"""
            <div class="question-card">
                {question['question']}
            </div>
            """, unsafe_allow_html=True)
            
            q_type = str(question.get('type', '')).lower()
            is_mcq = q_type in ('mcq', 'multiple_choice')
            if is_mcq and question.get('options'):
                # Multiple choice
                st.markdown("**Select the correct answer:**")
                options = question['options']
                answer = st.radio(
                    "Options:",
                    options,
                    key=f"q_{i}",
                    index=None,
                    label_visibility="collapsed"
                )
                answers[i] = answer
            else:
                # Open-ended
                st.markdown("**Your answer:**")
                q_height = 120
                if q_type in ('long', 'long_answer'):
                    q_height = 180
                answer = st.text_area(
                    "Type your answer here:",
                    key=f"q_{i}",
                    height=q_height,
                    label_visibility="collapsed",
                    placeholder="Enter your detailed answer here..."
                )
                answers[i] = answer
            
            st.markdown("---")
        
        submitted = st.form_submit_button("Submit Answers", use_container_width=True)
        
        if submitted:
            missing = []
            for idx, question in enumerate(questions):
                value = answers.get(idx)
                q_type = str(question.get('type', '')).lower()
                is_mcq = q_type in ('mcq', 'multiple_choice')
                if is_mcq and value is None:
                    missing.append(idx + 1)
                if (not is_mcq) and (not value or not str(value).strip()):
                    missing.append(idx + 1)

            if missing:
                st.error(f"Please complete all questions before submitting. Missing: {', '.join(map(str, missing))}")
            else:
                st.session_state.answers = answers
                evaluate_answers()
                st.rerun()

def evaluate_answers():
    """Evaluate user answers and update state."""
    from src.llm_service import LLMService
    
    # Check Ollama connection first
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        ollama_available = response.status_code == 200
    except:
        ollama_available = False
    
    if not ollama_available:
        st.error("Cannot connect to Ollama")
        st.warning("""
        **Ollama is not running.** Please start Ollama to enable AI evaluation.
        
        To start Ollama:
        1. Open a terminal
        2. Run: `ollama serve`
        
        Or ensure Ollama is running in the background.
        """)
        st.info("Your answers have been saved. Restart after starting Ollama to get AI evaluation.")
        return
    
    with st.status("Evaluating your answers...", expanded=True) as eval_status:
        show_loading_sequence(eval_status, [
            "Checking objective coverage question-by-question...",
            "Scoring responses against expected concepts...",
            "Compiling checkpoint feedback summary..."
        ])
        llm_service = LLMService()
        questions = st.session_state.questions
        answers = st.session_state.answers
        state = st.session_state.learning_state
        
        verification_results = []
        
        for i, question in enumerate(questions):
            if i % 2 == 0:
                eval_status.write(f"Evaluating response {i+1} of {len(questions)}...")
            user_answer = answers.get(i, "")
            
            if not user_answer or user_answer.strip() == "":
                # Empty answer
                result = {
                    "question_id": question.get('question_id', f'q_{i}'),
                    "question": question['question'],
                    "learner_answer": user_answer,
                    "score": 0.0,
                    "feedback": "No answer provided",
                    "scoring_details": {"score": 0.0, "feedback": "No answer provided", "reasoning": "Empty response"}
                }
                verification_results.append(result)
                continue
            
            q_type = str(question.get('type', '')).lower()
            is_mcq = q_type in ('mcq', 'multiple_choice')
            if is_mcq:
                correct_answer = question.get('correct_answer', '')
                # Check if user answer starts with correct letter
                is_correct = user_answer.strip()[0].upper() == correct_answer.strip()[0].upper()
                
                score = 1.0 if is_correct else 0.0
                feedback = "Correct!" if is_correct else f"Incorrect. The correct answer is {correct_answer}"
                
                scoring = {
                    "score": score,
                    "feedback": feedback,
                    "reasoning": feedback
                }
            else:
                # Open-ended - evaluate with LLM
                try:
                    expected_concepts = question.get('expected_concepts', [])
                    scoring = asyncio.run(llm_service.score_answer(
                        question['question'],
                        user_answer,
                        expected_concepts
                    ))
                except Exception as e:
                    logger.error(f"Error scoring answer: {e}")
                    # Check if it's an Ollama connection error
                    if "10061" in str(e) or "Connection" in str(e):
                        st.warning("Cannot connect to Ollama. Please ensure Ollama is running.")
                        scoring = {
                            "score": 0.7,
                            "feedback": "Your answer shows understanding. (Note: AI evaluation unavailable - Ollama not running)",
                            "reasoning": "Ollama service not available. Please start Ollama to enable AI evaluation."
                        }
                    else:
                        # Other errors - fallback scoring
                        scoring = {
                            "score": 0.5,
                            "feedback": "Unable to evaluate automatically. Manual review needed.",
                            "reasoning": f"Evaluation error: {str(e)}"
                        }
            
            result = {
                "question_id": question.get('question_id', f'q_{i}'),
                "question": question['question'],
                "learner_answer": user_answer,
                "score": scoring['score'],
                "feedback": scoring['feedback'],
                "scoring_details": scoring
            }
            
            verification_results.append(result)
        
        # Calculate score
        total_score = sum(r['score'] for r in verification_results)
        avg_score = total_score / len(verification_results) if verification_results else 0
        score_percentage = avg_score * 100
        
        st.session_state.learning_state['verification_results'] = verification_results
        st.session_state.learning_state['score_percentage'] = score_percentage
        st.session_state.learning_state['meets_threshold'] = score_percentage >= 70.0
        eval_status.update(label="Evaluation complete", state="complete")
        
        st.session_state.stage = 'results'

def render_results():
    """Render assessment results."""
    state = st.session_state.learning_state
    score = state.get('score_percentage', 0)
    meets_threshold = state.get('meets_threshold', False)
    results = state.get('verification_results', [])
    
    st.markdown('<div class="card"><div class="kicker">Outcome</div><h3 style="margin:0;">Assessment Results</h3></div>', unsafe_allow_html=True)

    # Visual score summary: ring on the left, stat tiles on the right
    ring_color = "#1b7a4b" if meets_threshold else "#b04238"
    pct = max(0.0, min(100.0, score))
    done = len(st.session_state.completed_checkpoints)
    total_checkpoints = len(st.session_state.selected_path['checkpoints'])

    left_col, right_col = st.columns([1, 2])
    with left_col:
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:center;height:100%;">'
            '<div class="score-ring" style="--p:%.0f;--ring-c:%s;">'
            '<div class="inner"><div class="val">%.0f%%</div><div class="lbl">Score</div></div>'
            '</div></div>' % (pct, ring_color, pct),
            unsafe_allow_html=True
        )

    with right_col:
        t1, t2 = st.columns(2)
        with t1:
            st.markdown('<div class="tile"><div class="v">70%</div><div class="k">Pass mark</div></div>', unsafe_allow_html=True)
        with t2:
            st.markdown(f'<div class="tile"><div class="v">{len(results)}</div><div class="k">Questions</div></div>', unsafe_allow_html=True)
        t3, t4 = st.columns(2)
        with t3:
            st.markdown(f'<div class="tile"><div class="v">{done} / {total_checkpoints}</div><div class="k">Checkpoints done</div></div>', unsafe_allow_html=True)
        with t4:
            if meets_threshold:
                st.markdown('<div class="tile"><div class="v ok">Passed</div><div class="k">Checkpoint</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="tile"><div class="v bad">Retry</div><div class="k">Checkpoint</div></div>', unsafe_allow_html=True)

    # Result message
    if meets_threshold:
        st.markdown("""
        <div class="success-box">
            <strong>Excellent work.</strong> You cleared this checkpoint — ready to progress to the next one.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-box">
            <strong>Keep learning.</strong> Your score is below the 70% pass mark.
            Review the feedback below — simplified explanations are ready.
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed feedback
    st.subheader("Detailed feedback")

    for i, result in enumerate(results, 1):
        score_pct = result['score'] * 100
        bar_color = "#1b7a4b" if result['score'] >= 0.7 else ("#b7791f" if result['score'] >= 0.5 else "#b04238")
        bar_width = max(3.0, min(100.0, score_pct))

        st.markdown(
            '<div class="qbar-wrap">'
            f'<div class="qbar-label"><span>Q{i} · {(result["question"] or "")[:80]}</span><span class="score">{score_pct:.0f}%</span></div>'
            f'<div class="qbar"><i style="width:{bar_width:.0f}%;--bar-c:{bar_color};"></i></div>'
            '</div>',
            unsafe_allow_html=True
        )

        with st.expander(f"Question {i} · {score_pct:.0f}% — open feedback", expanded=False):
            st.markdown(f"**Question:** {result['question']}")
            st.markdown(f"**Your Answer:** {result['learner_answer']}")
            st.markdown(f"**Feedback:** {result['feedback']}")
    
    # Feynman teaching if failed
    if not meets_threshold:
        render_feynman_teaching()
    
    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if not meets_threshold:
            if st.button("Retry Checkpoint", use_container_width=True):
                reset_checkpoint()
                st.rerun()
    
    with col2:
        if meets_threshold or st.session_state.learning_state.get('feynman_retry_count', 0) >= 3:
            if st.button("Next Checkpoint", use_container_width=True):
                proceed_to_next_checkpoint()
                st.rerun()

def render_feynman_teaching():
    """Render Feynman teaching explanations."""
    st.subheader("Simplified Explanations")
    st.info("Let's break down these concepts in simpler terms...")
    
    from src.feynman_teaching import get_feynman_teacher
    
    feynman_teacher = get_feynman_teacher()
    state = st.session_state.learning_state
    
    # Identify knowledge gaps
    verification_results = state.get('verification_results', [])
    knowledge_gaps = feynman_teacher.identify_knowledge_gaps(verification_results)
    
    if knowledge_gaps:
        # Generate explanations
        with st.status("Generating simplified explanations...", expanded=True) as teach_status:
            show_loading_sequence(teach_status, [
                "Finding low-confidence concepts...",
                "Reframing ideas with simpler mental models...",
                "Preparing targeted explanations for retry..."
            ])
            context_chunks = state.get('processed_context', [])
            
            # Check if Ollama is available before trying
            try:
                # Use run_coroutine_threadsafe or just run in new event loop
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Streamlit already has a running loop
                        import nest_asyncio
                        nest_asyncio.apply()
                except:
                    pass
                
                explanations = asyncio.run(
                    feynman_teacher.generate_all_explanations(knowledge_gaps, context_chunks)
                )
                teach_status.update(label="Simplified explanations ready", state="complete")
            except Exception as e:
                logger.error(f"Error generating Feynman explanations: {e}")
                if "10061" in str(e) or "Connection" in str(e):
                    st.error("Cannot generate explanations. Please ensure Ollama is running.")
                    st.info("Start Ollama with: `ollama serve` in a terminal")
                    explanations = []
                else:
                    st.error(f"Error generating explanations: {e}")
                    explanations = []
        
        # Display explanations
        for i, exp in enumerate(explanations, 1):
            with st.expander(f"Concept {i}: {exp['concept']}", expanded=(i == 1)):
                st.markdown(f"**Your understanding gap — score {exp['score']*100:.0f}%**")
                st.markdown(f"**Question:** {exp['question']}")
                st.markdown(f"**Your answer:** {str(exp['user_answer'])[:120]}...")
                st.markdown("### Simplified explanation")
                st.markdown(exp['simplified_explanation'])

def reset_checkpoint():
    """Reset current checkpoint for retry."""
    # Get retry count before resetting
    retry_count = st.session_state.learning_state.get('feynman_retry_count', 0) if st.session_state.learning_state else 0
    
    st.session_state.learning_state = None
    st.session_state.questions = []
    st.session_state.answers = {}
    st.session_state.study_materials_ready = False
    st.session_state.stage = 'learning'
    
    # Store retry count for next iteration
    if retry_count < 3:
        if 'feynman_retry_metadata' not in st.session_state:
            st.session_state.feynman_retry_metadata = {}
        st.session_state.feynman_retry_metadata['retry_count'] = retry_count + 1

def proceed_to_next_checkpoint():
    """Move to next checkpoint."""
    current_idx = st.session_state.current_checkpoint_index
    st.session_state.completed_checkpoints.append(
        st.session_state.selected_path['checkpoints'][current_idx]['id']
    )
    st.session_state.current_checkpoint_index += 1
    st.session_state.learning_state = None
    st.session_state.questions = []
    st.session_state.answers = {}
    st.session_state.study_materials_ready = False
    st.session_state.stage = 'learning'

def render_completion():
    """Render completion screen."""
    st.markdown(
        '<div class="card" style="text-align:center;padding:2.4rem 1.2rem;">'
        '<p class="kicker" style="margin-bottom:0.6rem;">Path complete</p>'
        '<h2 style="margin:0.2rem 0 0.5rem 0;font-size:2.1rem;">Every checkpoint passed</h2>'
        '<p style="margin:0;color:var(--muted);">You have mastered this learning path — review the results or begin a new one.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    path = st.session_state.selected_path
    completed = len(st.session_state.completed_checkpoints)
    total = len(path['checkpoints'])

    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown(f'<div class="tile"><div class="v">{completed} / {total}</div><div class="k">Checkpoints passed</div></div>', unsafe_allow_html=True)
    with colB:
        st.markdown('<div class="tile"><div class="v ok">100%</div><div class="k">Path complete</div></div>', unsafe_allow_html=True)
    with colC:
        st.markdown(f'<div class="tile"><div class="v">{path["title"][:20]}</div><div class="k">Path</div></div>', unsafe_allow_html=True)

    if st.button("Start a new path", use_container_width=True):
        # Reset everything
        st.session_state.stage = 'select_path'
        st.session_state.selected_path = None
        st.session_state.current_checkpoint_index = 0
        st.session_state.uploaded_files = []
        st.session_state.learning_state = None
        st.session_state.questions = []
        st.session_state.answers = {}
        st.session_state.completed_checkpoints = []
        st.rerun()

def main():
    """Main application entry point."""
    init_session_state()
    render_header()
    render_stage_rail()
    
    # Route to appropriate stage
    if st.session_state.stage == 'select_path':
        render_path_selection()
    elif st.session_state.stage == 'upload_files':
        render_file_upload()
    elif st.session_state.stage == 'study':
        render_study_materials()
    elif st.session_state.stage == 'learning':
        render_learning_session()
    elif st.session_state.stage == 'results':
        render_results()

if __name__ == "__main__":
    # Support both `streamlit run app.py` and accidental `python app.py`.
    # When no Streamlit script context exists, relaunch through Streamlit.
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        is_streamlit_context = get_script_run_ctx() is not None
    except Exception:
        is_streamlit_context = False

    if is_streamlit_context:
        main()
    else:
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", __file__]
        raise SystemExit(stcli.main())
