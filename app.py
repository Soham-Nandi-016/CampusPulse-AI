"""
CampusPulse AI — Main Streamlit Application Entry Point
=======================================================
Run with:
    streamlit run app.py

Features:
  - Custom dark theme with glassmorphism design system
  - Single-page routing via sidebar navigation
  - Cached ML engine singletons (load once, run many)
  - Auto-seeds mock data on first run
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# ── Streamlit page config — must be first Streamlit call ─────────────────────
st.set_page_config(
    page_title="CampusPulse AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "**CampusPulse AI** — Multimodal Campus Grievance Redressal System",
    },
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import APP_TITLE
from data.ticket_store import seed_mock_data


# ─────────────────────────────────────────────────────────────────────────────
# Design System — Global CSS
# ─────────────────────────────────────────────────────────────────────────────

GLOBAL_CSS = """
<style>
/* ── Google Fonts ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── CSS Variables ───────────────────────────────────────────────────────── */
:root {
  --bg-base:        #060b14;
  --bg-surface:     #0d1627;
  --bg-card:        #111827;
  --bg-card-hover:  #162032;
  --bg-input:       #0f172a;
  --border:         rgba(255,255,255,0.07);
  --border-hover:   rgba(124,58,237,0.4);
  --primary:        #7C3AED;
  --primary-light:  #9F67F5;
  --accent:         #06B6D4;
  --accent2:        #10B981;
  --text-primary:   #f1f5f9;
  --text-secondary: #94a3b8;
  --text-muted:     #475569;
  --radius:         12px;
  --radius-sm:      8px;
  --shadow:         0 4px 24px rgba(0,0,0,0.4);
  --glow-purple:    0 0 20px rgba(124,58,237,0.25);
  --glow-cyan:      0 0 20px rgba(6,182,212,0.2);
}

/* ── Base Overrides ──────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
  background: var(--bg-base) !important;
  font-family: 'Inter', sans-serif !important;
  color: var(--text-primary) !important;
}

[data-testid="stSidebar"] {
  background: var(--bg-surface) !important;
  border-right: 1px solid var(--border) !important;
}

/* ── Hide Streamlit chrome ───────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--primary); border-radius: 3px; }

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] .block-container {
  padding: 1.5rem 1rem !important;
}

.sidebar-logo {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 4px 20px 4px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 20px;
}
.sidebar-logo-icon {
  width: 42px; height: 42px;
  background: linear-gradient(135deg, var(--primary), var(--accent));
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 22px;
  box-shadow: var(--glow-purple);
}
.sidebar-logo-text { font-size: 16px; font-weight: 700; line-height: 1.2; }
.sidebar-logo-sub { font-size: 11px; color: var(--text-muted); }

.nav-section-label {
  font-size: 10px; font-weight: 600; letter-spacing: 1.2px;
  color: var(--text-muted); text-transform: uppercase;
  margin: 16px 4px 8px 4px;
}
.nav-item {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 12px;
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-size: 14px; font-weight: 500; color: var(--text-secondary);
  transition: all 0.2s ease;
  border: 1px solid transparent;
  margin-bottom: 4px;
}
.nav-item:hover {
  background: rgba(124,58,237,0.1);
  color: var(--text-primary);
  border-color: var(--border-hover);
}
.nav-item.active {
  background: linear-gradient(135deg, rgba(124,58,237,0.2), rgba(6,182,212,0.1));
  color: var(--primary-light);
  border-color: rgba(124,58,237,0.3);
  box-shadow: var(--glow-purple);
}
.sidebar-divider {
  border: none; border-top: 1px solid var(--border);
  margin: 16px 0;
}

/* ── Page Header ─────────────────────────────────────────────────────────── */
.page-header {
  padding: 0 0 24px 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 28px;
}
.page-header h1 {
  font-size: 28px !important;
  font-weight: 800 !important;
  background: linear-gradient(135deg, var(--primary-light), var(--accent));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 0 4px 0 !important;
}
.page-sub { color: var(--text-muted); font-size: 14px; margin: 0; }

/* ── Cards ───────────────────────────────────────────────────────────────── */
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
  margin-bottom: 16px;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.card:hover {
  border-color: var(--border-hover);
  box-shadow: var(--glow-purple);
}

/* ── KPI Cards ───────────────────────────────────────────────────────────── */
.kpi-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 16px;
  text-align: center;
  transition: all 0.25s ease;
}
.kpi-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow);
  border-color: var(--border-hover);
}
.kpi-icon { font-size: 22px; margin-bottom: 6px; }
.kpi-value { font-size: 28px; font-weight: 800; line-height: 1; margin-bottom: 4px; }
.kpi-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.8px; }

/* ── Gauge ───────────────────────────────────────────────────────────────── */
.gauge-wrap {
  display: flex; justify-content: center; align-items: center;
  padding: 12px 0;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {
  background: linear-gradient(135deg, var(--primary), #5B21B6) !important;
  color: white !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  padding: 0.5rem 1.2rem !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 2px 8px rgba(124,58,237,0.3) !important;
}
.stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 16px rgba(124,58,237,0.5) !important;
  filter: brightness(1.1) !important;
}
.stButton > button[kind="secondary"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-secondary) !important;
}

/* ── Form inputs ─────────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
  background: var(--bg-input) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  font-family: 'Inter', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
}
.stTextInput label, .stTextArea label, .stSelectbox label,
.stFileUploader label, .stMultiSelect label {
  color: var(--text-secondary) !important;
  font-weight: 500 !important;
  font-size: 13px !important;
}

/* ── File uploader ───────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
  background: var(--bg-input) !important;
  border: 2px dashed rgba(124,58,237,0.4) !important;
  border-radius: var(--radius) !important;
  transition: border-color 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--primary) !important;
  background: rgba(124,58,237,0.05) !important;
}

/* ── AI hint box ─────────────────────────────────────────────────────────── */
.ai-hint {
  background: linear-gradient(135deg, rgba(124,58,237,0.1), rgba(6,182,212,0.08));
  border: 1px solid rgba(124,58,237,0.3);
  border-radius: var(--radius-sm);
  padding: 8px 14px;
  font-size: 13px;
  color: var(--text-secondary);
  margin: 6px 0 10px 0;
}

/* ── Keyword chips ────────────────────────────────────────────────────────── */
.kw-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
.keyword-chip {
  background: rgba(239,68,68,0.12);
  color: #FCA5A5;
  border: 1px solid rgba(239,68,68,0.25);
  border-radius: 20px;
  padding: 2px 10px;
  font-size: 11px;
  font-weight: 500;
}

/* ── Sentiment badge ─────────────────────────────────────────────────────── */
.sentiment-badge {
  display: inline-block;
  padding: 3px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 600;
  margin-top: 8px;
}
.sentiment-negative { background: rgba(239,68,68,0.15); color: #FCA5A5; border: 1px solid rgba(239,68,68,0.3); }
.sentiment-mixed    { background: rgba(234,179,8,0.15);  color: #FDE68A; border: 1px solid rgba(234,179,8,0.3); }
.sentiment-neutral  { background: rgba(100,116,139,0.15);color: #94a3b8; border: 1px solid rgba(100,116,139,0.3); }

/* ── Visual defect badge ─────────────────────────────────────────────────── */
.vis-badge {
  background: rgba(6,182,212,0.12);
  color: #67E8F9;
  border: 1px solid rgba(6,182,212,0.3);
  border-radius: var(--radius-sm);
  padding: 4px 10px;
  font-size: 11px;
  margin-top: 6px;
}

/* ── Confirmation card ───────────────────────────────────────────────────── */
.confirmation-card {
  text-align: center;
  padding: 40px 20px;
  background: linear-gradient(135deg, rgba(124,58,237,0.08), rgba(6,182,212,0.06));
  border: 1px solid rgba(124,58,237,0.25);
  border-radius: var(--radius);
  margin-bottom: 24px;
}
.check-circle {
  width: 64px; height: 64px;
  background: linear-gradient(135deg, #10B981, #059669);
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 28px; color: white; font-weight: 800;
  margin: 0 auto 16px auto;
  box-shadow: 0 0 24px rgba(16,185,129,0.4);
}
.ticket-id-badge {
  background: rgba(124,58,237,0.15);
  border: 1px solid rgba(124,58,237,0.3);
  border-radius: 8px;
  padding: 8px 24px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 16px;
  display: inline-block;
  margin-top: 8px;
  color: var(--primary-light);
}

/* ── Complaint text ──────────────────────────────────────────────────────── */
.complaint-text {
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 12px;
  font-size: 14px;
  line-height: 1.6;
  color: var(--text-secondary);
  margin: 8px 0;
}

/* ── Auth card ───────────────────────────────────────────────────────────── */
.auth-card {
  max-width: 400px;
  margin: 80px auto;
}

/* ── Empty hint ──────────────────────────────────────────────────────────── */
.empty-hint {
  text-align: center;
  color: var(--text-muted);
  font-size: 13px;
  padding: 40px 16px;
  border: 2px dashed var(--border);
  border-radius: var(--radius);
}

/* ── Expander override ───────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  margin-bottom: 10px !important;
}
[data-testid="stExpander"]:hover {
  border-color: var(--border-hover) !important;
}
details summary {
  padding: 14px 16px !important;
  font-size: 14px !important;
}

/* ── Plotly chart background ─────────────────────────────────────────────── */
[data-testid="stPlotlyChart"] {
  border-radius: var(--radius) !important;
  overflow: hidden;
}

/* ── Alerts / info ───────────────────────────────────────────────────────── */
.stAlert {
  background: var(--bg-card) !important;
  border-radius: var(--radius-sm) !important;
}

/* ── Radio buttons for nav ────────────────────────────────────────────────── */
.stRadio > div {
  gap: 0 !important;
}
.stRadio > div > label {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  padding: 10px 14px !important;
  margin-bottom: 6px !important;
  cursor: pointer !important;
  color: var(--text-secondary) !important;
  font-size: 14px !important;
  transition: all 0.2s !important;
  display: flex !important;
  align-items: center !important;
}
.stRadio > div > label:hover {
  border-color: var(--border-hover) !important;
  background: rgba(124,58,237,0.08) !important;
  color: var(--text-primary) !important;
}
.stRadio [data-checked="true"] > div {
  color: var(--primary-light) !important;
}

/* ── Spinner ─────────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] {
  color: var(--primary) !important;
}

/* ── Responsive ──────────────────────────────────────────────────────────── */
@media (max-width: 768px) {
  .kpi-value { font-size: 20px !important; }
}
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────

def _render_sidebar() -> str:
    st.sidebar.markdown(
        """
        <div class="sidebar-logo">
          <div class="sidebar-logo-icon">🎓</div>
          <div>
            <div class="sidebar-logo-text">CampusPulse AI</div>
            <div class="sidebar-logo-sub">Grievance Intelligence System</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown('<div class="nav-section-label">Navigation</div>', unsafe_allow_html=True)

    page = st.sidebar.radio(
        label="",
        options=["📝 Student Portal", "🛡️ Admin Dashboard"],
        key="nav_page",
        label_visibility="collapsed",
    )

    st.sidebar.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="nav-section-label">System Status</div>', unsafe_allow_html=True)

    from config import USE_MOCK_MODELS
    mode_color = "#EAB308" if USE_MOCK_MODELS else "#22C55E"
    mode_label = "Mock / Demo Mode" if USE_MOCK_MODELS else "Production Mode"
    st.sidebar.markdown(
        f'<div style="font-size:12px;color:{mode_color};padding:4px 0">● {mode_label}</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Set `USE_MOCK_MODELS=false` in `.env` to activate real ML models.")

    st.sidebar.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="nav-section-label">Engine Stack</div>', unsafe_allow_html=True)
    for label, detail in [
        ("🔤 NLP", "DistilBERT ABSA"),
        ("👁️ Vision", "MobileNetV2"),
        ("🔀 Fuzzy", "Mamdani FIS"),
    ]:
        st.sidebar.markdown(
            f'<div style="font-size:12px;padding:3px 0;color:#64748b">'
            f'<span style="color:#94a3b8">{label}</span> — {detail}</div>',
            unsafe_allow_html=True,
        )

    return page


# ─────────────────────────────────────────────────────────────────────────────
# App entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # Inject global CSS
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # Seed demo data once
    if "seeded" not in st.session_state:
        seed_mock_data(n=12)
        st.session_state.seeded = True

    page = _render_sidebar()

    # Route to page
    if "Student Portal" in page:
        from pages.student_portal import render as render_portal
        render_portal()
    else:
        from pages.admin_dashboard import render as render_admin
        render_admin()


if __name__ == "__main__":
    main()
