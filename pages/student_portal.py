"""
CampusPulse AI — Student Submission Portal (Streamlit Page)
============================================================
Multi-input complaint form with:
  - Student identification fields
  - Auto-category suggestion from NLP engine
  - Image drag-and-drop uploader with preview
  - Real-time analysis pipeline (NLP + Vision + Fuzzy)
  - Ticket confirmation with urgency estimate
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CAMPUS_CATEGORIES, CATEGORY_ICONS, SEVERITY_COLORS
from engines.nlp_engine import NLPEngine
from engines.vision_engine import VisionEngine
from engines.fuzzy_engine import FuzzyEngine
from data.ticket_store import make_ticket, save_ticket


# ─────────────────────────────────────────────────────────────────────────────
# Cached engine singletons
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_nlp_engine() -> NLPEngine:
    return NLPEngine()

@st.cache_resource(show_spinner=False)
def get_vision_engine() -> VisionEngine:
    return VisionEngine()

@st.cache_resource(show_spinner=False)
def get_fuzzy_engine() -> FuzzyEngine:
    return FuzzyEngine()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: score ring gauge
# ─────────────────────────────────────────────────────────────────────────────

def _priority_gauge(score: float, severity: str) -> str:
    color = SEVERITY_COLORS.get(severity, "#94A3B8")
    pct = int(score)
    return f"""
    <div class="gauge-wrap">
      <svg viewBox="0 0 120 120" width="130" height="130">
        <circle cx="60" cy="60" r="50" fill="none" stroke="#1e293b" stroke-width="12"/>
        <circle cx="60" cy="60" r="50" fill="none" stroke="{color}" stroke-width="12"
          stroke-dasharray="{pct * 3.14} {314 - pct * 3.14}"
          stroke-dashoffset="78.5" stroke-linecap="round"
          transform="rotate(-90 60 60)"/>
        <text x="60" y="58" text-anchor="middle" fill="white" font-size="22" font-weight="700">{pct}</text>
        <text x="60" y="76" text-anchor="middle" fill="{color}" font-size="11">{severity.upper()}</text>
      </svg>
    </div>"""


def _bar(label: str, value: float, color: str = "#7C3AED") -> str:
    pct = int(value * 100)
    return f"""
    <div style="margin:6px 0">
      <div style="display:flex;justify-content:space-between;font-size:12px;color:#94a3b8;margin-bottom:3px">
        <span>{label}</span><span>{pct}%</span>
      </div>
      <div style="background:#1e293b;border-radius:4px;height:7px;overflow:hidden">
        <div style="width:{pct}%;height:100%;background:{color};border-radius:4px;
             transition:width 0.6s ease"></div>
      </div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main portal page
# ─────────────────────────────────────────────────────────────────────────────

def render() -> None:
    st.markdown("""
    <div class="page-header">
      <h1>📝 Student Grievance Portal</h1>
      <p class="page-sub">Submit your complaint and our AI will triage it instantly</p>
    </div>""", unsafe_allow_html=True)

    # ── Session state ─────────────────────────────────────────────────────────
    if "submitted" not in st.session_state:
        st.session_state.submitted = False
    if "live_result" not in st.session_state:
        st.session_state.live_result = None

    if st.session_state.submitted:
        _render_confirmation()
        return

    # ── Layout: form (left) | live analysis (right) ───────────────────────────
    col_form, col_analysis = st.columns([3, 2], gap="large")

    with col_form:
        _render_form()

    with col_analysis:
        _render_live_analysis()


# ─────────────────────────────────────────────────────────────────────────────
# Form
# ─────────────────────────────────────────────────────────────────────────────

def _render_form() -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 👤 Student Information")

    c1, c2 = st.columns(2)
    with c1:
        student_name = st.text_input("Full Name", placeholder="e.g. Arjun Sharma",
                                     key="f_name")
    with c2:
        student_id = st.text_input("Student ID", placeholder="e.g. STU20231042",
                                   key="f_id")

    department = st.selectbox("Department", [
        "Computer Science", "Mechanical Engineering", "Civil Engineering",
        "Electronics", "Information Technology", "Architecture", "MBA", "Other",
    ], key="f_dept")

    st.markdown("---")
    st.markdown("### 📋 Complaint Details")

    description = st.text_area(
        "Describe your issue",
        height=150,
        placeholder="Describe the problem in detail. Be specific about location, duration, and impact…",
        key="f_desc",
    )

    # Live NLP hint — run when user has typed enough
    if description and len(description) > 20:
        nlp = get_nlp_engine()
        result = nlp.analyse(description)
        suggested = result.top_category
        st.markdown(
            f'<div class="ai-hint">🤖 AI suggests category: '
            f'<strong>{CATEGORY_ICONS.get(suggested, "")} {suggested}</strong> '
            f'(confidence {result.category_probs.get(suggested, 0)*100:.0f}%)</div>',
            unsafe_allow_html=True
        )
        st.session_state.live_result = result
    else:
        suggested = CAMPUS_CATEGORIES[0]

    category = st.selectbox(
        "Category",
        CAMPUS_CATEGORIES,
        index=CAMPUS_CATEGORIES.index(suggested) if suggested in CAMPUS_CATEGORIES else 0,
        format_func=lambda c: f"{CATEGORY_ICONS.get(c, '📌')} {c}",
        key="f_cat",
    )

    st.markdown("---")
    st.markdown("### 📎 Attach Proof (Optional)")
    uploaded = st.file_uploader(
        "Drag & drop or click to upload",
        type=["jpg", "jpeg", "png", "webp"],
        key="f_img",
        help="Upload a photo showing the defect or issue",
    )

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded proof image", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Submit ────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Submit Complaint", type="primary", use_container_width=True, key="btn_submit"):
        _handle_submission(
            student_name=student_name,
            student_id=student_id,
            department=department,
            category=category,
            description=description,
            uploaded_file=uploaded,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Live analysis panel
# ─────────────────────────────────────────────────────────────────────────────

def _render_live_analysis() -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ⚡ Live AI Analysis")

    result = st.session_state.get("live_result")

    if not result:
        st.markdown(
            '<div class="empty-hint">Start typing your complaint to see real-time AI analysis here…</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    nlp_result = result

    # Category distribution
    st.markdown("**Category Probability Distribution**")
    for cat, prob in sorted(nlp_result.category_probs.items(), key=lambda x: -x[1]):
        icon = CATEGORY_ICONS.get(cat, "📌")
        color = "#7C3AED" if prob == max(nlp_result.category_probs.values()) else "#06B6D4"
        st.markdown(_bar(f"{icon} {cat}", prob, color), unsafe_allow_html=True)

    st.markdown("---")

    # Scores
    st.markdown("**Detected Signals**")
    st.markdown(_bar("🔥 Urgency Score", nlp_result.urgency_score, "#EF4444"), unsafe_allow_html=True)
    st.markdown(_bar("😤 Negativity Score", nlp_result.negativity_score, "#F97316"), unsafe_allow_html=True)
    st.markdown(_bar("🌀 Ambiguity (Entropy)", nlp_result.entropy, "#8B5CF6"), unsafe_allow_html=True)

    if nlp_result.keywords:
        st.markdown("**Flagged Keywords**")
        kw_html = " ".join(
            f'<span class="keyword-chip">{k}</span>' for k in nlp_result.keywords
        )
        st.markdown(f'<div class="kw-row">{kw_html}</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="sentiment-badge sentiment-{nlp_result.sentiment_label.lower()}">'
        f'Sentiment: {nlp_result.sentiment_label}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Submission handler
# ─────────────────────────────────────────────────────────────────────────────

def _handle_submission(
    student_name: str,
    student_id: str,
    department: str,
    category: str,
    description: str,
    uploaded_file,
) -> None:
    """Validate, run ML pipeline, save ticket."""
    errors = []
    if not student_name.strip():
        errors.append("Full Name is required.")
    if not student_id.strip():
        errors.append("Student ID is required.")
    if not description.strip() or len(description.strip()) < 20:
        errors.append("Please describe the issue in at least 20 characters.")

    if errors:
        for e in errors:
            st.error(e)
        return

    with st.spinner("🧠 Running AI Triage Pipeline…"):
        # 1) NLP
        nlp = get_nlp_engine()
        nlp_result = nlp.analyse(description)

        # 2) Vision
        vision = get_vision_engine()
        image_bytes = None
        if uploaded_file:
            image_bytes = uploaded_file.read()
            img = Image.open(io.BytesIO(image_bytes))
            vis_result = vision.analyse(img)
        else:
            from engines.vision_engine import VisionResult
            vis_result = VisionResult(confidence=0.0, defect_label="No Image Provided")

        # 3) Fuzzy triage
        fuzzy = get_fuzzy_engine()
        fuzzy_result = fuzzy.compute(
            text_urgency=nlp_result.urgency_score,
            negativity=nlp_result.negativity_score,
            visual_conf=vis_result.confidence,
        )

        # 4) Persist
        ticket = make_ticket(
            student_name=student_name,
            student_id=student_id,
            department=department,
            category=category,
            description=description,
            urgency_score=nlp_result.urgency_score,
            negativity_score=nlp_result.negativity_score,
            visual_conf=vis_result.confidence,
            priority_score=fuzzy_result.priority_score,
            top_category=nlp_result.top_category,
            sentiment_label=nlp_result.sentiment_label,
            entropy=nlp_result.entropy,
            keywords=nlp_result.keywords,
            defect_label=vis_result.defect_label,
            has_image=uploaded_file is not None,
            image_bytes=image_bytes,
        )
        ticket["_nlp"] = nlp_result
        ticket["_vis"] = vis_result
        ticket["_fuzzy"] = fuzzy_result
        ticket_id = save_ticket(ticket)

    st.session_state.submitted = True
    st.session_state.last_ticket = ticket
    st.session_state.last_ticket_id = ticket_id
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Confirmation screen
# ─────────────────────────────────────────────────────────────────────────────

def _render_confirmation() -> None:
    ticket = st.session_state.get("last_ticket", {})
    ticket_id = st.session_state.get("last_ticket_id", "CP-XXXXXX")

    priority = ticket.get("priority_score", 0)
    severity = ticket.get("severity", "Low")
    color = SEVERITY_COLORS.get(severity, "#94A3B8")

    st.markdown(f"""
    <div class="confirmation-card">
      <div class="check-circle">✓</div>
      <h2>Complaint Submitted Successfully</h2>
      <div class="ticket-id-badge">Ticket ID: <strong>{ticket_id}</strong></div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Priority Score**")
        st.markdown(_priority_gauge(priority, severity), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**AI Analysis**")
        st.markdown(_bar("🔥 Urgency", ticket.get("urgency_score", 0), "#EF4444"),
                    unsafe_allow_html=True)
        st.markdown(_bar("😤 Negativity", ticket.get("negativity_score", 0), "#F97316"),
                    unsafe_allow_html=True)
        st.markdown(_bar("📸 Visual Conf.", ticket.get("visual_conf", 0), "#06B6D4"),
                    unsafe_allow_html=True)
        kws = ticket.get("keywords", [])
        if kws:
            st.markdown(
                '<div class="kw-row">' +
                " ".join(f'<span class="keyword-chip">{k}</span>' for k in kws) +
                "</div>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Ticket Details**")
        st.markdown(f"**Category:** {CATEGORY_ICONS.get(ticket.get('category',''),'📌')} {ticket.get('category','—')}")
        st.markdown(f"**Status:** `Open`")
        st.markdown(f"**Severity:** <span style='color:{color}'>{severity}</span>",
                    unsafe_allow_html=True)
        st.markdown(f"**Dept:** {ticket.get('department','—')}")
        if ticket.get("has_image"):
            st.markdown(f"**Visual:** {ticket.get('defect_label','—')}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.info(
        f"**Expected Resolution:** {'⚠️ Priority escalation within 2 hours' if severity == 'Critical' else '📅 Standard queue — 24–72 hours'}"
    )

    if st.button("📝 Submit Another Complaint", use_container_width=True):
        for k in ["submitted", "last_ticket", "last_ticket_id", "live_result"]:
            st.session_state.pop(k, None)
        st.rerun()
