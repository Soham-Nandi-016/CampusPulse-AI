"""
CampusPulse AI — Admin Dashboard (Streamlit Page)
==================================================
Real-time triage board with:
  - KPI metric cards (total / critical / open / resolved)
  - Category issue heatmap (Plotly)
  - Priority-sorted filterable ticket feed
  - Per-ticket detail expanders with image verification preview
  - 1-click status update / dispatch
  - Fuzzy membership surface visualisation
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    CAMPUS_CATEGORIES,
    CATEGORY_ICONS,
    SEVERITY_COLORS,
    STATUS_OPTIONS,
    get_severity_label,
)
from data.ticket_store import load_all_tickets, update_ticket_status, delete_ticket
from engines.fuzzy_engine import FuzzyEngine


# ─────────────────────────────────────────────────────────────────────────────
# Cached resources
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_fuzzy_engine() -> FuzzyEngine:
    return FuzzyEngine()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _severity_badge(severity: str) -> str:
    color = SEVERITY_COLORS.get(severity, "#94A3B8")
    return (
        f'<span style="background:{color}22;color:{color};border:1px solid {color}44;'
        f'border-radius:12px;padding:2px 10px;font-size:12px;font-weight:600">{severity}</span>'
    )


def _mini_bar(value: float, color: str = "#7C3AED") -> str:
    pct = int(value * 100)
    return (
        f'<div style="background:#1e293b;border-radius:3px;height:5px;width:100%;overflow:hidden">'
        f'<div style="width:{pct}%;height:100%;background:{color};border-radius:3px"></div></div>'
        f'<span style="font-size:10px;color:#64748b">{pct}%</span>'
    )


def _priority_ring(score: float, severity: str) -> str:
    color = SEVERITY_COLORS.get(severity, "#94A3B8")
    pct = int(score)
    dash = pct * 2.51
    return (
        f'<svg viewBox="0 0 80 80" width="80" height="80">'
        f'<circle cx="40" cy="40" r="32" fill="none" stroke="#1e293b" stroke-width="8"/>'
        f'<circle cx="40" cy="40" r="32" fill="none" stroke="{color}" stroke-width="8"'
        f' stroke-dasharray="{dash} {201-dash}"'
        f' stroke-dashoffset="50.3" stroke-linecap="round"'
        f' transform="rotate(-90 40 40)"/>'
        f'<text x="40" y="44" text-anchor="middle" fill="white" font-size="15" font-weight="700">{pct}</text>'
        f'</svg>'
    )


def _format_dt(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso)
        return dt.strftime("%d %b %Y, %I:%M %p")
    except Exception:
        return iso


# ─────────────────────────────────────────────────────────────────────────────
# Main dashboard render
# ─────────────────────────────────────────────────────────────────────────────

def render() -> None:
    # ── Auth gate ─────────────────────────────────────────────────────────────
    if not _admin_auth():
        return

    st.markdown("""
    <div class="page-header">
      <h1>🛡️ Admin Triage Dashboard</h1>
      <p class="page-sub">AI-powered grievance management & real-time severity board</p>
    </div>""", unsafe_allow_html=True)

    tickets = load_all_tickets()  # Already sorted by priority desc

    if not tickets:
        st.info("No tickets found. Ask students to submit complaints via the Student Portal.")
        return

    df = pd.DataFrame(tickets)

    # ── KPI row ───────────────────────────────────────────────────────────────
    _render_kpis(df)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────────────────────
    col_heat, col_sev, col_trend = st.columns([2, 1, 2], gap="large")
    with col_heat:
        _render_category_heatmap(df)
    with col_sev:
        _render_severity_donut(df)
    with col_trend:
        _render_priority_timeline(df)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Filters & ticket feed ─────────────────────────────────────────────────
    _render_ticket_feed(tickets, df)

    # ── Fuzzy membership surface ──────────────────────────────────────────────
    with st.expander("📐 Fuzzy Membership Functions (Mamdani FIS)", expanded=False):
        _render_fuzzy_surfaces()


# ─────────────────────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────────────────────

def _admin_auth() -> bool:
    if st.session_state.get("admin_authenticated"):
        return True

    st.markdown('<div class="card auth-card">', unsafe_allow_html=True)
    st.markdown("### 🔐 Admin Authentication")
    st.caption("Enter the admin password to access the triage dashboard.")
    pwd = st.text_input("Password", type="password", key="admin_pwd_input")
    if st.button("Login", key="admin_login_btn"):
        from config import ADMIN_PASSWORD
        if pwd == ADMIN_PASSWORD:
            st.session_state.admin_authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.markdown("</div>", unsafe_allow_html=True)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# KPI cards
# ─────────────────────────────────────────────────────────────────────────────

def _render_kpis(df: pd.DataFrame) -> None:
    total = len(df)
    critical = int((df["severity"] == "Critical").sum())
    high = int((df["severity"] == "High").sum())
    open_cnt = int((df["status"] == "Open").sum())
    resolved = int(df["status"].isin(["Resolved", "Closed"]).sum())
    avg_priority = round(df["priority_score"].mean(), 1)
    resolve_rate = round(resolved / total * 100, 1) if total else 0

    cols = st.columns(6)
    kpis = [
        ("📋", "Total Tickets", str(total), "#7C3AED"),
        ("🚨", "Critical", str(critical), "#EF4444"),
        ("⚠️", "High Priority", str(high), "#F97316"),
        ("📂", "Open", str(open_cnt), "#06B6D4"),
        ("✅", "Resolved", str(resolved), "#22C55E"),
        ("📊", "Avg Priority", str(avg_priority), "#8B5CF6"),
    ]
    for col, (icon, label, val, color) in zip(cols, kpis):
        with col:
            st.markdown(
                f"""<div class="kpi-card" style="border-top:3px solid {color}">
                  <div class="kpi-icon">{icon}</div>
                  <div class="kpi-value" style="color:{color}">{val}</div>
                  <div class="kpi-label">{label}</div>
                </div>""",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────

def _render_category_heatmap(df: pd.DataFrame) -> None:
    st.markdown("**Category × Severity Heatmap**")
    if "category" not in df.columns:
        return

    heat = pd.crosstab(df["category"], df["severity"])
    for sev in ["Low", "Medium", "High", "Critical"]:
        if sev not in heat.columns:
            heat[sev] = 0
    heat = heat[["Low", "Medium", "High", "Critical"]]

    fig = px.imshow(
        heat,
        color_continuous_scale=[
            [0.0, "#0f172a"], [0.4, "#312e81"], [0.7, "#7C3AED"], [1.0, "#EF4444"]
        ],
        text_auto=True,
        aspect="auto",
    )
    fig.update_layout(
        height=240,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
        coloraxis_showscale=False,
    )
    fig.update_traces(textfont_color="white", textfont_size=14)
    st.plotly_chart(fig, use_container_width=True)


def _render_severity_donut(df: pd.DataFrame) -> None:
    st.markdown("**Severity Split**")
    counts = df["severity"].value_counts()
    colors = [SEVERITY_COLORS.get(s, "#94A3B8") for s in counts.index]
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        hole=0.65,
        marker_colors=colors,
        textinfo="label+percent",
        textfont_size=11,
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font_color="#cbd5e1",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_priority_timeline(df: pd.DataFrame) -> None:
    st.markdown("**Priority Score Distribution**")
    fig = px.histogram(
        df,
        x="priority_score",
        nbins=20,
        color="severity",
        color_discrete_map=SEVERITY_COLORS,
        labels={"priority_score": "Priority Score (0–100)", "count": "Tickets"},
    )
    fig.update_layout(
        height=240,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
        showlegend=False,
        bargap=0.1,
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Ticket feed
# ─────────────────────────────────────────────────────────────────────────────

def _render_ticket_feed(tickets: list[dict], df: pd.DataFrame) -> None:
    st.markdown("---")
    st.markdown("### 📋 Grievance Triage Board")

    # ── Filters ───────────────────────────────────────────────────────────────
    fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 1])
    with fc1:
        cat_filter = st.multiselect(
            "Category", CAMPUS_CATEGORIES,
            default=[], placeholder="All categories", key="af_cat"
        )
    with fc2:
        sev_filter = st.multiselect(
            "Severity", ["Critical", "High", "Medium", "Low"],
            default=[], placeholder="All severities", key="af_sev"
        )
    with fc3:
        status_filter = st.multiselect(
            "Status", STATUS_OPTIONS,
            default=[], placeholder="All statuses", key="af_status"
        )
    with fc4:
        sort_by = st.selectbox(
            "Sort", ["Priority ↓", "Priority ↑", "Newest", "Oldest"],
            key="af_sort"
        )

    # Apply filters
    filtered = tickets[:]
    if cat_filter:
        filtered = [t for t in filtered if t.get("category") in cat_filter]
    if sev_filter:
        filtered = [t for t in filtered if t.get("severity") in sev_filter]
    if status_filter:
        filtered = [t for t in filtered if t.get("status") in status_filter]

    if sort_by == "Priority ↑":
        filtered.sort(key=lambda t: t.get("priority_score", 0))
    elif sort_by == "Newest":
        filtered.sort(key=lambda t: t.get("created_at", ""), reverse=True)
    elif sort_by == "Oldest":
        filtered.sort(key=lambda t: t.get("created_at", ""))
    else:
        filtered.sort(key=lambda t: t.get("priority_score", 0), reverse=True)

    st.caption(f"Showing **{len(filtered)}** of {len(tickets)} tickets")

    # ── Ticket cards ──────────────────────────────────────────────────────────
    for ticket in filtered:
        _render_ticket_card(ticket)


def _render_ticket_card(ticket: dict) -> None:
    tid = ticket.get("ticket_id", "—")
    sev = ticket.get("severity", "Low")
    color = SEVERITY_COLORS.get(sev, "#94A3B8")
    cat = ticket.get("category", "—")
    icon = CATEGORY_ICONS.get(cat, "📌")
    priority = ticket.get("priority_score", 0)
    status = ticket.get("status", "Open")

    with st.expander(
        f"{_priority_ring(priority, sev)} &nbsp; **{tid}** — {icon} {cat} &nbsp; {_severity_badge(sev)}",
        expanded=False,
    ):
        c1, c2 = st.columns([3, 1])

        with c1:
            st.markdown(f"**👤 Student:** {ticket.get('student_name','—')} ({ticket.get('student_id','—')})")
            st.markdown(f"**🏫 Department:** {ticket.get('department','—')}")
            st.markdown(f"**🕐 Submitted:** {_format_dt(ticket.get('created_at',''))}")
            st.markdown(f"**📝 Description:**")
            st.markdown(
                f'<div class="complaint-text">{ticket.get("description","—")}</div>',
                unsafe_allow_html=True,
            )
            # Keywords
            kws = ticket.get("keywords", [])
            if kws:
                kw_html = " ".join(
                    f'<span class="keyword-chip">{k}</span>' for k in kws
                )
                st.markdown(f'<div class="kw-row">{kw_html}</div>', unsafe_allow_html=True)

        with c2:
            st.markdown("**AI Scores**")
            st.markdown(
                _mini_bar(ticket.get("urgency_score", 0), "#EF4444"),
                unsafe_allow_html=True,
            )
            st.caption("Urgency")
            st.markdown(
                _mini_bar(ticket.get("negativity_score", 0), "#F97316"),
                unsafe_allow_html=True,
            )
            st.caption("Negativity")
            st.markdown(
                _mini_bar(ticket.get("visual_conf", 0), "#06B6D4"),
                unsafe_allow_html=True,
            )
            st.caption("Visual Conf.")

            if ticket.get("has_image"):
                st.markdown(
                    f'<div class="vis-badge">📸 {ticket.get("defect_label","—")}</div>',
                    unsafe_allow_html=True,
                )

            sentiment = ticket.get("sentiment_label", "—")
            st.markdown(
                f'<div class="sentiment-badge sentiment-{sentiment.lower()}">{sentiment}</div>',
                unsafe_allow_html=True,
            )

        # ── Status update ─────────────────────────────────────────────────────
        st.markdown("---")
        sc1, sc2, sc3 = st.columns([2, 1, 1])
        with sc1:
            new_status = st.selectbox(
                "Update Status",
                STATUS_OPTIONS,
                index=STATUS_OPTIONS.index(status) if status in STATUS_OPTIONS else 0,
                key=f"status_{tid}",
            )
        with sc2:
            if st.button("✅ Save", key=f"save_{tid}", use_container_width=True):
                if update_ticket_status(tid, new_status):
                    st.success("Status updated.")
                    st.rerun()
        with sc3:
            if st.button("🗑️ Delete", key=f"del_{tid}", use_container_width=True):
                if delete_ticket(tid):
                    st.warning(f"Ticket {tid} deleted.")
                    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Fuzzy membership surface plot
# ─────────────────────────────────────────────────────────────────────────────

def _render_fuzzy_surfaces() -> None:
    fuzzy = get_fuzzy_engine()
    surfaces = fuzzy.get_membership_surface()

    if not surfaces:
        st.info("Install scikit-fuzzy to see membership function plots.")
        return

    input_sets  = {k: v for k, v in surfaces.items() if k.startswith("Input")}
    output_sets = {k: v for k, v in surfaces.items() if k.startswith("Output")}

    col_in, col_out = st.columns(2)

    with col_in:
        st.markdown("**Input Membership Functions**")
        fig = go.Figure()
        colors = {"Input: Low": "#22C55E", "Input: Medium": "#EAB308", "Input: High": "#EF4444"}
        for name, (x, y) in input_sets.items():
            fig.add_trace(go.Scatter(
                x=x, y=y, name=name.replace("Input: ", ""),
                fill="tozeroy", fillcolor=colors.get(name, "#7C3AED") + "33",
                line=dict(color=colors.get(name, "#7C3AED"), width=2),
            ))
        _apply_fuzzy_layout(fig, xaxis_title="Input Value [0–1]")
        st.plotly_chart(fig, use_container_width=True)

    with col_out:
        st.markdown("**Output Priority Membership Functions**")
        fig2 = go.Figure()
        out_colors = {
            "Output: Low": "#22C55E", "Output: Medium": "#EAB308",
            "Output: High": "#F97316", "Output: Critical": "#EF4444"
        }
        for name, (x, y) in output_sets.items():
            fig2.add_trace(go.Scatter(
                x=x, y=y, name=name.replace("Output: ", ""),
                fill="tozeroy",
                fillcolor=out_colors.get(name, "#7C3AED") + "33",
                line=dict(color=out_colors.get(name, "#7C3AED"), width=2),
            ))
        _apply_fuzzy_layout(fig2, xaxis_title="Priority Score [0–100]")
        st.plotly_chart(fig2, use_container_width=True)


def _apply_fuzzy_layout(fig: go.Figure, xaxis_title: str = "") -> None:
    fig.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0b1120",
        font_color="#cbd5e1",
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
        xaxis=dict(title=xaxis_title, gridcolor="#1e293b"),
        yaxis=dict(title="Membership Degree", range=[0, 1.05], gridcolor="#1e293b"),
    )
