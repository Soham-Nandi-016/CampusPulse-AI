"""
CampusPulse AI — Data Layer
============================
Handles ticket persistence (SQLite via sqlitedict), mock data generation,
and CRUD operations used by both the student portal and admin dashboard.
"""

from __future__ import annotations

import uuid
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from loguru import logger

# ── Optional sqlitedict ───────────────────────────────────────────────────────
_sqlitedict_available = False
try:
    from sqlitedict import SqliteDict  # type: ignore

    _sqlitedict_available = True
except ImportError:
    logger.warning("sqlitedict not installed — using in-memory store.")

# ── Config ────────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CAMPUS_CATEGORIES, STATUS_OPTIONS, get_severity_label

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "tickets.sqlite"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# In-memory fallback
_memory_store: dict[str, dict] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Ticket schema
# ─────────────────────────────────────────────────────────────────────────────

def make_ticket(
    *,
    student_name: str,
    student_id: str,
    department: str,
    category: str,
    description: str,
    urgency_score: float,
    negativity_score: float,
    visual_conf: float,
    priority_score: float,
    top_category: str,
    sentiment_label: str,
    entropy: float,
    keywords: list[str],
    defect_label: str,
    has_image: bool,
    image_bytes: Optional[bytes] = None,
) -> dict[str, Any]:
    """Create a new ticket dictionary."""
    ticket_id = f"CP-{uuid.uuid4().hex[:6].upper()}"
    return {
        "ticket_id":       ticket_id,
        "student_name":    student_name,
        "student_id":      student_id,
        "department":      department,
        "category":        category,
        "description":     description,
        "status":          "Open",
        "created_at":      datetime.now().isoformat(),
        "updated_at":      datetime.now().isoformat(),
        # NLP outputs
        "urgency_score":   round(urgency_score, 4),
        "negativity_score": round(negativity_score, 4),
        "sentiment_label": sentiment_label,
        "entropy":         round(entropy, 4),
        "keywords":        keywords,
        "top_category":    top_category,
        # Vision outputs
        "visual_conf":     round(visual_conf, 4),
        "defect_label":    defect_label,
        "has_image":       has_image,
        "image_bytes":     image_bytes,  # bytes or None
        # Fuzzy outputs
        "priority_score":  round(priority_score, 2),
        "severity":        get_severity_label(priority_score),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def _open_db() -> Any:
    """Return an open SqliteDict (or memory dict if unavailable)."""
    if _sqlitedict_available:
        return SqliteDict(str(DB_PATH), tablename="tickets", autocommit=True)
    return _memory_store


def save_ticket(ticket: dict[str, Any]) -> str:
    """Persist a ticket and return its ticket_id."""
    try:
        db = _open_db()
        # Don't store raw image bytes in DB for demo — keep reference flag
        storable = {k: v for k, v in ticket.items() if k != "image_bytes"}
        db[ticket["ticket_id"]] = storable
        if _sqlitedict_available:
            db.close()
        logger.info(f"Ticket saved: {ticket['ticket_id']}")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to save ticket: {exc}")
    return ticket["ticket_id"]


def load_all_tickets() -> list[dict[str, Any]]:
    """Load all tickets, sorted by priority_score descending."""
    try:
        db = _open_db()
        tickets = list(db.values())
        if _sqlitedict_available:
            db.close()
        tickets.sort(key=lambda t: t.get("priority_score", 0), reverse=True)
        return tickets
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to load tickets: {exc}")
        return []


def update_ticket_status(ticket_id: str, new_status: str) -> bool:
    """Update the status of a ticket by ID. Returns True on success."""
    try:
        db = _open_db()
        if ticket_id in db:
            t = dict(db[ticket_id])
            t["status"] = new_status
            t["updated_at"] = datetime.now().isoformat()
            db[ticket_id] = t
            if _sqlitedict_available:
                db.close()
            return True
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to update ticket {ticket_id}: {exc}")
        return False


def delete_ticket(ticket_id: str) -> bool:
    """Delete a ticket by ID. Returns True on success."""
    try:
        db = _open_db()
        if ticket_id in db:
            del db[ticket_id]
            if _sqlitedict_available:
                db.close()
            return True
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to delete ticket {ticket_id}: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Mock / seed data
# ─────────────────────────────────────────────────────────────────────────────

_MOCK_COMPLAINTS = [
    ("The washrooms on Floor 3 are absolutely filthy. No cleaning for days.", "Sanitation"),
    ("WiFi has been down in the computer lab for a week. Exams are approaching!", "Electrical / IT"),
    ("Broken ceiling fan in Room 204 is a safety hazard. Needs immediate repair.", "Facilities"),
    ("Professor never returns assignments. No feedback given.", "Academics"),
    ("Administration office never responds to scholarship queries.", "Administration"),
    ("Sewage smell in hostel corridor block B is unbearable.", "Sanitation"),
    ("Short circuit in lab 5 sparks when plugged in. Very dangerous.", "Electrical / IT"),
    ("Library has no AC and temperatures are extreme. Students can't study.", "Facilities"),
    ("Dean's office lost my leave application. No accountability.", "Administration"),
    ("Practical marks not updated despite submission 3 weeks ago.", "Academics"),
    ("Drinking water taps are broken in canteen. Health risk.", "Sanitation"),
    ("Projector in seminar hall doesn't work for 2 months.", "Electrical / IT"),
    ("Hostel stairs have broken railing. Someone will get hurt.", "Facilities"),
    ("Attendance portal shows wrong data. My percentage is 0%!", "Academics"),
    ("Bus timings were changed without notice. Students stranded.", "Administration"),
]

_STUDENT_NAMES = [
    "Arjun Sharma", "Priya Patel", "Rahul Singh", "Ananya Roy",
    "Vikram Nair", "Sneha Gupta", "Aditya Kumar", "Meena Krishnan",
    "Rohan Mehta", "Divya Menon", "Karthik Iyer", "Sanya Kapoor",
]

_DEPARTMENTS = [
    "Computer Science", "Mechanical Engineering", "Civil Engineering",
    "Electronics", "Information Technology", "Architecture", "MBA",
]


def seed_mock_data(n: int = 12) -> None:
    """Populate the database with realistic mock tickets for demo purposes."""
    existing = load_all_tickets()
    if len(existing) >= n:
        logger.info("Mock data already seeded — skipping.")
        return

    logger.info(f"Seeding {n} mock tickets …")
    rng = random.Random(42)

    for i in range(n):
        text, cat = _MOCK_COMPLAINTS[i % len(_MOCK_COMPLAINTS)]
        urgency    = rng.uniform(0.3, 1.0)
        negativity = rng.uniform(0.4, 1.0)
        visual     = rng.uniform(0.0, 0.9)
        priority   = (0.35 * urgency + 0.35 * negativity + 0.30 * visual) * 100

        # Vary created_at to simulate historical data
        created = (datetime.now() - timedelta(hours=rng.randint(1, 240))).isoformat()

        ticket = {
            "ticket_id":       f"CP-{uuid.uuid4().hex[:6].upper()}",
            "student_name":    rng.choice(_STUDENT_NAMES),
            "student_id":      f"STU{rng.randint(10000, 99999)}",
            "department":      rng.choice(_DEPARTMENTS),
            "category":        cat,
            "description":     text,
            "status":          rng.choice(STATUS_OPTIONS[:3]),  # Open/In Progress/Escalated
            "created_at":      created,
            "updated_at":      created,
            "urgency_score":   round(urgency, 4),
            "negativity_score": round(negativity, 4),
            "sentiment_label": "Negative" if negativity > 0.6 else "Mixed",
            "entropy":         round(rng.uniform(0.1, 0.8), 4),
            "keywords":        rng.sample(["urgent", "broken", "dirty", "dangerous", "ignored"], 3),
            "top_category":    cat,
            "visual_conf":     round(visual, 4),
            "defect_label":    "Broken / Damaged Infrastructure" if visual > 0.5 else "Normal / No Defect",
            "has_image":       visual > 0.3,
            "priority_score":  round(priority, 2),
            "severity":        get_severity_label(priority),
        }
        save_ticket(ticket)

    logger.success(f"Seeded {n} mock tickets.")
