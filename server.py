"""
CampusPulse AI — Flask Web Application Server
===============================================
Light-mode, high-responsiveness web application serving:
  - Student Grievance Portal (GET /)
  - Admin Triage Dashboard (GET /admin)
  - RESTful APIs for AI Triage pipeline execution and ticket management.

Replaces the legacy Streamlit entry point.
"""

from __future__ import annotations

import io
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

# ── Windows UTF-8 Console Encoding Fix ───────────────────────────────────────
# Ensures emoji/unicode in loguru output never crashes on Windows cp1252 stdout
if sys.platform == "win32":
    import codecs
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from flask import Flask, jsonify, render_template, request, send_from_directory
from loguru import logger
from PIL import Image
from werkzeug.utils import secure_filename

# ── Ensure Root Directory in Sys Path ─────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    ADMIN_PASSWORD,
    APP_TITLE,
    CAMPUS_CATEGORIES,
    CATEGORY_ICONS,
    SEVERITY_COLORS,
    STATUS_OPTIONS,
    get_severity_label,
)
from data.ticket_store import (
    delete_ticket,
    load_all_tickets,
    make_ticket,
    save_ticket,
    seed_mock_data,
    update_ticket_status,
)
from engines.fuzzy_engine import FuzzyEngine
from engines.nlp_engine import NLPEngine
from engines.vision_engine import VisionEngine, VisionResult

# ── Flask Application Setup ───────────────────────────────────────────────────
app = Flask(__name__, template_folder=str(ROOT_DIR / "templates"), static_folder=str(ROOT_DIR / "static"))
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "campuspulse-secret-key-2024")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.after_request
def add_cache_control_headers(response):
    """Prevent aggressive browser caching during development."""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.context_processor
def inject_cache_bust():
    """Inject dynamic asset version timestamp for cache busting."""
    import time
    return {"asset_version": int(time.time())}

UPLOAD_FOLDER = ROOT_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# ── Lazy Engine Singletons ────────────────────────────────────────────────────
_nlp_engine: Optional[NLPEngine] = None
_vision_engine: Optional[VisionEngine] = None
_fuzzy_engine: Optional[FuzzyEngine] = None


def get_nlp() -> NLPEngine:
    global _nlp_engine
    if _nlp_engine is None:
        logger.info("Initializing NLPEngine...")
        _nlp_engine = NLPEngine()
    return _nlp_engine


def get_vision() -> VisionEngine:
    global _vision_engine
    if _vision_engine is None:
        logger.info("Initializing VisionEngine...")
        _vision_engine = VisionEngine()
    return _vision_engine


def get_fuzzy() -> FuzzyEngine:
    global _fuzzy_engine
    if _fuzzy_engine is None:
        logger.info("Initializing FuzzyEngine...")
        _fuzzy_engine = FuzzyEngine()
    return _fuzzy_engine


def _is_allowed_image(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────────────────────────────────────
# Page Routes
# ─────────────────────────────────────────────────────────────────────────────


@app.route("/")
def student_portal():
    """Render Student Portal."""
    return render_template(
        "student.html",
        app_title=APP_TITLE,
        categories=CAMPUS_CATEGORIES,
        category_icons=CATEGORY_ICONS,
    )


@app.route("/admin")
def admin_dashboard():
    """Render Admin Dashboard."""
    return render_template(
        "admin.html",
        app_title=APP_TITLE,
        categories=CAMPUS_CATEGORIES,
        category_icons=CATEGORY_ICONS,
        statuses=STATUS_OPTIONS,
        severity_colors=SEVERITY_COLORS,
    )


# ─────────────────────────────────────────────────────────────────────────────
# REST API Endpoints
# ─────────────────────────────────────────────────────────────────────────────


@app.route("/api/analyze-text", methods=["POST"])
def analyze_text():
    """Real-time instant text analysis for category suggestion and intent feedback."""
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "").strip()
        if not text or len(text) < 3:
            return jsonify({
                "success": False,
                "suggested_category": None,
                "category_probs": {},
                "urgency_score": 0.0,
                "negativity_score": 0.0,
                "sentiment_label": "Neutral",
                "entropy": 0.0,
                "keywords": [],
            })

        nlp = get_nlp()
        result = nlp.analyse(text)

        return jsonify({
            "success": True,
            "suggested_category": result.top_category,
            "category_probs": result.category_probs,
            "urgency_score": round(result.urgency_score, 4),
            "negativity_score": round(result.negativity_score, 4),
            "sentiment_label": result.sentiment_label,
            "entropy": round(result.entropy, 4),
            "keywords": result.keywords[:8],
        })
    except Exception as exc:
        logger.error(f"Error in /api/analyze-text: {exc}", exc_info=True)
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/submit", methods=["POST"])
def submit_ticket():
    """
    Accept student text + image upload, execute full ML pipeline:
    NLP -> Vision -> Fuzzy -> Save ticket -> Return JSON.
    """
    try:
        # Check if form data or JSON
        if request.is_json:
            form = request.get_json() or {}
            uploaded_file = None
        else:
            form = request.form
            uploaded_file = request.files.get("image")

        student_name = (form.get("student_name") or "").strip()
        student_id = (form.get("student_id") or "").strip()
        department = (form.get("department") or "Other").strip()
        category_override = (form.get("category") or "").strip()
        description = (form.get("description") or "").strip()

        # Validation
        if not student_name:
            return jsonify({"success": False, "error": "Full Name is required."}), 400
        if not student_id:
            return jsonify({"success": False, "error": "Student ID is required."}), 400
        if not description or len(description) < 20:
            return jsonify({
                "success": False,
                "error": "Please describe the issue in at least 20 characters.",
            }), 400

        # 1. NLP Engine Execution
        nlp = get_nlp()
        nlp_result = nlp.analyse(description)

        chosen_category = category_override if category_override in CAMPUS_CATEGORIES else nlp_result.top_category

        # 2. Vision Engine Execution
        vision = get_vision()
        image_bytes: Optional[bytes] = None
        image_filename: Optional[str] = None
        image_url: Optional[str] = None

        if uploaded_file and uploaded_file.filename and _is_allowed_image(uploaded_file.filename):
            try:
                image_bytes = uploaded_file.read()
                # Re-seek to start: PIL needs to read from byte 0
                pil_img = Image.open(io.BytesIO(image_bytes))
                pil_img.verify()          # raises if corrupt / not a real image
                pil_img = Image.open(io.BytesIO(image_bytes))  # re-open after verify()

                # Save file to uploads folder
                ext = secure_filename(uploaded_file.filename).rsplit(".", 1)[-1].lower()
                safe_name = f"proof_{uuid.uuid4().hex[:10]}.{ext}"
                save_path = UPLOAD_FOLDER / safe_name
                with open(save_path, "wb") as f:
                    f.write(image_bytes)

                image_filename = safe_name
                image_url = f"/static/uploads/{safe_name}"

                vis_result = vision.analyse(pil_img)
            except Exception as vis_err:
                logger.error(f"Image processing error: {vis_err}")
                vis_result = VisionResult(confidence=0.0, defect_label="Image Error")
        else:
            vis_result = VisionResult(confidence=0.0, defect_label="No Image Provided")

        # 3. Fuzzy Severity Triage Engine Execution
        fuzzy = get_fuzzy()
        fuzzy_result = fuzzy.compute(
            text_urgency=nlp_result.urgency_score,
            negativity=nlp_result.negativity_score,
            visual_conf=vis_result.confidence,
        )

        # 4. Construct and Save Ticket
        ticket = make_ticket(
            student_name=student_name,
            student_id=student_id,
            department=department,
            category=chosen_category,
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
            has_image=image_bytes is not None,
            image_bytes=image_bytes,
        )

        if image_url:
            ticket["image_url"] = image_url
            ticket["image_filename"] = image_filename

        ticket_id = save_ticket(ticket)

        # Build clean JSON serializable response
        response_ticket = {k: v for k, v in ticket.items() if k != "image_bytes"}
        response_ticket["ticket_id"] = ticket_id

        return jsonify({
            "success": True,
            "ticket_id": ticket_id,
            "ticket": response_ticket,
            "priority_score": round(fuzzy_result.priority_score, 2),
            "severity": fuzzy_result.severity,
            "suggested_category": nlp_result.top_category,
            "message": "Grievance triaged and registered successfully.",
        })

    except Exception as exc:
        logger.error(f"Error submitting ticket: {exc}", exc_info=True)
        return jsonify({"success": False, "error": f"Internal triage error: {str(exc)}"}), 500


@app.route("/api/tickets", methods=["GET"])
def get_tickets():
    """Fetch all tickets from the database as JSON, sorted by priority."""
    try:
        tickets = load_all_tickets()
        # Ensure tickets have valid serializable fields
        cleaned = []
        for t in tickets:
            item = dict(t)
            item.pop("image_bytes", None)
            item.pop("_nlp", None)
            item.pop("_vis", None)
            item.pop("_fuzzy", None)
            cleaned.append(item)

        return jsonify({
            "success": True,
            "count": len(cleaned),
            "tickets": cleaned,
        })
    except Exception as exc:
        logger.error(f"Error fetching tickets: {exc}")
        return jsonify({"success": False, "error": str(exc), "tickets": []}), 500


@app.route("/api/tickets/update", methods=["POST"])
def update_ticket():
    """Update ticket status / dispatch technician."""
    try:
        data = request.get_json(silent=True) or request.form
        ticket_id = (data.get("ticket_id") or "").strip()
        new_status = (data.get("status") or "").strip()
        notes = (data.get("notes") or "").strip()

        if not ticket_id:
            return jsonify({"success": False, "error": "ticket_id is required."}), 400
        if not new_status or new_status not in STATUS_OPTIONS:
            return jsonify({
                "success": False,
                "error": f"Invalid status. Must be one of: {', '.join(STATUS_OPTIONS)}",
            }), 400

        success = update_ticket_status(ticket_id, new_status)
        if not success:
            return jsonify({"success": False, "error": f"Ticket {ticket_id} not found."}), 404

        logger.info(f"Updated ticket {ticket_id} status to '{new_status}' (notes: {notes})")
        return jsonify({
            "success": True,
            "ticket_id": ticket_id,
            "status": new_status,
            "notes": notes,
            "message": f"Ticket {ticket_id} status updated to {new_status}.",
        })
    except Exception as exc:
        logger.error(f"Error updating ticket: {exc}")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/tickets/seed", methods=["POST"])
def seed_tickets():
    """Helper to populate demo tickets if database has few."""
    try:
        seed_mock_data(n=12)
        tickets = load_all_tickets()
        return jsonify({
            "success": True,
            "message": "Seeded mock data successfully.",
            "count": len(tickets),
        })
    except Exception as exc:
        logger.error(f"Error seeding tickets: {exc}")
        return jsonify({"success": False, "error": str(exc)}), 500


# ── Server Startup ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Seed mock tickets if store is empty on launch
    try:
        seed_mock_data(n=12)
    except Exception as err:
        logger.warning(f"Could not auto-seed tickets: {err}")

    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info(f"Starting {APP_TITLE} on http://127.0.0.1:{port} (and http://localhost:{port})")
    app.run(host=host, port=port, debug=False)
