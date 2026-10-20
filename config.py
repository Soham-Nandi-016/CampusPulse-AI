"""
CampusPulse AI — Application Configuration
=========================================
Centralised settings loaded from environment variables with
sensible defaults so the app runs in mock/demo mode out of the box.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env if present ─────────────────────────────────────────────────────
load_dotenv()

# ── Project Root ──────────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parent

# ── Model Paths ───────────────────────────────────────────────────────────────
NLP_MODEL_PATH: Path = ROOT_DIR / os.getenv(
    "NLP_MODEL_PATH", "models/nlp/distilbert_absa"
)
VISION_MODEL_PATH: Path = ROOT_DIR / os.getenv(
    "VISION_MODEL_PATH", "models/vision/mobilenetv2_defect.pt"
)

# ── Feature Flags ─────────────────────────────────────────────────────────────
USE_MOCK_MODELS: bool = os.getenv("USE_MOCK_MODELS", "true").lower() == "true"
ENABLE_GPU: bool = os.getenv("ENABLE_GPU", "false").lower() == "true"

# ── App Settings ──────────────────────────────────────────────────────────────
APP_TITLE: str = os.getenv("APP_TITLE", "CampusPulse AI")
ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "admin123")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ── Domain Constants ──────────────────────────────────────────────────────────
CAMPUS_CATEGORIES: list[str] = [
    "Academics",
    "Facilities",
    "Electrical / IT",
    "Sanitation",
    "Administration",
]

CATEGORY_ICONS: dict[str, str] = {
    "Academics": "📚",
    "Facilities": "🏛️",
    "Electrical / IT": "⚡",
    "Sanitation": "🧹",
    "Administration": "🏫",
}

SEVERITY_COLORS: dict[str, str] = {
    "Critical": "#EF4444",
    "High": "#F97316",
    "Medium": "#EAB308",
    "Low": "#22C55E",
}

STATUS_OPTIONS: list[str] = [
    "Open",
    "In Progress",
    "Escalated",
    "Resolved",
    "Closed",
]

# ── Fuzzy Engine Thresholds ───────────────────────────────────────────────────
PRIORITY_THRESHOLDS: dict[str, tuple[float, float]] = {
    "Low": (0.0, 35.0),
    "Medium": (35.0, 65.0),
    "High": (65.0, 85.0),
    "Critical": (85.0, 100.0),
}


def get_severity_label(priority_score: float) -> str:
    """Map a 0–100 fuzzy priority score to a human-readable severity label."""
    for label, (lo, hi) in PRIORITY_THRESHOLDS.items():
        if lo <= priority_score <= hi:
            return label
    return "Low"
