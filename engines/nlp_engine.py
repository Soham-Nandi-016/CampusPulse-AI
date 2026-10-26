"""
CampusPulse AI — Text Analytics / NLP Engine
==============================================
Performs Aspect-Based Sentiment Analysis (ABSA) on student complaint text
using a fine-tuned DistilBERT transformer.

Falls back to a deterministic mock pipeline when model weights are absent
so the web app runs in full demo mode without a GPU or pre-trained files.

Public API
----------
    from engines.nlp_engine import NLPEngine
    engine = NLPEngine()
    result = engine.analyse("The labs are broken and no one cares")
    # NLPResult(
    #     category_probs={...},
    #     top_category="Electrical / IT",
    #     urgency_score=0.78,
    #     negativity_score=0.82,
    #     entropy=0.43,
    #     sentiment_label="Negative",
    #     keywords=[...]
    # )
"""

from __future__ import annotations

import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    CAMPUS_CATEGORIES,
    ENABLE_GPU,
    NLP_MODEL_PATH,
    USE_MOCK_MODELS,
)

# ── Optional heavy imports (deferred to avoid startup cost) ───────────────────
_transformers_available = False
try:
    import torch

    # PyTorch 2.6 CPU compatibility guard:
    # torch.accelerator.current_accelerator() raises RuntimeError on Windows CPU-only machines
    if hasattr(torch, "accelerator"):
        _orig_current_accelerator = torch.accelerator.current_accelerator

        def _safe_current_accelerator():
            try:
                if not torch.accelerator.is_available():
                    return None
                return _orig_current_accelerator()
            except RuntimeError:
                return None

        torch.accelerator.current_accelerator = _safe_current_accelerator

    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification  # type: ignore

    _transformers_available = True
except Exception as exc:
    logger.warning(f"Transformers / PyTorch import failed ({exc}) — running in mock mode.")


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NLPResult:
    """Output of the NLP Engine for a single complaint text."""

    # Per-category probability distribution (sums to ~1.0)
    category_probs: dict[str, float] = field(default_factory=dict)

    # Predicted primary category
    top_category: str = "Facilities"

    # Urgency score derived from temporal / imperative language [0, 1]
    urgency_score: float = 0.5

    # Negativity score from sentiment analysis [0, 1]
    negativity_score: float = 0.5

    # Shannon Predictive Entropy of the category distribution [0, 1] (normalised)
    # High entropy → vague/ambiguous complaint
    entropy: float = 0.5

    # Human-readable sentiment label
    sentiment_label: str = "Neutral"

    # Extracted urgency / negative keywords for display
    keywords: list[str] = field(default_factory=list)

    # Raw inference latency in milliseconds
    inference_ms: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Keyword lexicons for heuristic scoring (used in both real & mock pipelines)
# ─────────────────────────────────────────────────────────────────────────────

_URGENCY_WORDS: list[str] = [
    "urgent", "immediately", "asap", "emergency", "broken", "damaged",
    "flooding", "fire", "dangerous", "unsafe", "critical", "severe",
    "not working", "failure", "collapse", "leaking", "black-out", "outage",
    "serious", "dire", "desperate",
]

_NEGATIVE_WORDS: list[str] = [
    "dirty", "filthy", "horrible", "terrible", "disgusting", "poor",
    "worst", "bad", "awful", "unacceptable", "negligent", "ignored",
    "useless", "broken", "failed", "never", "no one", "corrupt",
    "waste", "pathetic",
]

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Academics": [
        "lecture", "professor", "assignment", "exam", "curriculum",
        "syllabus", "grades", "faculty", "teacher", "course", "timetable",
        "marks", "result", "class", "attendance", "lab", "practical",
    ],
    "Facilities": [
        "chair", "desk", "classroom", "library", "bench", "building",
        "roof", "wall", "window", "door", "stairs", "lift", "elevator",
        "furniture", "room", "hall", "auditorium", "sports", "canteen",
    ],
    "Electrical / IT": [
        "electricity", "power", "fan", "light", "ac", "wifi", "internet",
        "projector", "computer", "network", "socket", "wiring", "cable",
        "server", "blackout", "outage", "short circuit", "UPS",
    ],
    "Sanitation": [
        "toilet", "washroom", "bathroom", "garbage", "trash", "smell",
        "stink", "water", "drainage", "sewage", "dirty", "hygiene",
        "cockroach", "pest", "rat", "clean", "sweep", "mop", "dust",
    ],
    "Administration": [
        "office", "staff", "principal", "dean", "registrar", "fee",
        "scholarship", "document", "certificate", "id card", "hostel",
        "warden", "management", "policy", "rule", "complaint", "notice",
    ],
}


def _extract_keywords(text: str) -> list[str]:
    """Return urgency/negative keywords found in *text* (whole-word match only)."""
    import re
    lower = text.lower()
    found = []
    for w in _URGENCY_WORDS + _NEGATIVE_WORDS:
        # Use word-boundary search to avoid false matches (e.g. "lab" in "elaborate")
        pattern = r'\b' + re.escape(w) + r'\b'
        if re.search(pattern, lower) and w not in found:
            found.append(w)
    return found[:8]


def _heuristic_urgency(text: str) -> float:
    """Compute a [0, 1] urgency score from keyword density (whole-word match)."""
    import re
    lower = text.lower()
    hits = sum(1 for w in _URGENCY_WORDS if re.search(r'\b' + re.escape(w) + r'\b', lower))
    return min(hits / max(len(_URGENCY_WORDS) * 0.25, 1), 1.0)


def _heuristic_negativity(text: str) -> float:
    """Compute a [0, 1] negativity score from negative keyword density (whole-word match)."""
    import re
    lower = text.lower()
    hits = sum(1 for w in _NEGATIVE_WORDS if re.search(r'\b' + re.escape(w) + r'\b', lower))
    # Normalise against a reasonable saturation point
    return min(hits / max(len(_NEGATIVE_WORDS) * 0.20, 1), 1.0)


def _heuristic_category_probs(text: str) -> dict[str, float]:
    """Assign category probabilities from keyword overlap (whole-word matches)."""
    import re
    lower = text.lower()
    scores: dict[str, float] = {}
    for cat, kws in _CATEGORY_KEYWORDS.items():
        scores[cat] = sum(
            1 for kw in kws
            if re.search(r'\b' + re.escape(kw) + r'\b', lower)
        )

    total = sum(scores.values()) or 1.0
    probs = {cat: round(s / total, 4) for cat, s in scores.items()}

    # Ensure minimum floor probability so softmax-like distribution exists
    floor = 0.03
    for cat in probs:
        probs[cat] = max(probs[cat], floor)

    # Re-normalise
    total2 = sum(probs.values())
    probs = {cat: round(v / total2, 4) for cat, v in probs.items()}
    return probs


def _shannon_entropy(probs: dict[str, float]) -> float:
    """Return Shannon entropy of the distribution, normalised to [0, 1]."""
    n = len(probs)
    if n <= 1:
        return 0.0
    raw = -sum(p * math.log2(p + 1e-9) for p in probs.values())
    max_entropy = math.log2(n)
    return round(raw / max_entropy, 4)


# ─────────────────────────────────────────────────────────────────────────────
# NLP Engine
# ─────────────────────────────────────────────────────────────────────────────

class NLPEngine:
    """
    Façade for the text analytics pipeline.

    In *real* mode (USE_MOCK_MODELS=False, model weights present):
        Uses a fine-tuned DistilBERT checkpoint for sequence classification.

    In *mock* mode (default):
        Uses deterministic heuristics so the UI is fully functional without
        downloading any model weights.
    """

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._sentiment_pipe = None
        self._mode: str = "mock"

        if not USE_MOCK_MODELS and _transformers_available:
            self._load_model()
        else:
            logger.info("NLPEngine starting in MOCK mode.")

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Attempt to load fine-tuned DistilBERT from disk."""
        try:
            device = 0 if (ENABLE_GPU and _transformers_available and torch.cuda.is_available()) else -1

            if NLP_MODEL_PATH.exists():
                logger.info(f"Loading NLP model from {NLP_MODEL_PATH} …")
                self._tokenizer = AutoTokenizer.from_pretrained(str(NLP_MODEL_PATH))
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    str(NLP_MODEL_PATH)
                )
                self._model.eval()
                self._mode = "real"
                logger.success("NLPEngine loaded in REAL mode.")
            else:
                # Fall back: use zero-shot pipeline from HF hub
                logger.warning(
                    f"Model weights not found at {NLP_MODEL_PATH}. "
                    "Attempting zero-shot fallback from HuggingFace Hub …"
                )
                self._sentiment_pipe = pipeline(
                    "zero-shot-classification",
                    model="typeform/distilbert-base-uncased-mnli",
                    device=device,
                )
                self._mode = "zero_shot"
                logger.success("NLPEngine loaded in ZERO-SHOT mode.")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"NLPEngine model load failed: {exc} — falling back to mock.")
            self._mode = "mock"

    # ── Public API ────────────────────────────────────────────────────────────

    def analyse(self, text: str) -> NLPResult:
        """
        Run the full NLP pipeline on *text* and return an :class:`NLPResult`.

        Parameters
        ----------
        text : str
            Raw student complaint text.

        Returns
        -------
        NLPResult
        """
        if not text or not text.strip():
            return NLPResult()

        t0 = time.perf_counter()

        if self._mode == "real":
            result = self._analyse_real(text)
        elif self._mode == "zero_shot":
            result = self._analyse_zero_shot(text)
        else:
            result = self._analyse_mock(text)

        result.inference_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result

    # ── Real inference ────────────────────────────────────────────────────────

    def _analyse_real(self, text: str) -> NLPResult:
        """Run fine-tuned DistilBERT inference."""
        import torch

        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs_raw = torch.softmax(logits, dim=-1).squeeze()
            # Guard: if batch dim was squeezed away for 1-class output, ensure list
            probs_tensor_list = probs_raw.tolist()
            if isinstance(probs_tensor_list, float):
                probs_tensor_list = [probs_tensor_list]

        label_map: dict[int, str] = {
            i: cat for i, cat in enumerate(CAMPUS_CATEGORIES)
        }
        if hasattr(self._model, "config") and hasattr(self._model.config, "id2label") and self._model.config.id2label:
            label_map = {int(k): v for k, v in self._model.config.id2label.items()}

        cat_probs = {label_map.get(i, CAMPUS_CATEGORIES[i % len(CAMPUS_CATEGORIES)]): round(float(p), 4) for i, p in enumerate(probs_tensor_list)}

        return self._build_result(text, cat_probs)

    # ── Zero-shot inference ───────────────────────────────────────────────────

    def _analyse_zero_shot(self, text: str) -> NLPResult:
        """Use HuggingFace zero-shot classification pipeline."""
        out = self._sentiment_pipe(text, candidate_labels=CAMPUS_CATEGORIES)
        cat_probs = dict(zip(out["labels"], out["scores"]))
        return self._build_result(text, cat_probs)

    # ── Mock inference ────────────────────────────────────────────────────────

    def _analyse_mock(self, text: str) -> NLPResult:
        """Pure heuristic pipeline — no neural network required."""
        cat_probs = _heuristic_category_probs(text)
        return self._build_result(text, cat_probs)

    # ── Shared result builder ─────────────────────────────────────────────────

    def _build_result(self, text: str, cat_probs: dict[str, float]) -> NLPResult:
        top_cat = max(cat_probs, key=lambda k: cat_probs[k])
        urgency = _heuristic_urgency(text)
        negativity = _heuristic_negativity(text)
        entropy = _shannon_entropy(cat_probs)
        keywords = _extract_keywords(text)

        if negativity >= 0.6:
            sentiment_label = "Negative"
        elif negativity >= 0.3:
            sentiment_label = "Mixed"
        else:
            sentiment_label = "Neutral"

        return NLPResult(
            category_probs=cat_probs,
            top_category=top_cat,
            urgency_score=urgency,
            negativity_score=negativity,
            entropy=entropy,
            sentiment_label=sentiment_label,
            keywords=keywords,
        )
