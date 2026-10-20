"""
CampusPulse AI — Fuzzy Severity Triage Engine
==============================================
Implements a Mamdani Fuzzy Inference System (FIS) using scikit-fuzzy to
convert three crisp input signals into a unified Priority Score [0–100].

Crisp Inputs
------------
  • text_urgency   — urgency score from NLP engine  [0, 1]
  • negativity     — sentiment negativity score       [0, 1]
  • visual_conf    — visual defect confidence         [0, 1]

Fuzzy Sets (per input)
----------------------
  Low | Medium | High  (triangular / trapezoidal membership functions)

Output
------
  • priority_score — unified priority [0, 100]
    mapped to severity: Low / Medium / High / Critical

Defuzzification
---------------
  Centroid (Centre of Gravity) method.

Public API
----------
    from engines.fuzzy_engine import FuzzyEngine
    engine = FuzzyEngine()
    result = engine.compute(text_urgency=0.8, negativity=0.7, visual_conf=0.9)
    # FuzzyResult(priority_score=82.3, severity="Critical", ...)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_severity_label

# ── Optional scikit-fuzzy import ──────────────────────────────────────────────
_skfuzzy_available = False
try:
    import skfuzzy as fuzz  # type: ignore
    import skfuzzy.control as ctrl  # type: ignore

    _skfuzzy_available = True
except ImportError:
    logger.warning("scikit-fuzzy not installed — FuzzyEngine using analytic fallback.")


# ─────────────────────────────────────────────────────────────────────────────
# Data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FuzzyResult:
    """Output of the Fuzzy Severity Triage Engine."""

    # Defuzzified priority score on [0, 100]
    priority_score: float = 0.0

    # Human-readable severity level
    severity: str = "Low"

    # Per-rule firing strengths for transparency
    rule_activations: dict[str, float] | None = None

    # Membership degrees of the crisp inputs in each fuzzy set
    urgency_membership: dict[str, float] | None = None
    negativity_membership: dict[str, float] | None = None
    visual_membership: dict[str, float] | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Fuzzy Engine
# ─────────────────────────────────────────────────────────────────────────────

class FuzzyEngine:
    """
    Mamdani Fuzzy Inference System for grievance severity triage.

    The system is built once at construction time and kept alive
    for repeated inference calls.
    """

    def __init__(self) -> None:
        self._sim: Optional[object] = None  # ctrl.ControlSystemSimulation

        if _skfuzzy_available:
            self._build_fis()
        else:
            logger.info("FuzzyEngine using analytic (non-fuzzy) fallback.")

    # ── Build FIS ─────────────────────────────────────────────────────────────

    def _build_fis(self) -> None:
        """Construct the Mamdani FIS with membership functions and rules."""
        try:
            # ── Universe of discourse ─────────────────────────────────────────
            u_input = np.linspace(0, 1, 101)       # Input universe [0, 1]
            u_output = np.linspace(0, 100, 1001)   # Output universe [0, 100]

            # ── Antecedent variables ──────────────────────────────────────────
            urgency = ctrl.Antecedent(u_input, "urgency")
            negativity = ctrl.Antecedent(u_input, "negativity")
            visual = ctrl.Antecedent(u_input, "visual")

            # ── Consequent variable ───────────────────────────────────────────
            priority = ctrl.Consequent(u_output, "priority", defuzzify_method="centroid")

            # ── Membership functions — Inputs ─────────────────────────────────
            # Low  (trapezoidal)
            # Medium (triangular)
            # High (trapezoidal)

            for var in (urgency, negativity, visual):
                var["low"]    = fuzz.trapmf(var.universe, [0.0,  0.0,  0.2,  0.4 ])
                var["medium"] = fuzz.trimf( var.universe, [0.3,  0.5,  0.7        ])
                var["high"]   = fuzz.trapmf(var.universe, [0.6,  0.8,  1.0,  1.0 ])

            # ── Membership functions — Output ─────────────────────────────────
            priority["low"]      = fuzz.trapmf(u_output, [0,   0,   20,  40 ])
            priority["medium"]   = fuzz.trimf( u_output, [30,  50,  70       ])
            priority["high"]     = fuzz.trimf( u_output, [60,  75,  90       ])
            priority["critical"] = fuzz.trapmf(u_output, [80,  90,  100, 100])

            # ── Rule base ─────────────────────────────────────────────────────
            rules = [
                # --- Critical rules ---
                ctrl.Rule(
                    urgency["high"] & negativity["high"] & visual["high"],
                    priority["critical"],
                    label="R1_all_high",
                ),
                ctrl.Rule(
                    urgency["high"] & negativity["high"] & visual["medium"],
                    priority["critical"],
                    label="R2_urg_neg_high",
                ),
                # --- High rules ---
                ctrl.Rule(
                    urgency["high"] & negativity["medium"] & visual["high"],
                    priority["high"],
                    label="R3_urg_vis_high",
                ),
                ctrl.Rule(
                    urgency["high"] & negativity["low"] & visual["high"],
                    priority["high"],
                    label="R4_urg_vis_high_lneg",
                ),
                ctrl.Rule(
                    urgency["medium"] & negativity["high"] & visual["high"],
                    priority["high"],
                    label="R5_neg_vis_high",
                ),
                ctrl.Rule(
                    urgency["high"] & negativity["high"] & visual["low"],
                    priority["high"],
                    label="R6_text_high_novis",
                ),
                # --- Medium rules ---
                ctrl.Rule(
                    urgency["medium"] & negativity["medium"] & visual["medium"],
                    priority["medium"],
                    label="R7_all_medium",
                ),
                ctrl.Rule(
                    urgency["high"] & negativity["low"] & visual["low"],
                    priority["medium"],
                    label="R8_urg_only",
                ),
                ctrl.Rule(
                    urgency["low"] & negativity["high"] & visual["medium"],
                    priority["medium"],
                    label="R9_neg_vis",
                ),
                ctrl.Rule(
                    urgency["medium"] & negativity["low"] & visual["high"],
                    priority["medium"],
                    label="R10_vis_high",
                ),
                # --- Low rules ---
                ctrl.Rule(
                    urgency["low"] & negativity["low"] & visual["low"],
                    priority["low"],
                    label="R11_all_low",
                ),
                ctrl.Rule(
                    urgency["low"] & negativity["medium"] & visual["low"],
                    priority["low"],
                    label="R12_med_neg_only",
                ),
                ctrl.Rule(
                    urgency["low"] & negativity["low"] & visual["medium"],
                    priority["low"],
                    label="R13_vis_med",
                ),
            ]

            system = ctrl.ControlSystem(rules)
            self._sim = ctrl.ControlSystemSimulation(system)
            logger.success("FuzzyEngine FIS built successfully.")

        except Exception as exc:  # noqa: BLE001
            logger.error(f"FuzzyEngine build failed: {exc}")
            self._sim = None

    # ── Public API ────────────────────────────────────────────────────────────

    def compute(
        self,
        text_urgency: float,
        negativity: float,
        visual_conf: float,
    ) -> FuzzyResult:
        """
        Run the Mamdani FIS and return a :class:`FuzzyResult`.

        Parameters
        ----------
        text_urgency : float
            Urgency score from the NLP engine [0, 1].
        negativity : float
            Sentiment negativity score [0, 1].
        visual_conf : float
            Visual defect confidence from the Vision engine [0, 1].

        Returns
        -------
        FuzzyResult
        """
        # Clamp inputs to valid range
        u = float(np.clip(text_urgency, 0.0, 1.0))
        n = float(np.clip(negativity,   0.0, 1.0))
        v = float(np.clip(visual_conf,  0.0, 1.0))

        if self._sim is not None:
            score = self._fuzzy_compute(u, n, v)
        else:
            score = self._analytic_compute(u, n, v)

        severity = get_severity_label(score)

        # Compute membership degrees for transparency display
        u_mem = self._membership_degrees(u)
        n_mem = self._membership_degrees(n)
        v_mem = self._membership_degrees(v)

        return FuzzyResult(
            priority_score=round(score, 2),
            severity=severity,
            urgency_membership=u_mem,
            negativity_membership=n_mem,
            visual_membership=v_mem,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fuzzy_compute(self, u: float, n: float, v: float) -> float:
        """Run the scikit-fuzzy simulation."""
        try:
            self._sim.input["urgency"]    = u
            self._sim.input["negativity"] = n
            self._sim.input["visual"]     = v
            self._sim.compute()
            return float(self._sim.output["priority"])
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Fuzzy compute error: {exc} — using analytic fallback.")
            return self._analytic_compute(u, n, v)

    def _analytic_compute(self, u: float, n: float, v: float) -> float:
        """
        Weighted average fallback when scikit-fuzzy is unavailable.
        Weights: urgency 35%, negativity 35%, visual 30%.
        Scaled to [0, 100].
        """
        score = (0.35 * u + 0.35 * n + 0.30 * v) * 100.0
        return round(float(np.clip(score, 0.0, 100.0)), 2)

    def _membership_degrees(self, x: float) -> dict[str, float]:
        """Compute triangular/trapezoidal membership degrees for a scalar *x* in [0,1]."""
        u = np.linspace(0, 1, 101)
        if _skfuzzy_available:
            low    = float(fuzz.interp_membership(u, fuzz.trapmf(u, [0.0, 0.0, 0.2, 0.4]), x))
            medium = float(fuzz.interp_membership(u, fuzz.trimf( u, [0.3, 0.5, 0.7]),       x))
            high   = float(fuzz.interp_membership(u, fuzz.trapmf(u, [0.6, 0.8, 1.0, 1.0]), x))
        else:
            # Simple linear fallback
            low    = max(0.0, min(1.0, (0.4 - x) / 0.4))
            medium = max(0.0, min((x - 0.3) / 0.2, (0.7 - x) / 0.2, 1.0))
            high   = max(0.0, min(1.0, (x - 0.6) / 0.4))

        return {
            "low":    round(low,    4),
            "medium": round(medium, 4),
            "high":   round(high,   4),
        }

    # ── Membership surface data (for visualisation) ───────────────────────────

    def get_membership_surface(self) -> dict[str, np.ndarray]:
        """
        Return numpy arrays describing the membership functions for plotting.
        Shape: {set_name: (x_array, mf_array)}
        """
        u = np.linspace(0, 1, 101)
        surfaces: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        if _skfuzzy_available:
            surfaces["Input: Low"]    = (u, fuzz.trapmf(u, [0.0, 0.0, 0.2, 0.4]))
            surfaces["Input: Medium"] = (u, fuzz.trimf( u, [0.3, 0.5, 0.7]))
            surfaces["Input: High"]   = (u, fuzz.trapmf(u, [0.6, 0.8, 1.0, 1.0]))

        uo = np.linspace(0, 100, 1001)
        if _skfuzzy_available:
            surfaces["Output: Low"]      = (uo, fuzz.trapmf(uo, [0,   0,   20,  40]))
            surfaces["Output: Medium"]   = (uo, fuzz.trimf( uo, [30,  50,  70]))
            surfaces["Output: High"]     = (uo, fuzz.trimf( uo, [60,  75,  90]))
            surfaces["Output: Critical"] = (uo, fuzz.trapmf(uo, [80,  90,  100, 100]))

        return surfaces
