"""
CampusPulse AI — Visual Proof Verification Engine
==================================================
Processes uploaded images using MobileNetV2 (Transfer Learning) to produce
a Visual Defect Verification Confidence Score [0.0 – 1.0] confirming that
the image contains physical evidence of a campus defect.

Modes
-----
*real*      — loads fine-tuned MobileNetV2 weights from disk
*pretrained*— uses ImageNet-pretrained MobileNetV2 with heuristic mapping
*mock*      — deterministic scoring from image statistics (no model required)

Public API
----------
    from engines.vision_engine import VisionEngine
    engine = VisionEngine()
    result = engine.analyse(pil_image)
    # VisionResult(confidence=0.84, defect_label="Structural Damage", ...)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from PIL import Image

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ENABLE_GPU, USE_MOCK_MODELS, VISION_MODEL_PATH

# ── Optional heavy imports ────────────────────────────────────────────────────
_torch_available = False
_torchvision_available = False
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms  # type: ignore

    _torch_available = True
    _torchvision_available = True
except ImportError:
    logger.warning("PyTorch / torchvision not installed — VisionEngine running in mock mode.")


# ─────────────────────────────────────────────────────────────────────────────
# Defect taxonomy — binary: defect present / not present
# ─────────────────────────────────────────────────────────────────────────────

DEFECT_LABELS: list[str] = [
    "Broken / Damaged Infrastructure",
    "Water Leakage / Flooding",
    "Dirty / Unsanitary Conditions",
    "Electrical Hazard",
    "Structural Damage",
    "Normal / No Defect",
]

# ImageNet preprocessing transform
_IMAGENET_TRANSFORM = None
if _torchvision_available:
    _IMAGENET_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# Data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VisionResult:
    """Output of the Vision Engine for a single uploaded image."""

    # Primary confidence score [0, 1] — how likely the image shows a defect
    confidence: float = 0.0

    # Human-readable defect label
    defect_label: str = "Normal / No Defect"

    # Full per-class probability distribution
    class_probs: dict[str, float] = field(default_factory=dict)

    # Bounding heuristics extracted from the image
    mean_brightness: float = 0.0
    contrast_score: float = 0.0
    edge_density: float = 0.0

    # Whether the uploaded file appears to be a valid campus image
    is_valid_image: bool = True

    # Inference mode used
    mode: str = "mock"

    # Raw inference latency in milliseconds
    inference_ms: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Vision Engine
# ─────────────────────────────────────────────────────────────────────────────

class VisionEngine:
    """
    Façade for the visual proof verification pipeline.

    In *real* mode: runs fine-tuned MobileNetV2 binary/multi-class classifier.
    In *pretrained* mode: uses ImageNet MobileNetV2 + heuristic post-processing.
    In *mock* mode: derives confidence from image statistics (brightness/contrast/edges).
    """

    def __init__(self) -> None:
        self._model: Optional[object] = None
        self._mode: str = "mock"
        self._device: str = "cpu"

        if not USE_MOCK_MODELS and _torch_available:
            self._load_model()
        else:
            logger.info("VisionEngine starting in MOCK mode.")

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load MobileNetV2 — fine-tuned if weights exist, else pretrained ImageNet."""
        try:
            if ENABLE_GPU and torch.cuda.is_available():
                self._device = "cuda"

            if VISION_MODEL_PATH.exists():
                logger.info(f"Loading Vision model from {VISION_MODEL_PATH} …")
                state = torch.load(str(VISION_MODEL_PATH), map_location=self._device)
                model = models.mobilenet_v2(weights=None)
                if "classifier.4.weight" in state:
                    model.classifier = nn.Sequential(
                        nn.Dropout(0.3),
                        nn.Linear(model.last_channel, 256),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(256, len(DEFECT_LABELS)),
                    )
                else:
                    model.classifier[1] = nn.Linear(model.last_channel, len(DEFECT_LABELS))
                model.load_state_dict(state)
                model.eval()
                self._model = model.to(self._device)
                self._mode = "real"
                logger.success("VisionEngine loaded in REAL mode.")
            else:
                logger.warning(
                    f"Fine-tuned weights not found at {VISION_MODEL_PATH}. "
                    "Loading ImageNet-pretrained MobileNetV2 …"
                )
                model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
                model.eval()
                self._model = model.to(self._device)
                self._mode = "pretrained"
                logger.success("VisionEngine loaded in PRETRAINED mode.")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"VisionEngine load failed: {exc} — falling back to mock.")
            self._mode = "mock"

    # ── Public API ────────────────────────────────────────────────────────────

    def analyse(self, image: Image.Image) -> VisionResult:
        """
        Analyse *image* and return a :class:`VisionResult`.

        Parameters
        ----------
        image : PIL.Image.Image
            Uploaded image from the student portal.

        Returns
        -------
        VisionResult
        """
        t0 = time.perf_counter()

        # Always compute image statistics (used by all modes for richer output)
        stats = self._image_statistics(image)

        if self._mode == "real":
            result = self._analyse_real(image)
        elif self._mode == "pretrained":
            result = self._analyse_pretrained(image, stats)
        else:
            result = self._analyse_mock(stats)

        result.mean_brightness = stats["brightness"]
        result.contrast_score = stats["contrast"]
        result.edge_density = stats["edge_density"]
        result.mode = self._mode
        result.inference_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result

    # ── Image statistics ──────────────────────────────────────────────────────

    def _image_statistics(self, image: Image.Image) -> dict[str, float]:
        """Extract low-level image statistics for heuristic scoring."""
        img_rgb = image.convert("RGB")
        arr = np.array(img_rgb, dtype=np.float32)

        brightness = float(arr.mean() / 255.0)
        contrast = float(arr.std() / 128.0)

        # Simplified edge density via gradient magnitude on greyscale
        grey = np.array(img_rgb.convert("L"), dtype=np.float32)
        gx = np.abs(np.diff(grey, axis=1)).mean()
        gy = np.abs(np.diff(grey, axis=0)).mean()
        edge_density = float(min((gx + gy) / 128.0, 1.0))

        return {
            "brightness": round(brightness, 4),
            "contrast": round(contrast, 4),
            "edge_density": round(edge_density, 4),
        }

    # ── Real inference ────────────────────────────────────────────────────────

    def _analyse_real(self, image: Image.Image) -> VisionResult:
        """Run fine-tuned MobileNetV2 classifier."""
        tensor = _IMAGENET_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()

        class_probs = {label: round(float(p), 4) for label, p in zip(DEFECT_LABELS, probs)}
        defect_classes = [l for l in DEFECT_LABELS if l != "Normal / No Defect"]
        defect_conf = sum(class_probs.get(l, 0) for l in defect_classes)
        best = max(defect_classes, key=lambda l: class_probs.get(l, 0))

        return VisionResult(
            confidence=round(defect_conf, 4),
            defect_label=best if defect_conf > 0.4 else "Normal / No Defect",
            class_probs=class_probs,
            is_valid_image=True,
        )

    # ── Pretrained heuristic ──────────────────────────────────────────────────

    def _analyse_pretrained(
        self, image: Image.Image, stats: dict[str, float]
    ) -> VisionResult:
        """
        Use ImageNet features + heuristics.
        High edge density + low brightness → likely damaged/dirty environment.
        """
        tensor = _IMAGENET_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(self._device)
        with torch.no_grad():
            features = self._model.features(tensor)
            activation = features.mean(dim=[2, 3]).squeeze().cpu().numpy()

        # Defect proxy: high activation variance in feature maps signals unusual texture
        feature_variance = float(activation.std())
        heuristic_conf = self._heuristic_confidence(stats, feature_variance)

        label = self._map_label_from_stats(stats)
        return VisionResult(
            confidence=round(heuristic_conf, 4),
            defect_label=label,
            class_probs={},
            is_valid_image=True,
        )

    # ── Mock inference ────────────────────────────────────────────────────────

    def _analyse_mock(self, stats: dict[str, float]) -> VisionResult:
        """Deterministic confidence from image statistics only."""
        conf = self._heuristic_confidence(stats, feature_variance=0.5)
        label = self._map_label_from_stats(stats)
        return VisionResult(
            confidence=round(conf, 4),
            defect_label=label,
            class_probs={},
            is_valid_image=True,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _heuristic_confidence(
        self, stats: dict[str, float], feature_variance: float = 0.5
    ) -> float:
        """
        Combine image statistics to estimate defect confidence.

        Rules (approximate):
          - Low brightness (dark images) → likely damage / electrical issue
          - High contrast → visible boundaries / defects
          - High edge density → structural damage / debris
        """
        b, c, e = stats["brightness"], stats["contrast"], stats["edge_density"]

        # Darkness proxy: very bright images (blank walls) are likely not defects
        darkness_score = 1.0 - b if b > 0.5 else 0.6

        raw = 0.30 * darkness_score + 0.35 * min(c, 1.0) + 0.35 * min(e, 1.0)
        # Blend in feature variance if available
        raw = 0.85 * raw + 0.15 * min(feature_variance / 2.0, 1.0)
        return round(min(max(raw, 0.0), 1.0), 4)

    def _map_label_from_stats(self, stats: dict[str, float]) -> str:
        """Pick the most likely defect label from image statistics."""
        b, c, e = stats["brightness"], stats["contrast"], stats["edge_density"]

        if b < 0.25:
            return "Electrical Hazard"
        if e > 0.6:
            return "Structural Damage"
        if c > 0.7 and b > 0.3:
            return "Broken / Damaged Infrastructure"
        if b < 0.4 and c > 0.5:
            return "Dirty / Unsanitary Conditions"
        if e > 0.4:
            return "Water Leakage / Flooding"
        return "Normal / No Defect"
