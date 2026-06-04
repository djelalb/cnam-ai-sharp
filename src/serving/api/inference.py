"""
Service d'inférence découplé du transport (API/WS).
Respecte le principe de responsabilité unique (SRP).
"""

import logging
from typing import Any, Dict

import numpy as np
from ultralytics import YOLO

from src.config import settings

logger = logging.getLogger(__name__)


class InferenceService:
    """Gère le cycle de vie et l'exécution du modèle YOLO."""

    def __init__(self):
        self.model: YOLO | None = None
        self._load_model()

    def _load_model(self) -> None:
        """Charge le modèle en mémoire."""
        try:
            if not settings.MODEL_PATH.exists():
                logger.warning(f"Modèle non trouvé à {settings.MODEL_PATH}")
                return
            self.model = YOLO(settings.MODEL_PATH)
            logger.info(f"Modèle YOLO chargé depuis {settings.MODEL_PATH}")
        except Exception as e:
            logger.error(f"Erreur chargement modèle : {e}")

    def predict(self, frame: np.ndarray) -> Dict[str, Any]:
        """Exécute l'inférence et calcule la logique métier (somme doigts)."""
        if self.model is None:
            return {"boxes": [], "total_fingers": 0}

        results = self.model(frame, conf=settings.CONFIDENCE, verbose=False)
        detections = []
        total_fingers = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                total_fingers += cls
                detections.append(
                    {
                        "bbox": box.xyxy[0].tolist(),
                        "confidence": round(conf, 2),
                        "class": cls,
                    }
                )

        return {"boxes": detections, "total_fingers": total_fingers}
