"""Service d'inférence, découplé du transport (API/WS)."""

import logging
import re
from typing import Any

import numpy as np
from ultralytics import YOLO

from src.config import settings

logger = logging.getLogger(__name__)


def _finger_count(class_name: str) -> int:
    """Extrait le nombre de doigts du nom de classe (``"3"`` ou ``"3_doigts"``)."""
    match = re.search(r"\d+", class_name)
    return int(match.group()) if match else 0


class InferenceService:
    """Charge le modèle YOLO et produit les détections d'une frame."""

    def __init__(self, model: YOLO | None = None) -> None:
        self.model = model if model is not None else self._load_model()

    @staticmethod
    def _load_model() -> YOLO | None:
        """Charge le modèle depuis le disque, ou ``None`` s'il est introuvable."""
        if not settings.MODEL_PATH.exists():
            logger.warning("Modèle introuvable à %s", settings.MODEL_PATH)
            return None
        logger.info("Modèle YOLO chargé depuis %s", settings.MODEL_PATH)
        return YOLO(settings.MODEL_PATH)

    def predict(self, frame: np.ndarray) -> dict[str, Any]:
        """Détecte les mains et agrège le nombre total de doigts visibles."""
        if self.model is None:
            return {"boxes": [], "total_fingers": 0}

        results = self.model(
            frame,
            imgsz=settings.SERVING_IMG_SIZE,
            conf=settings.CONFIDENCE,
            iou=settings.IOU_THRESHOLD,
            max_det=settings.MAX_DET,
            verbose=False,
        )

        detections = []
        total_fingers = 0
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                fingers = _finger_count(result.names[cls])
                total_fingers += fingers
                detections.append(
                    {
                        "bbox": box.xyxy[0].tolist(),
                        "confidence": round(float(box.conf[0]), 2),
                        "class": cls,
                        "fingers": fingers,
                        "label": f"{fingers} fingers",
                    }
                )

        return {"boxes": detections, "total_fingers": total_fingers}
